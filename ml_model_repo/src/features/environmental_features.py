"""
Cyclone Horizon — Environmental Feature Engineering
SST, vertical wind shear, storm motion — all computed CAUSALLY (past-only).

Every feature must be computable using ONLY information available at or before
the timestamp it's attached to. No centered differences, no future data.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.utils.geo import haversine_distance, bearing, compute_storm_motion
from src.utils.logging_config import get_logger

logger = get_logger("features.environmental")


def compute_storm_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute storm motion vector (speed + bearing) from consecutive best-track fixes.
    Uses ONLY past fixes — never a centered or future-aware derivative.
    
    Adds columns: storm_speed_kph, storm_bearing_deg
    """
    df = df.copy()
    df["storm_speed_kph"] = np.nan
    df["storm_bearing_deg"] = np.nan
    
    for sid, group in df.groupby("storm_id"):
        idx = group.index
        lats = group["lat"].values
        lons = group["lon"].values
        times = group["timestamp"].values
        
        # Convert timestamps to hours
        t0 = times[0]
        hours = np.array([(t - t0) / np.timedelta64(1, 'h') for t in times])
        
        speeds, bearings = compute_storm_motion(lats, lons, hours)
        
        df.loc[idx, "storm_speed_kph"] = speeds
        df.loc[idx, "storm_bearing_deg"] = bearings
    
    logger.info(f"Storm motion features computed for {df['storm_id'].nunique()} storms")
    return df


def compute_intensity_trends(
    df: pd.DataFrame,
    windows_hours: list = [12, 24],
) -> pd.DataFrame:
    """
    Compute rolling intensity trends (CAUSAL: backward-looking only).
    
    For each window size, computes:
    - trend_wind_{w}h: wind change over past w hours
    - trend_pressure_{w}h: pressure change over past w hours
    - mean_wind_{w}h: rolling mean wind over past w hours
    
    These are explicit trend features, not just raw sequences.
    """
    df = df.copy()
    
    for window_h in windows_hours:
        # Approximate number of steps (3-hourly data)
        n_steps = max(1, window_h // 3)
        
        for sid, group in df.groupby("storm_id"):
            idx = group.index
            wind = group["max_wind_kt"].values
            pres = group["min_pressure_hpa"].values
            
            for i in range(len(idx)):
                start = max(0, i - n_steps)
                
                if i > start:
                    df.loc[idx[i], f"trend_wind_{window_h}h"] = wind[i] - wind[start]
                    df.loc[idx[i], f"trend_pressure_{window_h}h"] = pres[i] - pres[start]
                    df.loc[idx[i], f"mean_wind_{window_h}h"] = np.nanmean(wind[start:i+1])
                else:
                    df.loc[idx[i], f"trend_wind_{window_h}h"] = np.nan
                    df.loc[idx[i], f"trend_pressure_{window_h}h"] = np.nan
                    df.loc[idx[i], f"mean_wind_{window_h}h"] = np.nan
    
    logger.info(f"Intensity trend features computed for windows: {windows_hours}h")
    return df


def compute_all_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: compute all environmental features.
    Adds: storm motion, intensity trends, Coriolis proxy.
    
    SST and shear would come from ERA5/OISST if available;
    we create placeholder columns with NaN if the data isn't loaded yet.
    """
    # Storm motion
    df = compute_storm_motion_features(df)
    
    # Intensity trends
    df = compute_intensity_trends(df, windows_hours=[12, 24])
    
    # Coriolis parameter proxy (latitude)
    df["coriolis_proxy"] = df["lat"]  # f = 2Ω sin(φ), but lat is a good proxy
    
    # SST placeholder (filled when OISST data available)
    if "sst" not in df.columns:
        df["sst"] = np.nan
        logger.info("SST column initialized as NaN (OISST data not yet loaded)")
    
    # Shear placeholder (filled when ERA5 data available)
    if "shear_magnitude" not in df.columns:
        df["shear_magnitude"] = np.nan
        logger.info("Shear magnitude column initialized as NaN (ERA5 data not yet loaded)")
    
    if "relative_humidity_500" not in df.columns:
        df["relative_humidity_500"] = np.nan
    
    return df
