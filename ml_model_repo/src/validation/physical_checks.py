"""
Cyclone Horizon — Physical Plausibility & QC Checks
Validates track data against physical bounds before it enters feature engineering.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.constants import (
    BRIGHTNESS_TEMP_MIN_K, BRIGHTNESS_TEMP_MAX_K,
    MAX_WIND_KT, MIN_PRESSURE_HPA, MAX_PRESSURE_HPA,
    NIO_BOUNDS,
)
from src.utils.logging_config import get_logger

logger = get_logger("validation.physical_checks")


def check_wind_bounds(df: pd.DataFrame, col: str = "max_wind_kt") -> pd.Series:
    """
    Flag records where max wind is outside [0, MAX_WIND_KT].
    
    Returns
    -------
    pd.Series[bool] — True for records that PASS the check
    """
    valid = (df[col] >= 0) & (df[col] <= MAX_WIND_KT)
    n_fail = (~valid & df[col].notna()).sum()
    if n_fail > 0:
        logger.warning(f"Wind bounds check: {n_fail} records outside [0, {MAX_WIND_KT}] kt")
    return valid | df[col].isna()  # NaN = no data, not invalid


def check_pressure_bounds(df: pd.DataFrame, col: str = "min_pressure_hpa") -> pd.Series:
    """
    Flag records where pressure is outside [MIN_PRESSURE_HPA, MAX_PRESSURE_HPA].
    
    Returns
    -------
    pd.Series[bool] — True for records that PASS
    """
    valid = (df[col] >= MIN_PRESSURE_HPA) & (df[col] <= MAX_PRESSURE_HPA)
    n_fail = (~valid & df[col].notna()).sum()
    if n_fail > 0:
        logger.warning(f"Pressure bounds check: {n_fail} records outside [{MIN_PRESSURE_HPA}, {MAX_PRESSURE_HPA}] hPa")
    return valid | df[col].isna()


def check_basin_bounds(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    bounds: Dict = None,
) -> pd.Series:
    """
    Flag records where lat/lon is outside the expected basin bounding box.
    
    Returns
    -------
    pd.Series[bool] — True for records that PASS
    """
    if bounds is None:
        bounds = NIO_BOUNDS
    
    valid = (
        (df[lat_col] >= bounds["lat_min"]) &
        (df[lat_col] <= bounds["lat_max"]) &
        (df[lon_col] >= bounds["lon_min"]) &
        (df[lon_col] <= bounds["lon_max"])
    )
    n_fail = (~valid).sum()
    if n_fail > 0:
        logger.warning(f"Basin bounds check: {n_fail} records outside NIO bounds")
    return valid


def check_brightness_temperature(image: np.ndarray) -> Tuple[bool, str]:
    """
    Check if brightness temperature values are physically plausible.
    
    Parameters
    ----------
    image : np.ndarray — brightness temperature in Kelvin
    
    Returns
    -------
    (passes: bool, message: str)
    """
    valid_mask = ~np.isnan(image)
    if not valid_mask.any():
        return False, "All NaN values"
    
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    
    if min_val < BRIGHTNESS_TEMP_MIN_K or max_val > BRIGHTNESS_TEMP_MAX_K:
        return False, f"BT out of range: [{min_val:.1f}, {max_val:.1f}] K (expected [{BRIGHTNESS_TEMP_MIN_K}, {BRIGHTNESS_TEMP_MAX_K}])"
    
    return True, "OK"


def check_temporal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for temporal inconsistencies within each storm:
    - Non-monotonic timestamps
    - Unrealistically large jumps in position (>500km in 3 hours)
    
    Returns
    -------
    DataFrame with added 'temporal_flag' column
    """
    from src.utils.geo import haversine_distance
    
    df = df.copy()
    df["temporal_flag"] = "OK"
    
    for sid, group in df.groupby("storm_id"):
        idx = group.index
        times = group["timestamp"].values
        lats = group["lat"].values
        lons = group["lon"].values
        
        for i in range(1, len(idx)):
            # Check monotonicity
            if times[i] <= times[i-1]:
                df.loc[idx[i], "temporal_flag"] = "NON_MONOTONIC"
                continue
            
            # Check position jump
            dt_hours = (times[i] - times[i-1]) / np.timedelta64(1, 'h')
            if dt_hours > 0:
                dist = haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
                speed_kph = dist / dt_hours
                if speed_kph > 150:  # unrealistically fast
                    df.loc[idx[i], "temporal_flag"] = f"FAST_JUMP_{speed_kph:.0f}kph"
    
    n_flags = (df["temporal_flag"] != "OK").sum()
    if n_flags > 0:
        logger.warning(f"Temporal consistency: {n_flags} flagged records")
    
    return df


def run_all_track_checks(df: pd.DataFrame) -> Dict[str, int]:
    """
    Run all QC checks on a track DataFrame and return a summary.
    
    Returns
    -------
    Dict with check names → number of failures
    """
    summary = {}
    
    wind_ok = check_wind_bounds(df)
    summary["wind_out_of_bounds"] = int((~wind_ok).sum())
    
    pres_ok = check_pressure_bounds(df)
    summary["pressure_out_of_bounds"] = int((~pres_ok).sum())
    
    basin_ok = check_basin_bounds(df)
    summary["outside_basin"] = int((~basin_ok).sum())
    
    df_temp = check_temporal_consistency(df)
    summary["temporal_flags"] = int((df_temp["temporal_flag"] != "OK").sum())
    
    summary["total_records"] = len(df)
    summary["total_storms"] = int(df["storm_id"].nunique())
    summary["records_with_wind"] = int(df["max_wind_kt"].notna().sum())
    summary["records_with_pressure"] = int(df["min_pressure_hpa"].notna().sum())
    
    logger.info(f"QC summary: {summary}")
    return summary
