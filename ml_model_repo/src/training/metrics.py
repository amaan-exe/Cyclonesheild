"""
Cyclone Horizon — Evaluation Metrics
Track error, intensity error, skill scores, calibration.
All metrics directly comparable to published IMD/JTWC verification statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.utils.geo import haversine_distance
from src.utils.constants import EVAL_LEAD_TIMES_H, wind_to_imd_index
from src.utils.logging_config import get_logger

logger = get_logger("training.metrics")


def track_error_km(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance error between predicted and actual positions.
    
    Parameters
    ----------
    pred_lat, pred_lon : predicted coordinates (degrees)
    true_lat, true_lon : actual coordinates (degrees)
    
    Returns
    -------
    np.ndarray : distance errors in km
    """
    errors = np.array([
        haversine_distance(plat, plon, tlat, tlon)
        for plat, plon, tlat, tlon in zip(pred_lat, pred_lon, true_lat, true_lon)
    ])
    return errors


def track_error_by_lead_time(
    predictions: pd.DataFrame,
    lead_time_col: str = "lead_time_hours",
    lead_times: List[int] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute mean/median track error at each lead time.
    Directly comparable to IMD CPS published numbers:
    - 74 km at 12h, 200 km at 72h (IMD CPS 2009-2013)
    
    Returns
    -------
    Dict of lead_time → {"mean_km": ..., "median_km": ..., "n": ...}
    """
    if lead_times is None:
        lead_times = EVAL_LEAD_TIMES_H
    
    results = {}
    for lt in lead_times:
        mask = predictions[lead_time_col] == lt
        subset = predictions[mask]
        
        if len(subset) == 0:
            continue
        
        errors = track_error_km(
            subset["pred_lat"].values,
            subset["pred_lon"].values,
            subset["true_lat"].values,
            subset["true_lon"].values,
        )
        
        results[lt] = {
            "mean_km": float(np.mean(errors)),
            "median_km": float(np.median(errors)),
            "std_km": float(np.std(errors)),
            "n": len(errors),
        }
    
    return results


def intensity_error(
    pred_wind: np.ndarray,
    true_wind: np.ndarray,
    pred_pressure: np.ndarray = None,
    true_pressure: np.ndarray = None,
) -> Dict[str, float]:
    """
    Intensity forecast error metrics.
    
    Returns
    -------
    Dict with wind MAE (kt), pressure MAE (hPa)
    """
    result = {
        "wind_mae_kt": float(np.nanmean(np.abs(pred_wind - true_wind))),
        "wind_rmse_kt": float(np.sqrt(np.nanmean((pred_wind - true_wind) ** 2))),
        "wind_bias_kt": float(np.nanmean(pred_wind - true_wind)),
    }
    
    if pred_pressure is not None and true_pressure is not None:
        result["pressure_mae_hpa"] = float(np.nanmean(np.abs(pred_pressure - true_pressure)))
        result["pressure_rmse_hpa"] = float(np.sqrt(np.nanmean((pred_pressure - true_pressure) ** 2)))
    
    return result


def category_accuracy(
    pred_wind: np.ndarray,
    true_wind: np.ndarray,
    tolerance: int = 1,
) -> Dict[str, float]:
    """
    IMD category classification accuracy.
    
    Parameters
    ----------
    pred_wind, true_wind : max wind in knots
    tolerance : allowed category offset (1 = within ±1 category)
    
    Returns
    -------
    Dict with exact accuracy and within-tolerance accuracy
    """
    pred_cats = np.array([wind_to_imd_index(w) for w in pred_wind])
    true_cats = np.array([wind_to_imd_index(w) for w in true_wind])
    
    exact = np.mean(pred_cats == true_cats)
    within_tol = np.mean(np.abs(pred_cats - true_cats) <= tolerance)
    
    return {
        "exact_accuracy": float(exact),
        f"within_{tolerance}_category": float(within_tol),
    }


def cliper_baseline(
    df: pd.DataFrame,
    lead_time_h: int = 24,
) -> Dict[str, float]:
    """
    CLIPER-style persistence baseline.
    Assumes the storm continues at its current speed and direction.
    
    This is the minimum baseline — if our model can't beat persistence,
    it's not adding value.
    
    Parameters
    ----------
    df : DataFrame with lat, lon, storm_speed_kph, storm_bearing_deg
    lead_time_h : forecast lead time
    
    Returns
    -------
    Dict with mean/median track error of the persistence forecast
    """
    from src.utils.geo import destination_point
    
    errors = []
    
    for _, row in df.iterrows():
        if pd.isna(row.get("storm_speed_kph")) or pd.isna(row.get("storm_bearing_deg")):
            continue
        
        # Persistence: extrapolate current motion
        dist = row["storm_speed_kph"] * lead_time_h
        pred_lat, pred_lon = destination_point(
            row["lat"], row["lon"],
            row["storm_bearing_deg"], dist
        )
        
        # Need actual position at lead_time_h (ground truth)
        if f"true_lat_{lead_time_h}h" in df.columns:
            true_lat = row[f"true_lat_{lead_time_h}h"]
            true_lon = row[f"true_lon_{lead_time_h}h"]
            if pd.notna(true_lat) and pd.notna(true_lon):
                err = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
                errors.append(err)
    
    if not errors:
        return {"mean_km": float("nan"), "median_km": float("nan"), "n": 0}
    
    return {
        "mean_km": float(np.mean(errors)),
        "median_km": float(np.median(errors)),
        "n": len(errors),
    }


def skill_score(model_error: float, baseline_error: float) -> float:
    """
    Skill score relative to a baseline.
    
    SS = 1 - (model_error / baseline_error)
    
    SS > 0 means the model outperforms the baseline.
    SS = 1 means perfect forecast.
    SS <= 0 means no skill.
    """
    if baseline_error == 0 or np.isnan(baseline_error):
        return float("nan")
    return 1.0 - (model_error / baseline_error)


def cone_calibration(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    actual: np.ndarray,
    confidence_levels: List[float] = [0.5, 0.68, 0.95],
) -> Dict[float, float]:
    """
    Check if the actual track falls within the predicted X% confidence cone
    X% of the time (reliability diagram).
    
    A well-calibrated model: the 95% cone should contain the truth ~95% of the time.
    
    Returns
    -------
    Dict of confidence_level → actual containment fraction
    """
    from scipy import stats
    
    results = {}
    for level in confidence_levels:
        z = stats.norm.ppf(0.5 + level / 2)
        lower = pred_mean - z * pred_std
        upper = pred_mean + z * pred_std
        
        contained = np.mean((actual >= lower) & (actual <= upper))
        results[level] = float(contained)
    
    return results
