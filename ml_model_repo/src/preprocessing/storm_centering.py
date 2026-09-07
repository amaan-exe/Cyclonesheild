"""
Cyclone Horizon — Storm Centering
Crops satellite imagery around the storm center using best-track coordinates.
Handles edge-of-image padding gracefully.
"""

import numpy as np
from typing import Tuple

from src.utils.constants import EARTH_RADIUS_KM
from src.utils.geo import km_to_degrees_lat, km_to_degrees_lon
from src.utils.logging_config import get_logger

logger = get_logger("preprocessing.storm_centering")


def center_on_storm(
    image: np.ndarray,
    image_lat_range: Tuple[float, float],
    image_lon_range: Tuple[float, float],
    storm_lat: float,
    storm_lon: float,
    window_km: float = 601.0,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, dict]:
    """
    Center an image on the storm and crop to a fixed window.
    
    Parameters
    ----------
    image : np.ndarray — 2D array of brightness temperature or reflectance
    image_lat_range : (lat_min, lat_max) of the full image
    image_lon_range : (lon_min, lon_max) of the full image
    storm_lat : best-track latitude of storm center
    storm_lon : best-track longitude of storm center
    window_km : side length of the crop window in km (default 601)
    fill_value : value for padded pixels when storm is near edge
    
    Returns
    -------
    cropped : np.ndarray — cropped image centered on storm
    metadata : dict — crop coordinates and pixel scale
    
    Edge cases handled:
    - Storm center near image boundary → pad with fill_value
    - Storm center outside image → returns padded image with warning
    """
    h, w = image.shape[:2]
    
    # Pixel resolution (degrees per pixel)
    lat_res = (image_lat_range[1] - image_lat_range[0]) / h
    lon_res = (image_lon_range[1] - image_lon_range[0]) / w
    
    # Window size in degrees
    half_window_lat = km_to_degrees_lat(window_km / 2)
    half_window_lon = km_to_degrees_lon(window_km / 2, storm_lat)
    
    # Desired crop bounds in degrees
    crop_lat_min = storm_lat - half_window_lat
    crop_lat_max = storm_lat + half_window_lat
    crop_lon_min = storm_lon - half_window_lon
    crop_lon_max = storm_lon + half_window_lon
    
    # Convert to pixel indices
    row_center = int((image_lat_range[1] - storm_lat) / lat_res)  # lat decreases with row
    col_center = int((storm_lon - image_lon_range[0]) / lon_res)
    
    half_h = int(half_window_lat / lat_res)
    half_w = int(half_window_lon / lon_res)
    
    row_start = row_center - half_h
    row_end = row_center + half_h
    col_start = col_center - half_w
    col_end = col_center + half_w
    
    # Create output array (padded)
    crop_h = row_end - row_start
    crop_w = col_end - col_start
    
    if crop_h <= 0 or crop_w <= 0:
        logger.warning(f"Invalid crop dimensions: {crop_h}×{crop_w}")
        crop_h = max(1, crop_h)
        crop_w = max(1, crop_w)
    
    cropped = np.full((crop_h, crop_w), fill_value, dtype=image.dtype)
    
    # Compute valid source region
    src_row_start = max(0, row_start)
    src_row_end = min(h, row_end)
    src_col_start = max(0, col_start)
    src_col_end = min(w, col_end)
    
    # Compute destination region in cropped image
    dst_row_start = src_row_start - row_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - col_start
    dst_col_end = dst_col_start + (src_col_end - src_col_start)
    
    # Copy valid pixels
    if (src_row_end > src_row_start and src_col_end > src_col_start and
        dst_row_end <= crop_h and dst_col_end <= crop_w):
        cropped[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
            image[src_row_start:src_row_end, src_col_start:src_col_end]
    else:
        logger.warning(
            f"Storm center ({storm_lat:.2f}, {storm_lon:.2f}) mostly outside image bounds. "
            f"Image: lat=[{image_lat_range[0]:.2f}, {image_lat_range[1]:.2f}], "
            f"lon=[{image_lon_range[0]:.2f}, {image_lon_range[1]:.2f}]"
        )
    
    metadata = {
        "storm_lat": storm_lat,
        "storm_lon": storm_lon,
        "window_km": window_km,
        "pixel_lat_res": lat_res,
        "pixel_lon_res": lon_res,
        "crop_bounds_deg": {
            "lat_min": crop_lat_min, "lat_max": crop_lat_max,
            "lon_min": crop_lon_min, "lon_max": crop_lon_max,
        },
        "padded_pixels": int(np.sum(np.isnan(cropped) if np.isnan(fill_value) else cropped == fill_value)),
    }
    
    return cropped, metadata


def normalize(
    image: np.ndarray,
    method: str = "zscore",
    stats: dict = None,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize image values.
    
    CRITICAL: stats must be computed from TRAIN SPLIT ONLY.
    Never compute from the full dataset — this leaks test-set info.
    
    Parameters
    ----------
    image : np.ndarray
    method : "zscore" (z-score normalize) or "minmax" ([0,1] scale)
    stats : pre-computed {"mean": float, "std": float} or {"min": float, "max": float}
            If None, computed from this image (only valid during training stats computation)
    
    Returns
    -------
    normalized : np.ndarray
    stats_used : dict — the stats used (save these for val/test normalization)
    """
    valid_mask = ~np.isnan(image)
    
    if method == "zscore":
        if stats is None:
            mean = float(np.nanmean(image))
            std = float(np.nanstd(image))
            stats = {"mean": mean, "std": std}
        
        std = stats["std"]
        if std < 1e-8:
            std = 1.0  # avoid division by zero
        
        normalized = (image - stats["mean"]) / std
    
    elif method == "minmax":
        if stats is None:
            vmin = float(np.nanmin(image))
            vmax = float(np.nanmax(image))
            stats = {"min": vmin, "max": vmax}
        
        range_val = stats["max"] - stats["min"]
        if range_val < 1e-8:
            range_val = 1.0
        
        normalized = (image - stats["min"]) / range_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats
