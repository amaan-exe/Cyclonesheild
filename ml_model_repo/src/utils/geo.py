"""
Cyclone Horizon — Geospatial Utilities
Great-circle distance, bearing, bounding boxes, grid operations.
All angles in degrees, distances in km unless stated otherwise.
"""

import numpy as np
from typing import Tuple

from src.utils.constants import EARTH_RADIUS_KM


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points (degrees → km).
    
    Parameters
    ----------
    lat1, lon1 : float — point 1 coordinates in degrees
    lat2, lon2 : float — point 2 coordinates in degrees
    
    Returns
    -------
    float — distance in kilometers
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial bearing from point 1 to point 2 (degrees, 0=North, clockwise).
    
    Returns
    -------
    float — bearing in degrees [0, 360)
    """
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2_r - lon1_r
    x = np.sin(dlon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)
    
    bearing_rad = np.arctan2(x, y)
    return (np.degrees(bearing_rad) + 360) % 360


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    """
    Compute destination point given start, bearing, and distance.
    
    Parameters
    ----------
    lat, lon : float — start point (degrees)
    bearing_deg : float — bearing in degrees (0=North, clockwise)
    distance_km : float — distance to travel
    
    Returns
    -------
    (lat2, lon2) — destination point in degrees
    """
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    brng_r = np.radians(bearing_deg)
    d_r = distance_km / EARTH_RADIUS_KM
    
    lat2_r = np.arcsin(
        np.sin(lat_r) * np.cos(d_r) +
        np.cos(lat_r) * np.sin(d_r) * np.cos(brng_r)
    )
    lon2_r = lon_r + np.arctan2(
        np.sin(brng_r) * np.sin(d_r) * np.cos(lat_r),
        np.cos(d_r) - np.sin(lat_r) * np.sin(lat2_r)
    )
    
    return float(np.degrees(lat2_r)), float(np.degrees(lon2_r))


def bounding_box(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Compute a lat/lon bounding box around a center point.
    
    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) in degrees
    """
    # Approximate: 1 degree lat ≈ 111 km
    delta_lat = radius_km / 111.0
    delta_lon = radius_km / (111.0 * np.cos(np.radians(lat)))
    
    return (
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )


def km_to_degrees_lat(km: float) -> float:
    """Convert kilometers to degrees latitude (approximate)."""
    return km / 111.0


def km_to_degrees_lon(km: float, lat: float) -> float:
    """Convert kilometers to degrees longitude at a given latitude."""
    return km / (111.0 * np.cos(np.radians(lat)))


def compute_storm_motion(
    lats: np.ndarray,
    lons: np.ndarray,
    times_hours: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute storm motion (speed in kph, bearing in degrees) from
    consecutive best-track fixes. Uses ONLY past fixes (causal).
    
    Parameters
    ----------
    lats, lons : arrays of lat/lon (degrees)
    times_hours : array of hours since some reference
    
    Returns
    -------
    speeds_kph : array of motion speeds (kph), first element is NaN
    bearings_deg : array of motion bearings (degrees), first element is NaN
    """
    n = len(lats)
    speeds = np.full(n, np.nan)
    bearings_out = np.full(n, np.nan)
    
    for i in range(1, n):
        dt = times_hours[i] - times_hours[i - 1]
        if dt <= 0:
            continue
        dist = haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
        speeds[i] = dist / dt  # km/h
        bearings_out[i] = bearing(lats[i - 1], lons[i - 1], lats[i], lons[i])
    
    return speeds, bearings_out
