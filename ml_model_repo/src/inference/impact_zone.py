"""
Cyclone Horizon — Impact Zone Estimation
Simplified Holland vortex wind-radius buffer around predicted track.
Not a full storm-surge model — that's out of scope and stated honestly.
"""

import numpy as np
from typing import Dict, List, Tuple

from src.utils.constants import KT_TO_KPH, wind_to_imd_category
from src.utils.geo import destination_point, haversine_distance
from src.utils.logging_config import get_logger

logger = get_logger("inference.impact_zone")


# Approximate wind radii by IMD category (km) — simplified from operational tables
# Format: {category_code: (radius_hurricane_force, radius_storm_force, radius_gale_force)}
WIND_RADII_KM = {
    "LPA":  (0, 0, 50),
    "D":    (0, 0, 100),
    "DD":   (0, 50, 150),
    "CS":   (0, 100, 200),
    "SCS":  (50, 150, 300),
    "VSCS": (100, 200, 400),
    "ESCS": (150, 250, 450),
    "SuCS": (200, 300, 500),
}


def estimate_impact_zone(
    lat: float,
    lon: float,
    wind_kt: float,
    n_points: int = 36,
) -> Dict:
    """
    Estimate the impact zone around a cyclone position.
    
    Uses a simplified Holland vortex profile approximation:
    concentric rings at hurricane-force, storm-force, and gale-force
    wind radii, scaled by the predicted max wind.
    
    Parameters
    ----------
    lat, lon : cyclone center position (degrees)
    wind_kt : max sustained wind (knots)
    n_points : polygon resolution (points per ring)
    
    Returns
    -------
    Dict with concentric impact zones as GeoJSON-compatible polygons
    """
    cat = wind_to_imd_category(wind_kt)
    radii = WIND_RADII_KM.get(cat.code, (0, 0, 100))
    
    zones = {}
    zone_names = ["hurricane_force", "storm_force", "gale_force"]
    zone_colors = ["#DC2626", "#F97316", "#FBBF24"]  # red, orange, yellow
    
    for i, (name, color, radius) in enumerate(zip(zone_names, zone_colors, radii)):
        if radius <= 0:
            continue
        
        # Generate polygon ring
        points = []
        for angle in np.linspace(0, 360, n_points, endpoint=False):
            plat, plon = destination_point(lat, lon, angle, radius)
            points.append([round(plon, 4), round(plat, 4)])
        points.append(points[0])  # close the ring
        
        zones[name] = {
            "type": "Feature",
            "properties": {
                "zone_type": name,
                "radius_km": radius,
                "color": color,
                "wind_threshold": ["hurricane", "storm", "gale"][i],
                "imd_category": cat.name,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [points],
            },
        }
    
    return {
        "center": {"lat": lat, "lon": lon},
        "wind_kt": wind_kt,
        "wind_kph": round(wind_kt * KT_TO_KPH, 0),
        "imd_category": cat.name,
        "imd_code": cat.code,
        "zones": zones,
    }


def estimate_track_impact_corridor(
    track_predictions: List[Dict],
    buffer_km: float = 50.0,
) -> Dict:
    """
    Estimate the full impact corridor along the predicted track.
    
    Creates a buffered polygon around the predicted track path,
    expanding with lead time to reflect growing uncertainty.
    
    Parameters
    ----------
    track_predictions : list of {lat, lon, wind_kt, lead_h, ...}
    buffer_km : base buffer radius (expands with lead time)
    
    Returns
    -------
    GeoJSON-compatible polygon for the impact corridor
    """
    if not track_predictions:
        return {}
    
    left_side = []
    right_side = []
    
    for i, pred in enumerate(track_predictions):
        lat, lon = pred["lat"], pred["lon"]
        lead_h = pred.get("lead_h", (i + 1) * 3)
        
        # Buffer grows with lead time
        radius = buffer_km + 5 * (lead_h / 3)  # ~5km per 3h step
        
        # Get track bearing for perpendicular buffer
        if i < len(track_predictions) - 1:
            from src.utils.geo import bearing
            brng = bearing(lat, lon, track_predictions[i+1]["lat"], track_predictions[i+1]["lon"])
        elif i > 0:
            from src.utils.geo import bearing
            brng = bearing(track_predictions[i-1]["lat"], track_predictions[i-1]["lon"], lat, lon)
        else:
            brng = 0
        
        # Points perpendicular to track
        left_lat, left_lon = destination_point(lat, lon, (brng - 90) % 360, radius)
        right_lat, right_lon = destination_point(lat, lon, (brng + 90) % 360, radius)
        
        left_side.append([round(left_lon, 4), round(left_lat, 4)])
        right_side.append([round(right_lon, 4), round(right_lat, 4)])
    
    # Create corridor polygon (left side forward, right side backward)
    corridor_points = left_side + list(reversed(right_side))
    corridor_points.append(corridor_points[0])  # close polygon
    
    return {
        "type": "Feature",
        "properties": {
            "zone_type": "impact_corridor",
            "base_buffer_km": buffer_km,
            "color": "#EF4444",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [corridor_points],
        },
    }
