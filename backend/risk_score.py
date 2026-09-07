"""
Local Risk Score (LRS) Engine — Cyclone Shield AI
Computes hyper-localized composite multi-hazard risk scores based on:
1. Wind Hazard Index (S_wind): Holland vortex decay & gust exposure
2. Rainfall & Surge Hazard Index (S_rain): 24h precipitation & peak coastal storm surge
3. Population Exposure & Vulnerability (S_pop): Density, coastal distance, housing vulnerability

Composite Formula:
LRS = 0.35 * S_wind + 0.35 * S_rain + 0.30 * S_pop
Scale: 0 - 100
"""

import math
from typing import Dict, List, Optional, Any

COASTAL_DISTRICTS_DATA = {
    "Puri": {
        "lat": 19.8135,
        "lon": 85.8312,
        "state": "Odisha",
        "population_density_sqkm": 488,
        "coastal_distance_km": 1.8,
        "kutcha_housing_pct": 42.5,
        "vulnerable_population": 412000,
        "elevation_m": 4.2
    },
    "Jagatsinghpur": {
        "lat": 20.2625,
        "lon": 86.1685,
        "state": "Odisha",
        "population_density_sqkm": 682,
        "coastal_distance_km": 2.5,
        "kutcha_housing_pct": 36.8,
        "vulnerable_population": 365000,
        "elevation_m": 3.8
    },
    "Kendrapara": {
        "lat": 20.4985,
        "lon": 86.4225,
        "state": "Odisha",
        "population_density_sqkm": 545,
        "coastal_distance_km": 3.1,
        "kutcha_housing_pct": 46.2,
        "vulnerable_population": 389000,
        "elevation_m": 3.2
    },
    "Balasore": {
        "lat": 21.4934,
        "lon": 86.9135,
        "state": "Odisha",
        "population_density_sqkm": 609,
        "coastal_distance_km": 5.4,
        "kutcha_housing_pct": 33.1,
        "vulnerable_population": 298000,
        "elevation_m": 6.5
    },
    "Bhadrak": {
        "lat": 21.0543,
        "lon": 86.4957,
        "state": "Odisha",
        "population_density_sqkm": 601,
        "coastal_distance_km": 4.8,
        "kutcha_housing_pct": 38.0,
        "vulnerable_population": 245000,
        "elevation_m": 5.1
    },
    "Ganjam": {
        "lat": 19.3800,
        "lon": 85.0500,
        "state": "Odisha",
        "population_density_sqkm": 429,
        "coastal_distance_km": 6.2,
        "kutcha_housing_pct": 28.5,
        "vulnerable_population": 182000,
        "elevation_m": 8.4
    }
}


def compute_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance in kilometers between two GPS points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_local_risk_score(
    lat: float,
    lon: float,
    district_name: Optional[str] = None,
    eye_lat: float = 18.42,
    eye_lon: float = 86.85,
    storm_max_wind_kph: float = 135.0
) -> Dict[str, Any]:
    """
    Computes a multi-factor Local Risk Score for an exact GPS coordinate or district.
    """
    dist_to_eye_km = compute_haversine_km(lat, lon, eye_lat, eye_lon)

    # 1. Wind Hazard Sub-score (0 - 100)
    # Using modified Rankine / Holland vortex radial decay from cyclone eye
    r_max = 35.0  # km (radius of maximum winds)
    if dist_to_eye_km <= r_max:
        local_wind_kph = storm_max_wind_kph
    else:
        decay = (r_max / dist_to_eye_km) ** 0.58
        local_wind_kph = max(45.0, storm_max_wind_kph * decay)

    # Convert local wind into 0-100 index (60 km/h = 40, 90 km/h = 65, 120 km/h = 90, 140+ = 98)
    wind_subscore = min(100.0, max(10.0, (local_wind_kph / 145.0) * 100.0))

    # 2. Rainfall & Inundation Sub-score (0 - 100)
    # Compounded by rain band proximity and storm surge exposure
    if dist_to_eye_km < 120:
        rain_24h_mm = 280.0 - (dist_to_eye_km * 0.4)
        surge_m = 3.2 - (dist_to_eye_km * 0.005)
    elif dist_to_eye_km < 220:
        rain_24h_mm = 230.0 - ((dist_to_eye_km - 120) * 0.55)
        surge_m = 2.6 - ((dist_to_eye_km - 120) * 0.009)
    elif dist_to_eye_km < 350:
        rain_24h_mm = 175.0 - ((dist_to_eye_km - 220) * 0.5)
        surge_m = max(0.5, 1.7 - ((dist_to_eye_km - 220) * 0.008))
    else:
        rain_24h_mm = max(35.0, 110.0 - ((dist_to_eye_km - 350) * 0.3))
        surge_m = 0.4

    # Surge + Rain Index
    rain_subscore = min(100.0, max(15.0, (rain_24h_mm / 300.0) * 60.0 + (surge_m / 3.5) * 40.0))

    # 3. Population Exposure & Vulnerability Sub-score (0 - 100)
    matched_meta = COASTAL_DISTRICTS_DATA.get(district_name or "")
    if not matched_meta:
        closest_dist = min(
            COASTAL_DISTRICTS_DATA.keys(),
            key=lambda d: compute_haversine_km(lat, lon, COASTAL_DISTRICTS_DATA[d]["lat"], COASTAL_DISTRICTS_DATA[d]["lon"])
        )
        matched_meta = COASTAL_DISTRICTS_DATA[closest_dist]
        if not district_name:
            district_name = closest_dist

    density = matched_meta["population_density_sqkm"]
    coast_dist = matched_meta["coastal_distance_km"]
    kutcha_pct = matched_meta["kutcha_housing_pct"]

    # Density component (up to 35 pts), Coastal proximity (up to 35 pts), Housing fragility (up to 30 pts)
    density_pts = min(35.0, (density / 700.0) * 35.0)
    coast_pts = max(5.0, 35.0 - (coast_dist * 4.5))
    kutcha_pts = (kutcha_pct / 50.0) * 30.0
    pop_subscore = min(100.0, max(20.0, density_pts + coast_pts + kutcha_pts))

    # 4. Composite Local Risk Score
    composite_score = round(0.35 * wind_subscore + 0.35 * rain_subscore + 0.30 * pop_subscore, 1)

    # Classify Tier
    if composite_score >= 80.0:
        tier = "EXTREME RISK"
        color = "#DC2626"
        badge = "badge-red"
        action = "MANDATORY IMMEDIATE EVACUATION: Surge & gale danger within 5 km. Concrete shelter refuge mandatory."
        signal = "Great Danger Signal GD-10"
    elif composite_score >= 60.0:
        tier = "HIGH RISK"
        color = "#EA580C"
        badge = "badge-orange"
        action = "PRE-EVACUATION ACTIVE: Targeted relocation of vulnerable/kutcha dwellers and low-lying coastal wards."
        signal = "Danger Signal D-8"
    elif composite_score >= 40.0:
        tier = "MODERATE RISK"
        color = "#F59E0B"
        badge = "badge-yellow"
        action = "COASTAL WATCH & PREPAREDNESS: Secure structures, verify generator fuel and drinking water supplies."
        signal = "Local Cautionary Signal LC-3"
    else:
        tier = "LOW RISK"
        color = "#10B981"
        badge = "badge-green"
        action = "ADVISORY MONITORING: Normal tracking surveillance with fishermen warnings active."
        signal = "Information Bulletin"

    return {
        "location": {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "district": district_name or "Coastal Sector",
            "distance_to_storm_eye_km": round(dist_to_eye_km, 1),
            "distance_to_coast_km": round(coast_dist, 1)
        },
        "composite_risk_score": composite_score,
        "risk_tier": tier,
        "badge_class": badge,
        "color_hex": color,
        "operational_directive": action,
        "warning_signal": signal,
        "factors": {
            "wind_hazard": {
                "score": round(wind_subscore, 1),
                "weight_pct": 35,
                "local_wind_speed_kph": round(local_wind_kph, 1),
                "peak_gust_kph": round(local_wind_kph * 1.22, 1),
                "hazard_level": "Destructive Hurricane Gale" if local_wind_kph >= 115 else ("Storm Force" if local_wind_kph >= 85 else "Gale Wind")
            },
            "rainfall_inundation": {
                "score": round(rain_subscore, 1),
                "weight_pct": 35,
                "projected_rain_24h_mm": round(rain_24h_mm, 1),
                "projected_surge_height_m": round(surge_m, 2),
                "hazard_level": "Extremely Severe Inundation" if surge_m >= 2.5 else ("Severe Surge Risk" if surge_m >= 1.5 else "Moderate Tidal Wash")
            },
            "population_exposure": {
                "score": round(pop_subscore, 1),
                "weight_pct": 30,
                "density_per_sqkm": density,
                "kutcha_dwellings_pct": kutcha_pct,
                "vulnerable_population": matched_meta["vulnerable_population"],
                "hazard_level": "High Demographic Density & Shoreline Exposure"
            }
        }
    }


def get_all_districts_risk_matrix(
    eye_lat: float = 18.42,
    eye_lon: float = 86.85,
    max_wind_kph: float = 135.0
) -> List[Dict[str, Any]]:
    """
    Returns the comprehensive District Vulnerability Matrix enriched with Local Risk Scores (0-100).
    """
    matrix = []
    for d_name, meta in COASTAL_DISTRICTS_DATA.items():
        res = calculate_local_risk_score(
            lat=meta["lat"],
            lon=meta["lon"],
            district_name=d_name,
            eye_lat=eye_lat,
            eye_lon=eye_lon,
            storm_max_wind_kph=max_wind_kph
        )

        matrix.append({
            "district": d_name,
            "state": meta["state"],
            "local_risk_score": res["composite_risk_score"],
            "risk_level": res["risk_tier"],
            "badge_class": res["badge_class"],
            "color_hex": res["color_hex"],
            "distance_km": res["location"]["distance_to_storm_eye_km"],
            "wind_forecast_kph": f"{int(res['factors']['wind_hazard']['local_wind_speed_kph'])}-{int(res['factors']['wind_hazard']['peak_gust_kph'])}",
            "wind_score": res["factors"]["wind_hazard"]["score"],
            "surge_m": f"{res['factors']['rainfall_inundation']['projected_surge_height_m']:.1f}",
            "rainfall_mm": f"{int(res['factors']['rainfall_inundation']['projected_rain_24h_mm'])}",
            "rain_score": res["factors"]["rainfall_inundation"]["score"],
            "pop_density": meta["population_density_sqkm"],
            "pop_score": res["factors"]["population_exposure"]["score"],
            "vulnerable_population": meta["vulnerable_population"],
            "evacuation_status": "Mandatory Evacuation" if res["composite_risk_score"] >= 80 else ("Targeted Relocation" if res["composite_risk_score"] >= 60 else "Advisory Vigil"),
            "warning_signal": res["warning_signal"],
            "lat": meta["lat"],
            "lon": meta["lon"]
        })

    # Sort descending by risk score
    matrix.sort(key=lambda x: x["local_risk_score"], reverse=True)
    return matrix
