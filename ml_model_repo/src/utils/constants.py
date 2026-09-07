"""
Cyclone Horizon — Constants & Definitions
IMD operational scale, physical bounds, basin coordinates, Dvorak mapping.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


# ============================================================================
# IMD Intensity Categories
# ============================================================================

@dataclass(frozen=True)
class IMDCategory:
    """One row of the IMD tropical cyclone intensity scale."""
    name: str
    code: str
    wind_min_kt: int
    wind_max_kt: int
    index: int  # 0-7 for model output ordering

IMD_CATEGORIES: List[IMDCategory] = [
    IMDCategory("Low Pressure Area",                "LPA",  0,   17,  0),
    IMDCategory("Depression",                       "D",    17,  27,  1),
    IMDCategory("Deep Depression",                  "DD",   28,  33,  2),
    IMDCategory("Cyclonic Storm",                   "CS",   34,  47,  3),
    IMDCategory("Severe Cyclonic Storm",            "SCS",  48,  63,  4),
    IMDCategory("Very Severe Cyclonic Storm",       "VSCS", 64,  89,  5),
    IMDCategory("Extremely Severe Cyclonic Storm",  "ESCS", 90,  119, 6),
    IMDCategory("Super Cyclonic Storm",             "SuCS", 120, 999, 7),
]

IMD_CATEGORY_NAMES = [c.name for c in IMD_CATEGORIES]
IMD_CATEGORY_CODES = [c.code for c in IMD_CATEGORIES]
NUM_IMD_CATEGORIES = len(IMD_CATEGORIES)


def wind_to_imd_category(wind_kt: float) -> IMDCategory:
    """Map max sustained wind (knots) to IMD category."""
    for cat in reversed(IMD_CATEGORIES):
        if wind_kt >= cat.wind_min_kt:
            return cat
    return IMD_CATEGORIES[0]


def wind_to_imd_index(wind_kt: float) -> int:
    """Map max sustained wind (knots) to IMD category index (0-7)."""
    return wind_to_imd_category(wind_kt).index


# ============================================================================
# Dvorak T-Number Mapping
# ============================================================================

# T-number → approximate max wind (knots), from Dvorak (1984) Atlantic table
DVORAK_T_TO_WIND_KT: Dict[float, float] = {
    1.0: 25,  1.5: 25,  2.0: 30,  2.5: 35,  3.0: 45,
    3.5: 55,  4.0: 65,  4.5: 77,  5.0: 90,  5.5: 102,
    6.0: 115, 6.5: 127, 7.0: 140, 7.5: 155, 8.0: 170,
}

DVORAK_PATTERN_TYPES = [
    "curved_band",
    "shear",
    "eye",
    "cdo",               # Central Dense Overcast
    "embedded_center",
    "annular",           # Annular hurricane pattern
]
NUM_DVORAK_PATTERNS = len(DVORAK_PATTERN_TYPES)


def t_number_to_wind_kt(t_number: float) -> float:
    """Interpolate Dvorak T-number to estimated max wind (kt)."""
    t_vals = sorted(DVORAK_T_TO_WIND_KT.keys())
    w_vals = [DVORAK_T_TO_WIND_KT[t] for t in t_vals]
    return float(np.interp(t_number, t_vals, w_vals))


# ============================================================================
# Physical Plausibility Bounds (for QC validation)
# ============================================================================

BRIGHTNESS_TEMP_MIN_K = 170.0
BRIGHTNESS_TEMP_MAX_K = 330.0
MAX_WIND_KT = 200.0
MIN_PRESSURE_HPA = 870.0
MAX_PRESSURE_HPA = 1020.0

# North Indian Ocean basin bounding box
NIO_BOUNDS = {
    "lat_min": 0.0,
    "lat_max": 30.0,
    "lon_min": 50.0,
    "lon_max": 100.0,
}

# ============================================================================
# IBTrACS Column Mappings (for the NI basin CSV)
# ============================================================================

IBTRACS_COLUMNS = {
    "storm_id": "SID",
    "name": "NAME",
    "timestamp": "ISO_TIME",
    "lat": "LAT",
    "lon": "LON",
    "basin": "BASIN",
    "max_wind": "WMO_WIND",      # knots
    "min_pressure": "WMO_PRES",  # hPa
    "nature": "NATURE",          # TS, ET, DS, etc.
    "dist2land": "DIST2LAND",    # km to nearest land
    "storm_speed": "STORM_SPEED",
    "storm_dir": "STORM_DIR",
}

# ============================================================================
# Prediction Model Defaults
# ============================================================================

PREDICTION_INPUT_FEATURES = [
    "lat", "lon", "max_wind_kt", "min_pressure_hpa",
    "sst", "shear_magnitude",
    "storm_speed_kph", "storm_bearing_deg",
    "time_since_genesis_hours",
    "delta_wind_6h", "delta_pressure_6h",
]

PREDICTION_OUTPUT_FEATURES = [
    "delta_lat", "delta_lon", "max_wind_kt", "min_pressure_hpa",
]

# Lead times for evaluation (hours)
EVAL_LEAD_TIMES_H = [6, 12, 24, 48, 72]

# ============================================================================
# Earth Constants
# ============================================================================

EARTH_RADIUS_KM = 6371.0
KT_TO_KPH = 1.852
KPH_TO_KT = 1.0 / KT_TO_KPH
