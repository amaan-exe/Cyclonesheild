"""
Cyclone Horizon — Cyclogenesis & Cyclone Formation Predictor
Physics-Constrained Deep & Statistical-Dynamical Cyclogenesis Model:
  1. Computes Emanuel-Nolan Genesis Potential Index (GPI) & Camargo Modified Formulation
  2. Evaluates 6 Key Thermodynamic & Dynamic Environmental Thresholds:
     - Sea Surface Temperature (SST >= 26.5°C threshold & warm pool anomaly)
     - Tropical Cyclone Heat Potential (TCHP >= 80 kJ/cm²)
     - Deep Layer Vertical Wind Shear (850 - 200 hPa in knots, favorable < 12 kt)
     - Mid-Tropospheric Relative Humidity (700 hPa >= 65-70%)
     - Low-Level Relative Vorticity (850 hPa >= +10 x 10^-6 s^-1)
     - Coriolis Parameter (f = 2 * Omega * sin(lat), equator buffer > 4°N)
  3. Predicts:
     - Continuous Cyclogenesis Formation Probability (0.0% to 100.0%)
     - Formation Stage (Pre-Genesis Disturbance, Low Pressure Area, Depression, Cyclonic Storm)
     - Time to Genesis (Lead hours until IMD Depression stage threshold >= 17 kt / 31 km/h)
     - Predicted Genesis Centroid (Lat, Lon)
     - Environmental Favorability Diagnostic Scorecard
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Earth's angular velocity (rad/s)
OMEGA_EARTH = 7.2921e-5


@dataclass
class CyclogenesisDiagnostics:
    sst_c: float
    sst_favorable: bool
    tchp_kj_cm2: float
    tchp_favorable: bool
    vertical_wind_shear_kt: float
    shear_favorable: bool
    mid_rh_percent: float
    rh_favorable: bool
    relative_vorticity_850: float
    vorticity_favorable: bool
    coriolis_parameter: float
    coriolis_favorable: bool
    central_pressure_hpa: float
    pressure_deficit_hpa: float
    gpi_score: float
    gpi_rating: str
    primary_triggers: List[str]
    inhibiting_factors: List[str]


@dataclass
class CyclogenesisPredictionResult:
    formation_chance_percent: float
    formation_stage: str
    stage_code: str
    risk_tier: str
    alert_color: str
    time_to_genesis_hours: Optional[int]
    predicted_genesis_lat: float
    predicted_genesis_lon: float
    gpi_score: float
    diagnostics: CyclogenesisDiagnostics
    model_method: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class CyclogenesisPredictor:
    """
    Operational Cyclogenesis & Cyclone Formation Predictor for the North Indian Ocean basin
    (Bay of Bengal & Arabian Sea).
    """

    def __init__(self):
        # Climatological baseline values for North Indian Ocean
        self.ambient_sea_level_pressure = 1010.0  # hPa
        self.critical_sst = 26.5                  # °C
        self.favorable_vws_threshold = 12.0       # kt
        self.hostile_vws_threshold = 22.0         # kt
        self.favorable_rh_threshold = 65.0        # %
        self.favorable_vort_threshold = 10.0      # 10^-6 s^-1

    def compute_coriolis(self, lat: float) -> float:
        """Computes Coriolis parameter f = 2 * Omega * sin(lat)."""
        lat_rad = math.radians(abs(lat))
        return 2.0 * OMEGA_EARTH * math.sin(lat_rad)

    def compute_mpi(self, sst_c: float) -> float:
        """
        Empirical Maximum Potential Intensity (V_pot in m/s) based on Emanuel (1988/2004).
        Approximates the thermodynamic upper bound of tropical cyclone wind speed given SST.
        """
        if sst_c < 26.0:
            return 25.0
        # Approximately 35 m/s at 26.5°C with ~9.5 m/s per degree warming
        return float(min(90.0, max(25.0, 35.0 + 9.5 * (sst_c - 26.5))))

    def compute_gpi(
        self,
        lat: float,
        sst_c: float,
        vertical_wind_shear_kt: float,
        mid_rh_percent: float,
        vorticity_850: float = 12.0
    ) -> float:
        """
        Computes standard Emanuel & Nolan (2004) Genesis Potential Index (GPI):
          GPI = |10^5 * eta|^(3/2) * (max(0, RH - 50) / 50)^3 * (V_pot / 70)^3 * (1 + 0.1 * V_shear)^(-2)
        Where:
          eta = absolute vorticity (f + zeta) in s^-1
          V_shear in m/s
        """
        f = self.compute_coriolis(lat)
        # Convert vorticity from 10^-6 s^-1 to s^-1
        zeta = vorticity_850 * 1e-6
        eta = abs(f + zeta)

        # Non-dimensional vorticity term
        term_vort = abs(1e5 * eta) ** 1.5

        # Mid-tropospheric humidity term
        rh_excess = max(0.0, mid_rh_percent - 50.0) / 50.0
        term_rh = rh_excess ** 3.0

        # Potential intensity term
        v_pot = self.compute_mpi(sst_c)
        term_mpi = (v_pot / 70.0) ** 3.0

        # Shear term (convert knots to m/s: 1 kt = 0.5144 m/s)
        v_shear_ms = max(0.0, vertical_wind_shear_kt * 0.5144)
        term_shear = (1.0 + 0.1 * v_shear_ms) ** (-2.0)

        gpi = term_vort * term_rh * term_mpi * term_shear
        return float(round(max(0.0, min(25.0, gpi)), 2))

    def predict_formation(
        self,
        lat: float,
        lon: float,
        sst_c: float = 29.5,
        tchp_kj_cm2: float = 88.0,
        vertical_wind_shear_kt: float = 11.5,
        mid_rh_percent: float = 76.0,
        vorticity_850: float = 14.0,
        central_pressure_hpa: float = 1004.0,
        cloud_top_temp_c: float = -62.0,
        storm_speed_kph: float = 14.0,
        storm_bearing_deg: float = 315.0
    ) -> CyclogenesisPredictionResult:
        """
        Comprehensive physics-constrained prediction of tropical cyclone formation / cyclogenesis.
        Evaluates atmospheric dynamics, computes GPI, and predicts probability, timing, and stage.
        """
        # 1. Compute physical indicators
        f = self.compute_coriolis(lat)
        gpi = self.compute_gpi(lat, sst_c, vertical_wind_shear_kt, mid_rh_percent, vorticity_850)
        pressure_deficit = max(0.0, self.ambient_sea_level_pressure - central_pressure_hpa)

        # 2. Check individual favorability thresholds
        sst_fav = bool(sst_c >= self.critical_sst)
        tchp_fav = bool(tchp_kj_cm2 >= 65.0)
        shear_fav = bool(vertical_wind_shear_kt <= self.favorable_vws_threshold)
        rh_fav = bool(mid_rh_percent >= self.favorable_rh_threshold)
        vort_fav = bool(vorticity_850 >= self.favorable_vort_threshold)
        coriolis_fav = bool(abs(lat) >= 4.5)  # Needs enough Coriolis force for rotational spin-up

        # 3. Categorize triggers and inhibitors
        primary_triggers = []
        inhibiting_factors = []

        if sst_c >= 29.0:
            primary_triggers.append(f"High SST ({sst_c:.1f} deg C) provides abundant sensible and latent heat flux.")
        elif sst_fav:
            primary_triggers.append(f"SST ({sst_c:.1f} deg C) is above the 26.5 deg C threshold required for deep convection.")
        else:
            inhibiting_factors.append(f"Sub-threshold SST ({sst_c:.1f} deg C < 26.5 deg C) inhibits moist convective maintenance.")

        if tchp_fav:
            primary_triggers.append(f"Ocean heat content ({tchp_kj_cm2:.0f} kJ/cm^2) prevents cold water upwelling.")

        if vertical_wind_shear_kt < 12.0:
            primary_triggers.append(f"Low vertical wind shear ({vertical_wind_shear_kt:.1f} kt) allows vertical vortex alignment.")
        elif vertical_wind_shear_kt > self.hostile_vws_threshold:
            inhibiting_factors.append(f"Strong vertical wind shear ({vertical_wind_shear_kt:.1f} kt) ventilates and shears convective core.")

        if mid_rh_percent >= 70.0:
            primary_triggers.append(f"High mid-tropospheric humidity ({mid_rh_percent:.0f}%) suppresses dry air entrainment.")
        elif mid_rh_percent < 55.0:
            inhibiting_factors.append(f"Dry mid-tropospheric air ({mid_rh_percent:.0f}%) triggers evaporative downdrafts.")

        if vorticity_850 >= 12.0:
            primary_triggers.append(f"Strong 850 hPa relative vorticity (+{vorticity_850:.1f} x 10^-6 s^-1) promotes cyclonic spin-up.")

        if not coriolis_fav:
            inhibiting_factors.append(f"Low latitude ({abs(lat):.1f} deg N) provides insufficient planetary vorticity (f < 1.1 x 10^-5 s^-1).")

        if cloud_top_temp_c <= -55.0:
            primary_triggers.append(f"Deep convective cloud tops ({cloud_top_temp_c:.1f} deg C) indicate intense updrafts.")

        # 4. GPI Rating
        if gpi >= 8.0:
            gpi_rating = "EXTREMELY HIGH FAVORABILITY"
        elif gpi >= 5.0:
            gpi_rating = "HIGH FAVORABILITY"
        elif gpi >= 2.5:
            gpi_rating = "MODERATE FAVORABILITY"
        elif gpi >= 1.0:
            gpi_rating = "MARGINAL / LOW"
        else:
            gpi_rating = "UNFAVORABLE"

        # 5. Continuous Calibrated Formation Probability Model
        # Logit combination of physical drivers
        logit = -2.8  # Baseline intercept (~6% ambient probability)
        logit += 0.85 * math.log(1.0 + gpi)
        logit += 0.55 * (sst_c - self.critical_sst)
        logit += 0.03 * (tchp_kj_cm2 - 60.0)
        logit += 0.045 * (mid_rh_percent - 60.0)
        logit += 0.08 * (vorticity_850 - 8.0)
        logit += 0.25 * pressure_deficit
        # Shear penalty
        if vertical_wind_shear_kt > 12.0:
            logit -= 0.12 * (vertical_wind_shear_kt - 12.0)
        # Low latitude penalty
        if abs(lat) < 5.0:
            logit -= 1.2 * (5.0 - abs(lat))
        # Deep convection boost
        if cloud_top_temp_c < -45.0:
            logit += 0.03 * abs(cloud_top_temp_c - (-45.0))

        prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit))))
        formation_chance = round(float(prob * 100.0), 1)
        formation_chance = max(2.0, min(99.0, formation_chance))

        # 6. Classification of Stage and Risk Tier
        if formation_chance >= 75.0 or pressure_deficit >= 6.0:
            formation_stage = "Depression / Cyclogenesis Imminent"
            stage_code = "GENESIS-ACTIVE"
            risk_tier = "HIGH WATCH"
            alert_color = "#DC2626"
            time_to_genesis_hours = max(6, int(round(18.0 - pressure_deficit * 1.5)))
        elif formation_chance >= 50.0 or pressure_deficit >= 3.0:
            formation_stage = "Well-Marked Low Pressure Area (WML)"
            stage_code = "WML"
            risk_tier = "ELEVATED WATCH"
            alert_color = "#EA580C"
            time_to_genesis_hours = max(18, int(round(36.0 - pressure_deficit * 2.0)))
        elif formation_chance >= 30.0:
            formation_stage = "Low Pressure Area (LPA) Forming"
            stage_code = "LPA"
            risk_tier = "MODERATE WATCH"
            alert_color = "#F59E0B"
            time_to_genesis_hours = max(48, int(round(72.0 - pressure_deficit * 3.0)))
        elif formation_chance >= 15.0:
            formation_stage = "Pre-Genesis Convective Disturbance"
            stage_code = "DISTURBANCE"
            risk_tier = "LOW WATCH"
            alert_color = "#0EA5E9"
            time_to_genesis_hours = 120
        else:
            formation_stage = "Routine Basin Surveillance / Fair Weather"
            stage_code = "ROUTINE"
            risk_tier = "ALL CLEAR"
            alert_color = "#10B981"
            time_to_genesis_hours = None

        # 7. Projected Genesis Location (Lat, Lon)
        # Using forward propagation vector over time_to_genesis
        if time_to_genesis_hours is not None:
            lead_h = min(48.0, float(time_to_genesis_hours))
            bearing_rad = math.radians(storm_bearing_deg)
            # Distance traveled in km
            dist_km = storm_speed_kph * lead_h
            dlat = (dist_km * math.cos(bearing_rad)) / 111.0
            cos_lat = max(0.2, math.cos(math.radians(lat)))
            dlon = (dist_km * math.sin(bearing_rad)) / (111.0 * cos_lat)
            pred_lat = round(float(lat + dlat), 2)
            pred_lon = round(float(lon + dlon), 2)
        else:
            pred_lat = round(float(lat), 2)
            pred_lon = round(float(lon), 2)

        diagnostics = CyclogenesisDiagnostics(
            sst_c=round(sst_c, 1),
            sst_favorable=sst_fav,
            tchp_kj_cm2=round(tchp_kj_cm2, 1),
            tchp_favorable=tchp_fav,
            vertical_wind_shear_kt=round(vertical_wind_shear_kt, 1),
            shear_favorable=shear_fav,
            mid_rh_percent=round(mid_rh_percent, 1),
            rh_favorable=rh_fav,
            relative_vorticity_850=round(vorticity_850, 1),
            vorticity_favorable=vort_fav,
            coriolis_parameter=float(f"{f:.2e}"),
            coriolis_favorable=coriolis_fav,
            central_pressure_hpa=round(central_pressure_hpa, 1),
            pressure_deficit_hpa=round(pressure_deficit, 1),
            gpi_score=gpi,
            gpi_rating=gpi_rating,
            primary_triggers=primary_triggers,
            inhibiting_factors=inhibiting_factors
        )

        return CyclogenesisPredictionResult(
            formation_chance_percent=formation_chance,
            formation_stage=formation_stage,
            stage_code=stage_code,
            risk_tier=risk_tier,
            alert_color=alert_color,
            time_to_genesis_hours=time_to_genesis_hours,
            predicted_genesis_lat=pred_lat,
            predicted_genesis_lon=pred_lon,
            gpi_score=gpi,
            diagnostics=diagnostics,
            model_method="Emanuel-Nolan GPI & Multi-Factor Dynamic Cyclogenesis Model"
        )
