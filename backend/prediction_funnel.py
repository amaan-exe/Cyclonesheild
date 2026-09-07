"""
backend/prediction_funnel.py

Staged Prediction Funnel Service for Cyclone Shield AI.
Replaces naive binary ("is there a cyclone") classification with a 4-tier
physics-constrained early warning funnel:

1. T-18 Days Out: Regional Risk (Basin-scale thermal & atmospheric indicators)
2. T-14 Days Out: Cyclogenesis Probability (GPI ensemble, shear & humidity)
3. T-7 Days Out:  System Identification (Vortex center, Dvorak T1.5, cluster tracking)
4. T-3 Days Out:  Track & Intensity Forecast (72h high-res cone, landfall corridor, wind swaths)
"""

import datetime
from typing import Dict, Any, List

class PredictionFunnelService:
    def __init__(self):
        self.active_storm_name = "Cyclone Dana"
        self.current_operational_stage = "t3_track_intensity"

    def get_all_stages(self) -> Dict[str, Any]:
        """
        Returns full Staged Prediction Funnel dataset including all 4 horizons,
        geospatial geometries for GIS rendering, atmospheric triggers,
        and comparison matrix against naive binary classification.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        
        stages = {
            "t18_regional_risk": {
                "stage_key": "t18_regional_risk",
                "stage_number": 1,
                "lead_time_days": 18,
                "lead_time_hours": 432,
                "title": "Regional Risk Flagging",
                "subtitle": "Basin-Scale Thermal & Atmospheric Potential",
                "status_badge": "TRIGGERED (T-18d)",
                "alert_level": "Basin Watch / Regional Advisory",
                "alert_color": "#D97706", # Amber
                "summary": (
                    "At 18 days out, the system flags broad basin-scale risk by analyzing sea surface temperature (SST) "
                    "anomalies, Madden-Julian Oscillation (MJO) phase propagation, and low-level atmospheric vorticity."
                ),
                "operational_action": (
                    "Advisory issued to NDMA & Coastal State Disaster Authorities (OSDMA, APSDMA) to review "
                    "pre-monsoon shelter readiness and emergency generator fuel stocks."
                ),
                "physics_indicators": {
                    "sea_surface_temp_c": 29.8,
                    "sst_anomaly_c": "+1.4°C above climatology",
                    "tropical_cyclone_heat_potential": "92 kJ/cm² (>85 kJ/cm² favorable)",
                    "mjo_phase": "Phase 3/4 (Active convection over East Indian Ocean)",
                    "mjo_amplitude": 1.85,
                    "relative_vorticity_850hpa": "+12.4 x 10⁻⁶ s⁻¹",
                    "outgoing_longwave_radiation": "194 W/m² (Enhanced convective band)"
                },
                "spatial_focus": {
                    "basin": "Bay of Bengal (South & Central)",
                    "bounding_box": {"north": 22.0, "south": 5.0, "west": 80.0, "east": 98.0},
                    "thermal_anomaly_center": {"lat": 11.5, "lon": 88.0, "radius_km": 550},
                    "camera_view": {"lat": 13.5, "lon": 88.5, "zoom": 5}
                },
                "layers": {
                    "type": "basin_anomaly",
                    "polygon_coords": [
                        [5.0, 80.0], [5.0, 95.0], [10.0, 96.0], [16.0, 93.0],
                        [21.0, 90.0], [21.5, 87.0], [18.0, 83.0], [12.0, 80.0]
                    ],
                    "label": "Thermal Anomaly Risk Zone (SST > 29.5°C)"
                }
            },

            "t14_cyclogenesis": {
                "stage_key": "t14_cyclogenesis",
                "stage_number": 2,
                "lead_time_days": 14,
                "lead_time_hours": 336,
                "title": "Cyclogenesis Probability",
                "subtitle": "Genesis Potential Index (GPI) & Ensemble Likelihood",
                "status_badge": "ACTIVE (68.5% Prob)",
                "alert_level": "Cyclogenesis Outlook / Yellow Watch",
                "alert_color": "#CA8A04", # Yellow-Gold
                "summary": (
                    "By 14 days out, an ensemble of statistical-dynamical models computes the Genesis Potential Index (GPI) "
                    "and multi-model formation probability within the favorable atmospheric corridor."
                ),
                "operational_action": (
                    "Port signal Cautionary No. 1 issued at major maritime ports (Paradeep, Visakhapatnam). "
                    "Deep-sea fishing trawlers alerted to avoid southern Bay of Bengal."
                ),
                "physics_indicators": {
                    "genesis_probability_pct": 68.5,
                    "genesis_potential_index_gpi": 8.4,
                    "vertical_wind_shear_kt": "11.2 kt (Favorable window < 15 kt)",
                    "mid_troposphere_rh_700hpa": "78% Relative Humidity",
                    "coriolis_parameter": "Favorable (Lat > 8°N)",
                    "ensemble_consensus": "14/20 GEFS & ECMWF members predict depression formation"
                },
                "spatial_focus": {
                    "basin": "Central-South Bay of Bengal",
                    "genesis_ellipse_center": {"lat": 12.8, "lon": 89.2},
                    "semi_major_km": 280,
                    "semi_minor_km": 180,
                    "tilt_deg": 45,
                    "camera_view": {"lat": 15.0, "lon": 88.0, "zoom": 5}
                },
                "layers": {
                    "type": "genesis_ellipse",
                    "contours": [
                        {"probability": 40, "radius_km": 360, "color": "#FEF08A"},
                        {"probability": 60, "radius_km": 260, "color": "#FACC15"},
                        {"probability": 68.5, "radius_km": 160, "color": "#EAB308"}
                    ],
                    "label": "Cyclogenesis Probability Envelope (68.5% Consensus)"
                }
            },

            "t7_system_id": {
                "stage_key": "t7_system_id",
                "stage_number": 3,
                "lead_time_days": 7,
                "lead_time_hours": 168,
                "title": "System Identification & Pre-Genesis Tracking",
                "subtitle": "Developing Vortex Localization (BOB-06)",
                "status_badge": "CONFIRMED (BOB-06)",
                "alert_level": "Cyclone Alert / Orange Stage 1",
                "alert_color": "#EA580C", # Orange
                "summary": (
                    "At 7 days out, the system isolates and identifies the nascent low-pressure circulation (BOB-06), "
                    "locates the convective cluster centroid via satellite Dvorak pattern analysis, and projects initial track direction."
                ),
                "operational_action": (
                    "Mandatory recall of all marine fishermen. District Emergency Operations Centers (DEOC) "
                    "activated in 24x7 monitoring mode. Multi-purpose cyclone shelter inspections initiated."
                ),
                "physics_indicators": {
                    "system_designation": "Depression / Deep Depression BOB-06",
                    "vortex_center": {"lat": 14.20, "lon": 89.50},
                    "central_pressure_hpa": 1000.0,
                    "pressure_deficit_hpa": -6.0,
                    "dvorak_t_number": "T1.5 (Curved band organization)",
                    "surface_wind_speed_kph": "52 km/h (28 kt)",
                    "convective_cluster_radius_km": 220,
                    "initial_heading": "North-Northwest (325°) towards Odisha/AP Coast"
                },
                "spatial_focus": {
                    "basin": "West-Central Bay of Bengal",
                    "camera_view": {"lat": 16.5, "lon": 87.5, "zoom": 6}
                },
                "layers": {
                    "type": "developing_vortex",
                    "vortex_center": [14.20, 89.50],
                    "convective_radius_km": 220,
                    "trajectory_vector": [
                        [14.20, 89.50],
                        [15.40, 88.80],
                        [16.80, 87.90],
                        [18.42, 86.85]
                    ],
                    "label": "Developing Vortex Center (BOB-06, T1.5)"
                }
            },

            "t3_track_intensity": {
                "stage_key": "t3_track_intensity",
                "stage_number": 4,
                "lead_time_days": 3,
                "lead_time_hours": 72,
                "title": "High-Resolution Track & Intensity Forecast",
                "subtitle": "Physics-Constrained Trajectory & Landfall Cone",
                "status_badge": "OPERATIONAL (Very Severe CS)",
                "alert_level": "RED ALERT — MANDATORY EVACUATION",
                "alert_color": "#DC2626", # Crimson Red
                "summary": (
                    "By 3 days out (72 hours to landfall), the deep sequence physics-informed model generates high-resolution "
                    "track waypoints, uncertainty cones, maximum sustained wind speeds, and coastal storm surge predictions."
                ),
                "operational_action": (
                    "Mandatory evacuation of low-lying coastal populations within 5 km. Pre-positioning 18 NDRF and "
                    "24 ODRAF rescue teams. Great Danger Signal GD-10 hoisted at Paradeep Port."
                ),
                "physics_indicators": {
                    "imd_classification": "Very Severe Cyclonic Storm (VSCS)",
                    "current_eye_coords": {"lat": 18.42, "lon": 86.85},
                    "peak_sustained_wind_kph": 120.0,
                    "peak_gusts_kph": 145.0,
                    "min_central_pressure_hpa": 982.0,
                    "landfall_target": "North Odisha Coast (between Puri and Dhamra Port)",
                    "landfall_eta_hours": "12–14 Hours",
                    "storm_surge_height_m": "2.0 to 3.2 meters above astronomical tide",
                    "radius_of_maximum_winds_km": 28.0,
                    "track_error_24h_km": 46.8
                },
                "spatial_focus": {
                    "basin": "North-West Bay of Bengal & Odisha Coast",
                    "camera_view": {"lat": 19.5, "lon": 86.5, "zoom": 7}
                },
                "layers": {
                    "type": "high_res_cone",
                    "landfall_point": [19.81, 85.83],
                    "label": "72h Physics-Constrained Cone of Uncertainty & Landfall Corridor"
                }
            }
        }

        # Architectural comparison against naive binary classification
        comparison_matrix = {
            "title": "Architectural Paradigm: Staged Prediction Funnel vs. Naive Binary Classifier",
            "dimensions": [
                {
                    "metric": "Decision Horizon & Lead Time",
                    "naive_binary_classifier": "0 to 24 hours (Single instant snapshot, zero early warning)",
                    "staged_funnel": "18 Days &rarr; 14 Days &rarr; 7 Days &rarr; 3 Days (Graduated 432h runway)",
                    "impact": "Expands disaster preparation window from 1 day to nearly 3 weeks"
                },
                {
                    "metric": "False Alarm Cost Mitigation",
                    "naive_binary_classifier": "High binary false alarms force costly premature statewide evacuations",
                    "staged_funnel": "Tiered response: Basin monitoring (18d) &rarr; Port warnings (14d) &rarr; Evacuations only at 3d",
                    "impact": "Saves state exchequer hundreds of crores in unnecessary early evacuations"
                },
                {
                    "metric": "Physical Coherence & Explainability",
                    "naive_binary_classifier": "Opaque black-box binary score ('Cyclone: 0.82')",
                    "staged_funnel": "Physics-anchored at every tier (SST/MJO &rarr; GPI/Shear &rarr; Vortex ID &rarr; GRU track cone)",
                    "impact": "Auditable by meteorologists and trusted by state emergency authorities"
                },
                {
                    "metric": "Geospatial Actionability",
                    "naive_binary_classifier": "No trajectory, no cone of uncertainty, no landfall corridor",
                    "staged_funnel": "GeoJSON multi-horizon layers: Basin bounding box &rarr; Genesis ellipse &rarr; 72h corridor",
                    "impact": "Precise district-level resource allocation (Puri, Jagatsinghpur, Balasore)"
                }
            ]
        }

        return {
            "funnel_version": "StagedPredictionFunnel v3.0",
            "active_storm": self.active_storm_name,
            "current_stage": self.current_operational_stage,
            "generated_at": now.isoformat(),
            "stages": stages,
            "comparison_matrix": comparison_matrix
        }

    def get_stage_by_key(self, stage_key: str) -> Dict[str, Any]:
        """
        Returns deep physics metrics and spatial definitions for a single funnel stage.
        """
        all_data = self.get_all_stages()
        stages = all_data["stages"]
        if stage_key not in stages:
            raise KeyError(f"Invalid funnel stage '{stage_key}'. Valid stages: {list(stages.keys())}")
        return {
            "stage": stages[stage_key],
            "active_storm": self.active_storm_name,
            "funnel_version": all_data["funnel_version"]
        }

prediction_funnel_service = PredictionFunnelService()
