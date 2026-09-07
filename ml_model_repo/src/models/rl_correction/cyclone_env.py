"""
Cyclone Horizon — Historical Cyclone Replay Environment for RL Forecast Correction
Gym-compatible episodic environment replaying historical storm episodes frame-by-frame:
  - State: [ML_delta_lat, ML_delta_lon, phys_steering_u, phys_steering_v, recent_error_km, current_wind, current_pressure, shear]
  - Action: Bounded correction [dlat_corr, dlon_corr, dwind_corr]
  - Reward: Potential-based reward shaping normalized by climatological storm difficulty
  - Safety: Hard safety clipping on action bounds
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class HistoricalEpisode:
    """Historical storm sequence for RL environment replay."""
    storm_id: str
    storm_name: str
    lats: np.ndarray
    lons: np.ndarray
    winds: np.ndarray
    pressures: np.ndarray
    shears: np.ndarray
    raw_ml_predictions: np.ndarray  # (steps, 3) [dlat, dlon, dwind]
    steering_vectors: np.ndarray    # (steps, 2) [u, v]


class CycloneForecastEnv:
    """
    Simulated sequential decision environment for RL forecast correction.
    """

    def __init__(
        self,
        episodes: List[HistoricalEpisode],
        max_track_nudge_deg: float = 0.45,    # ~50 km max bounded clip
        max_intensity_nudge_kt: float = 8.0,  # ~8 kt max bounded clip
        gamma: float = 0.95
    ):
        self.episodes = episodes
        self.max_track_nudge = max_track_nudge_deg
        self.max_intensity_nudge = max_intensity_nudge_kt
        self.gamma = gamma

        # State dimension = 8
        self.state_dim = 8
        # Action dimension = 3: [dlat_corr, dlon_corr, dwind_corr]
        self.action_dim = 3

        self.current_ep_idx = 0
        self.current_step = 0
        self.recent_error_km = 30.0

    def reset(self, episode_idx: Optional[int] = None) -> np.ndarray:
        if episode_idx is not None:
            self.current_ep_idx = episode_idx % len(self.episodes)
        else:
            self.current_ep_idx = np.random.randint(0, len(self.episodes))

        self.current_step = 0
        self.recent_error_km = 25.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        ep = self.episodes[self.current_ep_idx]
        step = self.current_step

        ml_pred = ep.raw_ml_predictions[step]
        steer = ep.steering_vectors[step]
        cur_wind = ep.winds[step]
        cur_press = ep.pressures[step]
        cur_shear = ep.shears[step]

        state = np.array([
            ml_pred[0],               # ML predicted dlat
            ml_pred[1],               # ML predicted dlon
            steer[0] / 30.0,          # Normalized steering U
            steer[1] / 30.0,          # Normalized steering V
            self.recent_error_km / 100.0, # Recent error
            cur_wind / 150.0,         # Normalized wind
            (cur_press - 950.0) / 50.0, # Normalized pressure
            cur_shear / 40.0          # Normalized shear
        ], dtype=np.float32)

        state = np.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)
        return np.clip(state, -10.0, 10.0)

    def _potential(self, error_km: float) -> float:
        """Potential function Phi(s) for potential-based reward shaping."""
        clean_err = float(np.nan_to_num(error_km, nan=50.0))
        return -clean_err / 50.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes an RL correction step with HARD SAFETY CLIPPING.
        
        Parameters
        ----------
        action : [dlat_nudge, dlon_nudge, dwind_nudge]
        """
        # Hard Safety Clip (guarantees RL cannot cause catastrophic divergence)
        dlat_nudge = np.clip(action[0], -self.max_track_nudge, self.max_track_nudge)
        dlon_nudge = np.clip(action[1], -self.max_track_nudge, self.max_track_nudge)
        dwind_nudge = np.clip(action[2], -self.max_intensity_nudge, self.max_intensity_nudge)

        ep = self.episodes[self.current_ep_idx]
        step = self.current_step

        raw_ml = ep.raw_ml_predictions[step]
        # Applied correction
        corrected_dlat = raw_ml[0] + dlat_nudge
        corrected_dlon = raw_ml[1] + dlon_nudge
        corrected_wind = ep.winds[step] + raw_ml[2] + dwind_nudge

        # Ground truth delta to next fix
        actual_dlat = ep.lats[step + 1] - ep.lats[step]
        actual_dlon = ep.lons[step + 1] - ep.lons[step]
        actual_wind = ep.winds[step + 1]

        # Great-circle error approximation in km
        dlat_km = (corrected_dlat - actual_dlat) * 111.0
        cos_lat = math.cos(math.radians(ep.lats[step]))
        dlon_km = (corrected_dlon - actual_dlon) * 111.0 * cos_lat
        step_track_error_km = math.sqrt(dlat_km**2 + dlon_km**2)
        step_wind_error_kt = abs(corrected_wind - actual_wind)

        # Baseline raw ML error (for comparison)
        raw_dlat_km = (raw_ml[0] - actual_dlat) * 111.0
        raw_dlon_km = (raw_ml[1] - actual_dlon) * 111.0 * cos_lat
        raw_track_error_km = math.sqrt(raw_dlat_km**2 + raw_dlon_km**2)
        raw_wind_error_kt = abs(ep.winds[step] + raw_ml[2] - actual_wind)

        track_improvement_km = raw_track_error_km - step_track_error_km
        wind_improvement_kt = raw_wind_error_kt - step_wind_error_kt

        # Direct comparative reward: positive when RL improves upon raw forecast
        reward = (track_improvement_km / 12.0) + (wind_improvement_kt / 5.0) - (step_track_error_km / 120.0)

        # Update tracking state
        self.recent_error_km = step_track_error_km
        self.current_step += 1
        done = (self.current_step >= len(ep.lats) - 2)

        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "corrected_track_error_km": step_track_error_km,
            "raw_track_error_km": raw_track_error_km,
            "improvement_km": raw_track_error_km - step_track_error_km,
            "wind_error_kt": step_wind_error_kt
        }

        return next_state, float(reward), done, info
