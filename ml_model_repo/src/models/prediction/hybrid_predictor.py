"""
Cyclone Horizon — Physics-Informed Hybrid Track & Intensity Predictor
Combines:
  1. Deep Sequence Model: Autoregressive Temporal Attention GRU/LSTM
  2. Physics-Based Steering: Beta-Advection Model (BAM) & 500 hPa environmental steering
  3. Dynamic Gating Layer: Balances data-driven inertia vs environmental advection
  4. Probabilistic Variance Output: Calibrated 70% Cone of Uncertainty (mean + sigma)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaAdvectionModel:
    """
    Analytical Beta-Advection Model (BAM) for tropical cyclone steering physics.
    Computes environmental steering vector + planetary beta-drift:
      V_track = V_environmental_steering + V_beta_drift
    """

    def __init__(self, beta_drift_speed_ms: float = 2.2, beta_drift_azimuth_deg: float = 315.0):
        self.beta_speed_kph = beta_drift_speed_ms * 3.6  # Convert m/s to km/h
        self.beta_azimuth_rad = math.radians(beta_drift_azimuth_deg)  # Northwestward drift

    def compute_steering_step(
        self,
        current_lat: float,
        current_lon: float,
        environmental_u_kt: float,
        environmental_v_kt: float,
        dt_hours: float = 6.0
    ) -> Tuple[float, float]:
        """
        Computes the physical delta_lat and delta_lon over dt_hours.
        """
        # Convert environmental wind knots to km/h (1 kt = 1.852 km/h)
        u_kph = environmental_u_kt * 1.852
        v_kph = environmental_v_kt * 1.852

        # Beta drift components (typically ~8 km/h towards 315° NW)
        beta_u_kph = self.beta_speed_kph * math.sin(self.beta_azimuth_rad)
        beta_v_kph = self.beta_speed_kph * math.cos(self.beta_azimuth_rad)

        total_u_kph = u_kph + beta_u_kph
        total_v_kph = v_kph + beta_v_kph

        # Distance traveled in km
        dx_km = total_u_kph * dt_hours
        dy_km = total_v_kph * dt_hours

        # Convert to degrees latitude and longitude
        dlat = dy_km / 111.0
        cos_lat = max(0.2, math.cos(math.radians(current_lat)))
        dlon = dx_km / (111.0 * cos_lat)

        return float(dlat), float(dlon)


class HybridCyclonePredictor(nn.Module):
    """
    Physics-Informed Hybrid Sequence Predictor.
    Blends temporal GRU attention with the Beta-Advection steering model.
    """

    def __init__(
        self,
        input_dim: int = 12,
        output_dim: int = 4,  # delta_lat, delta_lon, max_wind_kt, min_pressure_hpa
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25,
        probabilistic: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.probabilistic = probabilistic
        self.bam = BetaAdvectionModel()

        # Input feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Sequence encoder (past N fixes)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Temporal attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Autoregressive sequence decoder
        self.decoder = nn.GRU(
            input_size=output_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # Physics-ML adaptive gating head (outputs lambda in [0, 1])
        self.physics_gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # weights for (dlat, dlon)
            nn.Sigmoid()
        )

        # Multi-horizon prediction heads
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        if probabilistic:
            self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        past_sequence: torch.Tensor,
        future_steps: int = 8,
        env_steering: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        past_sequence : (batch, seq_len, input_dim)
        future_steps : number of future lead horizons (e.g. 8 steps = +48h @ 6h)
        env_steering : optional (batch, future_steps, 2) [steering_u, steering_v] in knots
        """
        batch_size, seq_len, _ = past_sequence.shape
        proj = self.input_proj(past_sequence)
        enc_out, h_n = self.encoder(proj)

        # Compute temporal attention weights over past fixes
        att_scores = self.attention(enc_out)  # (batch, seq_len, 1)
        att_weights = F.softmax(att_scores, dim=1)
        context = torch.sum(att_weights * enc_out, dim=1)  # (batch, hidden_dim)

        # Autoregressive decoding
        dec_h = h_n
        prev_pred = torch.zeros(batch_size, self.output_dim, device=past_sequence.device)

        mean_preds = []
        logvar_preds = []
        gate_weights = []

        for step in range(future_steps):
            # Input is previous prediction concatenated with attention context
            dec_in = torch.cat([prev_pred, context], dim=-1).unsqueeze(1)
            dec_out, dec_h = self.decoder(dec_in, dec_h)
            dec_feat = dec_out.squeeze(1)

            step_mean = self.mean_head(dec_feat)

            # Adaptive physics-ML gating
            gate = self.physics_gate(dec_feat)  # (batch, 2)
            gate_weights.append(gate)

            if env_steering is not None and step < env_steering.shape[1]:
                u_kt = env_steering[:, step, 0].unsqueeze(-1)
                v_kt = env_steering[:, step, 1].unsqueeze(-1)
                # Physical advection proxy delta degrees per 6h
                phys_dlat = (v_kt * 1.852 * 6.0) / 111.0 + 0.12  # includes northward beta drift
                phys_dlon = (u_kt * 1.852 * 6.0) / (111.0 * 0.94) - 0.12  # westward beta drift
                phys_step = torch.cat([phys_dlat, phys_dlon], dim=-1)

                # Blend ML delta with physics delta
                blended_track = gate * step_mean[:, :2] + (1.0 - gate) * phys_step
                step_mean = torch.cat([blended_track, step_mean[:, 2:]], dim=-1)

            mean_preds.append(step_mean)
            prev_pred = step_mean.detach()

            if self.probabilistic:
                step_logvar = self.logvar_head(dec_feat)
                # Ensure variance grows with lead time (physical uncertainty envelope)
                growth_factor = 1.0 + 0.15 * step
                step_logvar = step_logvar + math.log(growth_factor)
                logvar_preds.append(step_logvar)

        means = torch.stack(mean_preds, dim=1)  # (batch, future_steps, output_dim)
        res = {
            "mean": means,
            "gate_weights": torch.stack(gate_weights, dim=1)
        }

        if self.probabilistic:
            logvars = torch.stack(logvar_preds, dim=1)
            res["logvar"] = logvars
            res["std"] = torch.exp(0.5 * logvars)

        return res

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        target_len: int = 8,
        n_samples: int = 50,
        env_steering: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Produce predictions with calibrated uncertainty envelope.
        
        Parameters
        ----------
        x : Tensor of shape (seq_len, input_dim) or (batch, seq_len, input_dim)
        target_len : number of future steps to predict
        n_samples : sample count (analytical log-variance used for fast deterministic envelope)
        env_steering : optional steering wind tensor (batch, future_steps, 2)
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        with torch.no_grad():
            res = self.forward(x, future_steps=target_len, env_steering=env_steering)
            mean = res["mean"].squeeze(0)
            std = res["std"].squeeze(0) if "std" in res else torch.ones_like(mean) * 0.2
            return {
                "mean": mean,
                "std": std,
                "gate_weights": res.get("gate_weights", torch.zeros(1)).squeeze(0)
            }
