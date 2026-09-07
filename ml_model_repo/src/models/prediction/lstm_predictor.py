"""
Cyclone Horizon — LSTM/GRU Track & Intensity Predictor
Sequence-to-sequence model for forecasting cyclone position and intensity.

Outputs probabilistic predictions (mean + variance) for uncertainty cone.
Uses only causal features — no future information leaks into the input.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class CyclonePredictor(nn.Module):
    """
    GRU/LSTM sequence-to-sequence model for tropical cyclone track
    and intensity prediction.
    
    Architecture:
    - Encoder: multi-layer GRU/LSTM processes past N timesteps
    - Decoder: autoregressive — feeds previous output as next input
    - Output: mean + log_variance for each predicted variable
      (enables Gaussian NLL loss → calibrated uncertainty cone)
    
    Input features per timestep:
        lat, lon, max_wind_kt, min_pressure_hpa, sst, shear_magnitude,
        storm_speed_kph, storm_bearing_deg, time_since_genesis_hours,
        delta_wind_6h, delta_pressure_6h
    
    Output per timestep:
        delta_lat, delta_lon, max_wind_kt, min_pressure_hpa
        + log_variance for each (total 8 outputs)
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        output_dim: int = 4,  # delta_lat, delta_lon, wind, pressure
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "gru",
        bidirectional: bool = False,
        probabilistic: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.probabilistic = probabilistic
        self.rnn_type = rnn_type
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Encoder RNN
        RNNClass = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.encoder = RNNClass(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        enc_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Decoder RNN (always unidirectional — causal)
        self.decoder = RNNClass(
            input_size=output_dim + enc_output_dim,  # prev output + context
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Attention mechanism for context vector
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + enc_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Output heads
        if probabilistic:
            # Output mean + log_variance for each variable
            self.output_mean = nn.Linear(hidden_dim, output_dim)
            self.output_logvar = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_fc = nn.Linear(hidden_dim, output_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input sequence.
        
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        
        Returns
        -------
        enc_outputs : (batch, seq_len, enc_output_dim)
        hidden : final hidden state for decoder initialization
        """
        projected = self.input_proj(x)
        enc_outputs, hidden = self.encoder(projected)
        return enc_outputs, hidden
    
    def compute_attention(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention-weighted context vector.
        
        Parameters
        ----------
        decoder_hidden : (batch, hidden_dim) — current decoder state
        encoder_outputs : (batch, src_len, enc_dim)
        
        Returns
        -------
        context : (batch, enc_dim)
        """
        src_len = encoder_outputs.size(1)
        
        # Repeat decoder hidden for each source position
        hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Compute attention scores
        combined = torch.cat([hidden_expanded, encoder_outputs], dim=2)
        energy = self.attention(combined).squeeze(2)  # (batch, src_len)
        
        attn_weights = torch.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context
    
    def decode_step(
        self,
        prev_output: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoder step.
        
        Parameters
        ----------
        prev_output : (batch, output_dim) — previous timestep prediction
        hidden : decoder hidden state
        encoder_outputs : (batch, src_len, enc_dim)
        
        Returns
        -------
        prediction : (batch, output_dim) or (batch, 2*output_dim) if probabilistic
        hidden : updated hidden state
        """
        # Get decoder hidden for attention
        if self.rnn_type == "gru":
            dec_h = hidden[-1]  # last layer
        else:
            dec_h = hidden[0][-1]  # h from (h, c)
        
        # Attention context
        context = self.compute_attention(dec_h, encoder_outputs)
        
        # Decoder input: previous output + context
        dec_input = torch.cat([prev_output, context], dim=1).unsqueeze(1)
        
        dec_output, hidden = self.decoder(dec_input, hidden)
        dec_output = dec_output.squeeze(1)
        
        if self.probabilistic:
            mean = self.output_mean(dec_output)
            logvar = self.output_logvar(dec_output)
            return mean, logvar, hidden
        else:
            pred = self.output_fc(dec_output)
            return pred, None, hidden
    
    def forward(
        self,
        x: torch.Tensor,
        target_len: int = 8,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode input sequence, decode target_len steps.
        
        Parameters
        ----------
        x : (batch, input_seq_len, input_dim) — past observations
        target_len : number of future timesteps to predict
        target : (batch, target_len, output_dim) — ground truth for teacher forcing
        teacher_forcing_ratio : probability of using teacher forcing
        
        Returns
        -------
        Dict with:
            'predictions': (batch, target_len, output_dim) — predicted means
            'log_variances': (batch, target_len, output_dim) — log variances (if probabilistic)
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode
        encoder_outputs, hidden = self.encode(x)
        
        # If encoder is bidirectional but decoder is not, we need to transform hidden
        if self.encoder.bidirectional:
            if self.rnn_type == "gru":
                # Reshape from (num_layers*2, batch, hidden) to (num_layers, batch, hidden*2)
                # Then project down
                hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
                hidden = hidden.mean(dim=1)  # average forward/backward
            else:
                h, c = hidden
                h = h.view(self.num_layers, 2, batch_size, self.hidden_dim).mean(dim=1)
                c = c.view(self.num_layers, 2, batch_size, self.hidden_dim).mean(dim=1)
                hidden = (h, c)
        
        # Initialize decoder with last input's output features
        prev_output = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Collect predictions
        all_means = []
        all_logvars = []
        
        for t in range(target_len):
            mean, logvar, hidden = self.decode_step(
                prev_output, hidden, encoder_outputs
            )
            
            all_means.append(mean)
            if logvar is not None:
                all_logvars.append(logvar)
            
            # Teacher forcing
            if target is not None and np.random.random() < teacher_forcing_ratio:
                prev_output = target[:, t, :]
            else:
                prev_output = mean.detach()
        
        result = {
            "predictions": torch.stack(all_means, dim=1),  # (batch, target_len, output_dim)
        }
        
        if all_logvars:
            result["log_variances"] = torch.stack(all_logvars, dim=1)
        
        return result
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        target_len: int = 8,
        n_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Monte Carlo prediction for uncertainty estimation.
        Runs multiple forward passes with dropout enabled.
        
        Returns
        -------
        Dict with:
            'mean': (target_len, output_dim) — ensemble mean
            'std': (target_len, output_dim) — ensemble standard deviation
            'samples': (n_samples, target_len, output_dim) — all samples
        """
        self.train()  # enable dropout for MC sampling
        
        all_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x.unsqueeze(0), target_len=target_len)
                all_preds.append(out["predictions"].squeeze(0))
        
        samples = torch.stack(all_preds, dim=0)
        
        self.eval()
        
        return {
            "mean": samples.mean(dim=0),
            "std": samples.std(dim=0),
            "samples": samples,
        }
