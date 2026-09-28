"""Train the track predictor checkpoint from storm-level IBTrACS features."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import CycloneInferencePipeline
from src.models.prediction.hybrid_predictor import HybridCyclonePredictor
from src.utils.geo import haversine_distance


FEATURES = [
    "lat", "lon", "max_wind_kt", "min_pressure_hpa", "sst", "shear_magnitude",
    "storm_speed_kph", "storm_bearing_deg", "time_since_genesis_hours",
    "delta_wind_6h", "delta_pressure_6h", "coriolis_proxy",
]


def build_sequences(frame, storm_ids):
    inputs, targets, anchors = [], [], []
    for storm_id in storm_ids:
        storm = frame[frame["storm_id"] == storm_id].sort_values("timestamp").reset_index(drop=True)
        if len(storm) < 14:
            continue
        values_frame = storm[FEATURES].apply(pd.to_numeric, errors="coerce")
        values_frame = values_frame.interpolate(limit_direction="both").fillna(0)
        coordinates_frame = storm[["lat", "lon", "max_wind_kt", "min_pressure_hpa"]].apply(pd.to_numeric, errors="coerce")
        coordinates_frame = coordinates_frame.interpolate(limit_direction="both").bfill().ffill()
        if coordinates_frame[["lat", "lon"]].isna().any().any():
            continue
        values = values_frame.to_numpy(dtype=np.float32)
        coordinates = coordinates_frame.fillna(0).to_numpy(dtype=np.float32)
        for index in range(len(storm) - 14):
            inputs.append(values[index:index + 6])
            anchor = coordinates[index + 5]
            target = np.zeros((8, 4), dtype=np.float32)
            for step in range(8):
                future = coordinates[index + 6 + step]
                target[step, 0:2] = future[0:2] - anchor[0:2]
                target[step, 2:4] = future[2:4]
            targets.append(target)
            anchors.append(anchor)
    return np.asarray(inputs), np.asarray(targets), np.asarray(anchors)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    data_path = PROJECT_ROOT / "data" / "features" / "ibtracs_ni_features.parquet"
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / "hybrid_predictor_imd.pt"
    frame = pd.read_parquet(data_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

    storm_ids = np.asarray(frame["storm_id"].dropna().unique(), dtype=object)
    np.random.shuffle(storm_ids)
    split = max(1, int(len(storm_ids) * 0.8))
    train_ids, test_ids = storm_ids[:split], storm_ids[split:]
    train_x, train_y, _ = build_sequences(frame, train_ids)
    test_x, test_y, test_anchors = build_sequences(frame, test_ids)
    if len(train_x) == 0 or len(test_x) == 0:
        raise RuntimeError("Insufficient storm sequences for a train/test checkpoint")

    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    device = torch.device("cpu")
    model = HybridCyclonePredictor(input_dim=12, output_dim=4, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=64,
        shuffle=True,
    )

    model.train()
    for epoch in range(30):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x.to(device), future_steps=8)
            coordinate_loss = loss_fn(output["mean"][:, :, :2], batch_y[:, :, :2]) * 15.0
            intensity_loss = loss_fn(output["mean"][:, :, 2:], batch_y[:, :, 2:]) * 0.05
            loss = coordinate_loss + intensity_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1:02d}/30 loss={total_loss / max(1, len(loader)):.5f}")

    model.eval()
    errors_24h, errors_48h = [], []
    with torch.no_grad():
        for start in range(0, len(test_x), 64):
            output = model(torch.from_numpy(test_x[start:start + 64]).to(device), future_steps=8)["mean"].numpy()
            anchors = test_anchors[start:start + 64]
            actual = test_y[start:start + 64]
            for index in range(len(output)):
                anchor = anchors[index]
                errors_24h.append(haversine_distance(
                    anchor[0] + output[index, 3, 0], anchor[1] + output[index, 3, 1],
                    anchor[0] + actual[index, 3, 0], anchor[1] + actual[index, 3, 1]))
                errors_48h.append(haversine_distance(
                    anchor[0] + output[index, 7, 0], anchor[1] + output[index, 7, 1],
                    anchor[0] + actual[index, 7, 0], anchor[1] + actual[index, 7, 1]))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": mean,
        "scaler_std": std,
        "mean_track_24h": float(np.mean(errors_24h)),
        "mean_track_48h": float(np.mean(errors_48h)),
        "train_storms": int(len(train_ids)),
        "test_storms": int(len(test_ids)),
        "train_sequences": int(len(train_x)),
        "test_sequences": int(len(test_x)),
        "training_data": "IBTrACS North Indian Ocean best-track features",
        "training_type": "supervised storm-level split",
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }, checkpoint_path)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_sequences": len(train_x),
        "test_sequences": len(test_x),
        "mean_track_error_24h_km": round(float(np.mean(errors_24h)), 2),
        "mean_track_error_48h_km": round(float(np.mean(errors_48h)), 2),
    }, indent=2))


if __name__ == "__main__":
    main()