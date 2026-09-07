"""
Verify and evaluate all 5 trained model checkpoints in outputs/checkpoints.
Checks:
- Checkpoint integrity & loadability
- Model architecture & parameter counts
- Real inference / forward pass validation
- Actual performance metrics
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import joblib

def check_vortex_detector(ckpt_dir):
    p = ckpt_dir / "vortex_detector_model.pt"
    print(f"\n[1/5] Checking Vortex Detector: {p}")
    if not p.exists():
        return {"status": "MISSING"}
    
    from src.models.detection.vortex_detector import CycloneVortexDetector
    state_dict = torch.load(p, map_location="cpu", weights_only=False)
    model = CycloneVortexDetector(in_channels=3, base_channels=32)
    model.load_state_dict(state_dict)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test forward pass with realistic input
    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)
        detections = model.detect(dummy_input[0])
    
    result = {
        "status": "LOADED",
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "heatmap_shape": list(out["heatmap"].shape),
        "detections_count": len(detections),
        "top_confidence": float(detections[0].confidence) if detections else 0.0,
    }
    print(f"  Result: {result}")
    return result

def check_pattern_classifier(ckpt_dir):
    p = ckpt_dir / "pattern_classifier_real_ir.pt"
    print(f"\n[2/5] Checking Pattern Classifier: {p}")
    if not p.exists():
        return {"status": "MISSING"}
    
    from src.models.classification.pattern_classifier import CycloneClassifier
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    
    model = CycloneClassifier(input_channels=3, num_patterns=6, num_intensities=8)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        model.load_state_dict(ckpt)
        meta = {}
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)
        emb = model.extract_features(dummy_input)
    
    result = {
        "status": "LOADED",
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "total_params": total_params,
        "metadata": meta,
        "pattern_logits_shape": list(out["pattern_logits"].shape),
        "t_number_range": [float(out["t_number"].min()), float(out["t_number"].max())],
        "feature_dim": emb.shape[1],
    }
    print(f"  Result: {result}")
    return result

def check_hybrid_intensity(ckpt_dir):
    p = ckpt_dir / "hybrid_intensity_model.joblib"
    print(f"\n[3/5] Checking Hybrid Intensity Classifier: {p}")
    if not p.exists():
        return {"status": "MISSING"}
    
    model = joblib.load(p)
    
    # Test inference (512 visual + 6 environmental features: SST, SST excess, dPres, dWind, shear, lat)
    dummy_vis = np.random.randn(5, 512)
    dummy_env = np.random.randn(5, 6)
    preds = model.predict(dummy_vis, dummy_env)
    
    result = {
        "status": "LOADED",
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "is_fitted": getattr(model, "is_fitted", True),
        "rf_estimators": len(model.rf_classifier.estimators_) if hasattr(model, "rf_classifier") else None,
        "predicted_categories": preds["predicted_categories"].tolist(),
        "ri_probabilities": [round(float(x), 3) for x in preds["ri_probabilities"]],
    }
    print(f"  Result: {result}")
    return result

def check_track_predictor(ckpt_dir):
    p = ckpt_dir / "hybrid_predictor_imd.pt"
    print(f"\n[4/5] Checking Track Predictor: {p}")
    if not p.exists():
        return {"status": "MISSING"}
    
    from src.models.prediction.hybrid_predictor import HybridCyclonePredictor
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    
    model = HybridCyclonePredictor(input_dim=12, output_dim=4, hidden_dim=128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    meta = {k: (v if not isinstance(v, (np.ndarray, torch.Tensor)) else f"array shape {v.shape}") for k, v in ckpt.items() if k != "model_state_dict"}
    
    # Test prediction for lead times 6h, 12h, 18h, 24h
    dummy_past = torch.randn(2, 6, 12)
    with torch.no_grad():
        pred = model(dummy_past, future_steps=4)
    
    result = {
        "status": "LOADED",
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "total_params": total_params,
        "metadata": meta,
        "pred_mean_shape": list(pred["mean"].shape),
        "pred_std_shape": list(pred["std"].shape),
        "gate_weights_shape": list(pred["gate_weights"].shape),
    }
    print(f"  Result: {result}")
    return result

def check_rl_agent(ckpt_dir):
    p = ckpt_dir / "rl_correction_agent.pt"
    print(f"\n[5/5] Checking RL Correction Agent: {p}")
    if not p.exists():
        return {"status": "MISSING"}
    
    from src.models.rl_correction.correction_agent import RLForecastCorrectionAgent
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    
    agent = RLForecastCorrectionAgent(state_dim=8, action_dim=3, device=torch.device("cpu"))
    agent.net.load_state_dict(ckpt["net_state_dict"])
    agent.net.eval()
    
    total_params = sum(p.numel() for p in agent.net.parameters())
    meta = {k: v for k, v in ckpt.items() if k != "net_state_dict"}
    
    # Test action selection
    dummy_state = np.random.randn(8)
    action, value, log_prob = agent.select_action(dummy_state, deterministic=True)
    
    result = {
        "status": "LOADED",
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
        "total_params": total_params,
        "metadata": meta,
        "sample_action": [round(float(a), 4) for a in action],
        "sample_value": round(float(value), 4) if value is not None else None,
        "track_clip_deg": agent.max_track_nudge,
        "intensity_clip_kt": agent.max_intensity_nudge,
    }
    print(f"  Result: {result}")
    return result

def evaluate_on_real_data():
    print("\n" + "=" * 60)
    print("Evaluating models on real historical data...")
    print("=" * 60)
    
    data_path = PROJECT_ROOT / "data" / "features" / "ibtracs_ni_features.parquet"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "raw" / "ibtracs_ni.parquet"
    
    if not data_path.exists():
        print(f"Data path not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} records across {df['storm_id'].nunique()} storms.")
    
    from src.inference.pipeline import CycloneInferencePipeline
    pipeline = CycloneInferencePipeline.from_trained_checkpoints(str(PROJECT_ROOT / "outputs" / "checkpoints"))
    
    # Pick 5 sample storms with >= 10 fixes
    storm_counts = df.groupby("storm_id").size()
    eligible = storm_counts[storm_counts >= 10].index.tolist()
    sample_storms = eligible[:5]
    
    evaluation_results = []
    for sid in sample_storms:
        time_col = "timestamp" if "timestamp" in df.columns else ("time_utc" if "time_utc" in df.columns else df.columns[1])
        sdf = df[df["storm_id"] == sid].sort_values(time_col)
        storm_name = sdf["name"].iloc[0] if "name" in sdf.columns else (sdf["storm_name"].iloc[0] if "storm_name" in sdf.columns else sid)
        
        # Take first 6 fixes as history, next 4 fixes as ground truth (+6h, +12h, +18h, +24h)
        history = sdf.iloc[:6]
        ground_truth = sdf.iloc[6:10]
        
        try:
            forecast_dict = pipeline.predict_track(history, lead_hours=[6, 12, 18, 24])
            predictions = forecast_dict.get("predictions", [])
            
            # Find 24h forecast point
            pred_24_list = [p for p in predictions if p.get("lead_h") == 24 or p.get("lead_time_hours") == 24]
            if pred_24_list and len(ground_truth) > 0:
                pred_24 = pred_24_list[0]
                actual_24 = ground_truth.iloc[-1]
                
                from src.utils.geo import haversine_distance
                err_km = haversine_distance(
                    pred_24["lat"], pred_24["lon"],
                    actual_24["lat"], actual_24["lon"]
                )
                
                evaluation_results.append({
                    "storm_id": sid,
                    "storm_name": storm_name,
                    "pred_lat": round(pred_24["lat"], 2),
                    "pred_lon": round(pred_24["lon"], 2),
                    "actual_lat": round(actual_24["lat"], 2),
                    "actual_lon": round(actual_24["lon"], 2),
                    "error_24h_km": round(err_km, 1),
                    "raw_error_km": round(haversine_distance(pred_24.get("raw_lat", pred_24["lat"]), pred_24.get("raw_lon", pred_24["lon"]), actual_24["lat"], actual_24["lon"]), 1),
                    "pred_wind_kt": round(pred_24.get("wind_kt", 0), 1),
                    "actual_wind_kt": round(actual_24.get("max_wind_kt", actual_24.get("wind_kt", 0)), 1)
                })
        except Exception as e:
            print(f"Error evaluating storm {sid}: {e}")
            
    print("\nSample Storm Evaluation Results (+24h):")
    for res in evaluation_results:
        rl_diff = round(res['raw_error_km'] - res['error_24h_km'], 1)
        print(f"  Storm {res['storm_name']} ({res['storm_id']}): Raw={res['raw_error_km']}km | RL-Corrected={res['error_24h_km']}km (Delta={rl_diff:+}km)")
    
    if evaluation_results:
        mean_raw = np.mean([r["raw_error_km"] for r in evaluation_results])
        mean_err = np.mean([r["error_24h_km"] for r in evaluation_results])
        print(f"\nSummary across {len(evaluation_results)} real storms:")
        print(f"  Mean Raw 24h Track Error:          {mean_raw:.1f} km")
        print(f"  Mean RL-Corrected 24h Track Error: {mean_err:.1f} km")
        print(f"  Net RL Improvement:                {mean_raw - mean_err:+.1f} km")

def main():
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    print("=" * 60)
    print("Cyclone Horizon — Checkpoint Integrity & Evaluation Audit")
    print("=" * 60)
    
    results = {
        "vortex_detector": check_vortex_detector(ckpt_dir),
        "pattern_classifier": check_pattern_classifier(ckpt_dir),
        "hybrid_intensity": check_hybrid_intensity(ckpt_dir),
        "track_predictor": check_track_predictor(ckpt_dir),
        "rl_correction_agent": check_rl_agent(ckpt_dir),
    }
    
    evaluate_on_real_data()

if __name__ == "__main__":
    main()
