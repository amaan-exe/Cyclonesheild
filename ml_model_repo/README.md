# Cyclone Horizon — ML System for Tropical Cyclone Identification, Classification & Prediction

**SIH 2026 | Problem Statement 26070 | Ministry of Earth Sciences (IMD)**

An AI/ML system to identify, classify, and predict tropical cyclone patterns from multi-source satellite data, with a focus on the North Indian Ocean basin.

---

## Quick Start

### 1. Install dependencies

```bash
# Install PyTorch with CUDA support (RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Fetch IBTrACS data (free, instant — no registration needed)

```bash
python scripts/fetch_ibtracs.py
```

This downloads North Indian Ocean best-track data, runs QC checks, computes features, checks for data leakage, and splits the dataset.

### 3. Train & Evaluate Models

```bash
python scripts/train_and_evaluate_all.py
```

Trains and benchmarks all ML models:
- Identification: U-Net Vortex Center Localizer
- Pattern: Multi-task Dvorak CNN with Grad-CAM explainability
- Intensity: Hybrid Visual-Thermodynamic Stacking Classifier
- Track: Physics-informed Beta-Advection + GRU Sequence Predictor
- Correction: RL Offline Policy Correction Agent

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Full pipeline (all stages)

```bash
python scripts/run_pipeline.py --config config/pipeline_config.yaml

# Or individual stages:
python scripts/run_pipeline.py --stage ingest
python scripts/run_pipeline.py --stage validate
python scripts/run_pipeline.py --stage features
python scripts/run_pipeline.py --stage split

# Dry run (shows what would happen without doing it):
python scripts/run_pipeline.py --dry-run
```

---

## Architecture

```
Input Data → Ingestion → Validation/QC → Preprocessing → Feature Engineering → Dataset Assembly
                                                                                      ↓
Metrics & Reports ← Evaluation ← Inference Pipeline ← Trained Models ← Training Loop ← Split Dataset
```

### ML Components

| Component | Architecture | Status |
|-----------|-------------|--------|
| **Track Prediction** | GRU seq-to-seq with attention + probabilistic output | 🟢 Ready (IBTrACS) |
| **Intensity Classification** | EfficientNet-B3 multi-task CNN | 🟡 Architecture ready (needs imagery) |
| **Pattern Classification** | Dvorak pattern types → CNN heads | 🟡 Architecture ready (needs imagery) |
| **Detection** | U-Net segmentation | 🟡 Designed |
| **RL Correction** | PPO/CQL offline agent | 🔴 Stretch goal |

### Data Sources

| Source | Status | Access |
|--------|--------|--------|
| **IBTrACS** (NOAA) | 🟢 Working | Public, instant |
| **ERA5** (ECMWF) | 🟢 Connector ready | Free CDS account needed |
| **OISST** (NOAA) | 🟢 Connector ready | Public |
| **MOSDAC** (INSAT) | 🔴 Pending approval | Registration under review |
| **GOES-16** (AWS) | 🟡 Available as proxy | AWS Open Data |

---

## Project Structure

```
ML/
├── config/                  # YAML configs (all params, nothing hardcoded)
├── src/
│   ├── ingestion/           # Data source connectors
│   ├── validation/          # QC checks & reporting
│   ├── preprocessing/       # Calibration, centering, normalization
│   ├── features/            # Feature engineering (all causal/past-only)
│   ├── dataset/             # Storm-level splitting, versioned export
│   ├── models/              # Model architectures
│   │   ├── prediction/      # LSTM/GRU track predictor
│   │   ├── classification/  # CNN pattern/intensity classifier
│   │   ├── detection/       # U-Net segmentation
│   │   └── rl_correction/   # RL forecast correction agent & environment
│   ├── training/            # Training loop, losses, metrics
│   ├── inference/           # Standalone inference pipeline & wind radii
│   ├── evaluation/          # Meteorological skill metrics & verification
│   ├── visualization/       # Spatial maps & Grad-CAM plots
│   └── utils/               # Constants, geo, logging, I/O
├── tests/                   # Core ML unit & integration tests
├── scripts/                 # CLI tools
├── data/                    # Raw/processed/features/datasets (gitignored)
└── outputs/                 # Checkpoints/predictions/reports (gitignored)
```

---

## Key Design Decisions

1. **Storm-level splitting**: Train/val/test split by storm event, never by frame — prevents data leakage
2. **Causal features only**: Every feature computable from past data only, with automated leakage checks
3. **Probabilistic output**: Predictions include mean + variance → uncertainty cone
4. **Physics-hybrid track model**: Steering flow + ML correction, not pure black-box
5. **Focal loss**: For class-imbalanced IMD categories (Super Cyclonic Storm is rare)

---

## Known Limitations (stated honestly)

- **Rapid intensification forecasting** is an open research problem industry-wide — our model will inherit this limitation
- **MOSDAC access pending** — CNN classifier trained on proxy data until INSAT imagery available
- **RL correction module** is a proof-of-concept on small offline data, not a production system
- **Track prediction baselines**: Compare against IMD CPS published numbers (74km at 12h, 200km at 72h)
- **NIO data scarcity**: ~4% of global cyclones — we use transfer learning from other basins
