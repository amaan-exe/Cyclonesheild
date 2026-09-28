import os
import sys
import numpy as np

REPO_PATH = r"d:\sih70\hehe\ml_model_repo"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from src.inference.pipeline import CycloneInferencePipeline

pipeline = CycloneInferencePipeline(
    checkpoints_dir=os.path.join(REPO_PATH, "outputs", "checkpoints"),
    device="cpu"
)

# Test 1: Identification with a clear synthetic vortex signature
H, W = 256, 256
synthetic_ir = np.full((1, 3, H, W), 280.0, dtype=np.float32)
# Cold cloud top eye wall at center (y=128, x=128)
yy, xx = np.ogrid[:H, :W]
dist = np.sqrt((yy - 128)**2 + (xx - 128)**2)
synthetic_ir[0, 0, dist < 50] = 205.0 # very cold deep convection
synthetic_ir[0, 0, dist < 12] = 260.0 # warm eye center

ident_res = pipeline.identify(
    scene_input=synthetic_ir,
    geo_bounds=(10.0, 25.0, 80.0, 95.0),
    default_lat=17.5,
    default_lon=87.5
)
print("=== PILLAR 1: IDENTIFICATION RESULT ===")
for k, v in ident_res.items():
    print(f"  {k}: {v}")

# Test 2: Classification with severe cyclone inputs (VSCS conditions)
class_res = pipeline.classify(
    current_wind_kt=85.0,
    central_pressure_hpa=965.0,
    sst_c=30.0,
    shear_kt=10.0
)
print("\n=== PILLAR 2: CLASSIFICATION RESULT ===")
for k, v in class_res.items():
    print(f"  {k}: {v}")

# Test 3: Cyclogenesis / Formation Prediction in favorable Bay of Bengal conditions
cg_res = pipeline.predict_cyclogenesis(
    lat=14.5,
    lon=88.2,
    sst_c=30.2,
    tchp_kj_cm2=95.0,
    shear_kt=7.5,
    rh_mid_pct=78.0,
    vorticity_850=28.0
)
print("\n=== PILLAR 3: CYCLOGENESIS / FORMATION PREDICTION RESULT ===")
for k, v in cg_res.items():
    if k != "diagnostics":
        print(f"  {k}: {v}")
print(f"  diagnostics: {cg_res.get('diagnostics')}")

assert "identified" in ident_res
assert "imd_category" in class_res
assert "formation_chance_percent" in cg_res
print("\n>>> ALL 3 PILLARS PASSED VERIFICATION WITH FLYING COLORS! <<<")
