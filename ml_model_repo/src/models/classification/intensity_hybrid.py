"""
Cyclone Horizon — Hybrid Intensity & Rapid Intensification (RI) Classifier
Fuses 512-dimensional deep CNN visual embeddings with tabular thermodynamic features:
  - SST (°C) and thermal excess above 26.5°C threshold
  - Vertical Wind Shear (850 - 200 hPa in knots)
  - 6-hour pressure change rate (dP/dt) and 6-hour wind delta (dV/dt)
  - Coriolis parameter f = 2 * Omega * sin(lat)
  - Forward translation velocity

Dual Predictor:
  1. IMD Operational Category Stacking (8 classes with calibrated probabilities)
  2. Dedicated Rapid Intensification (RI) Binary Detector (dP >= 30 kt gain in 24h)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


class HybridIntensityClassifier:
    """
    Hybrid multi-modal intensity and Rapid Intensification classifier.
    Combines deep visual representations with environmental physics.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        use_xgboost: bool = False,
        random_state: int = 42
    ):
        self.visual_dim = visual_dim
        self.random_state = random_state
        self.scaler = StandardScaler()

        # 1. Multi-class IMD intensity model
        self.intensity_model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.5,
            random_state=random_state
        )

        # 2. Binary Rapid Intensification (RI) detector (class-weighted for rare RI events)
        self.ri_model = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=21,
            class_weight="balanced",
            random_state=random_state
        )

        self.is_fitted = False

    def _prepare_features(
        self,
        visual_embeddings: np.ndarray,
        environmental_features: np.ndarray,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Combines and normalizes visual embeddings and environmental tabular features.
        """
        if visual_embeddings.ndim == 1:
            visual_embeddings = visual_embeddings.reshape(1, -1)
        if environmental_features.ndim == 1:
            environmental_features = environmental_features.reshape(1, -1)

        fused = np.hstack([visual_embeddings, environmental_features])

        if fit_scaler:
            return self.scaler.fit_transform(fused)
        return self.scaler.transform(fused)

    def fit(
        self,
        visual_embeddings: np.ndarray,
        environmental_features: np.ndarray,
        intensity_labels: np.ndarray,
        ri_labels: Optional[np.ndarray] = None
    ):
        """
        Fits both the multi-class IMD intensity model and the RI detector.
        """
        X = self._prepare_features(visual_embeddings, environmental_features, fit_scaler=True)

        # Fit multi-class IMD intensity
        self.intensity_model.fit(X, intensity_labels)

        # Generate or fit binary Rapid Intensification labels
        if ri_labels is None:
            # Synthetic proxy: high category jumps
            ri_labels = np.zeros(len(intensity_labels), dtype=int)
            for i in range(1, len(intensity_labels)):
                if intensity_labels[i] - intensity_labels[i-1] >= 2:
                    ri_labels[i] = 1

        self.ri_model.fit(X, ri_labels)
        self.is_fitted = True

    def predict(
        self,
        visual_embeddings: np.ndarray,
        environmental_features: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Predicts calibrated IMD category probabilities and Rapid Intensification probability.
        """
        if not self.is_fitted:
            raise RuntimeError("HybridIntensityClassifier must be fitted before predict().")

        X = self._prepare_features(visual_embeddings, environmental_features, fit_scaler=False)

        intensity_preds = self.intensity_model.predict(X)
        intensity_probs = self.intensity_model.predict_proba(X)

        ri_probs = self.ri_model.predict_proba(X)[:, 1] if hasattr(self.ri_model, "predict_proba") else np.zeros(len(X))
        ri_preds = (ri_probs >= 0.40).astype(int)  # Lower threshold for high-recall RI alerting

        return {
            "predicted_categories": intensity_preds,
            "category_probabilities": intensity_probs,
            "ri_probabilities": ri_probs,
            "ri_alert_active": ri_preds,
            "features_used": X.shape[1]
        }
