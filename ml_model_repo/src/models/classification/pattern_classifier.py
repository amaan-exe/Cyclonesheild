"""
Cyclone Horizon — Multi-Task Dvorak Pattern & Intensity Classifier
Multi-task deep convolutional neural network for tropical cyclone structural analysis:
  1. Dvorak Pattern Type (6 classes: Curved Band, Shear, Eye, CDO, Embedded Center, Annular)
  2. Continuous Dvorak T-Number Regression (1.0 to 8.0 scale)
  3. IMD Operational Intensity Category (8-class IMD scale: LPA to SuCS)

Architecture:
  - Backbone: Torchvision EfficientNet-B3 / ResNet-50 with multi-channel support (IR + WV + Microwave)
  - 512-dimensional shared latent representation
  - Multi-task heads with Focal Loss & Label Smoothing
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    _BaseModule = nn.Module
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False
    _BaseModule = object

try:
    import torchvision.models as tv_models
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False

from src.utils.constants import NUM_IMD_CATEGORIES, NUM_DVORAK_PATTERNS


class FocalLoss(_BaseModule):
    """Focal loss for multi-class classification to counter severe class imbalance."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[Any] = None, label_smoothing: float = 0.0):
        if TORCH_AVAILABLE:
            super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs: Any, targets: Any) -> Any:
        if not TORCH_AVAILABLE:
            return 0.0
        ce_loss = F.cross_entropy(inputs, targets, label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class LightweightBackbone(_BaseModule):
    """Fallback high-performance ConvNet when torchvision pretrained weights are not downloaded."""
    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        if TORCH_AVAILABLE:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            self.features = None

    def forward(self, x: Any) -> Any:
        if not TORCH_AVAILABLE or self.features is None:
            return np.zeros((1, 512), dtype=np.float32)
        feat = self.features(x)
        return torch.flatten(feat, 1)


class CycloneClassifier(_BaseModule):
    """
    Multi-task deep CNN for cyclone structure and intensity.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b3",
        pretrained: bool = False,
        num_patterns: int = NUM_DVORAK_PATTERNS,
        num_intensities: int = NUM_IMD_CATEGORIES,
        input_channels: int = 3,
        dropout: float = 0.3
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        self.num_patterns = num_patterns
        self.num_intensities = num_intensities
        self.input_channels = input_channels

        if not TORCH_AVAILABLE:
            self.backbone = None
            return

        # Build backbone
        feature_dim = 1536 if backbone_name == "efficientnet_b3" else 512
        if HAS_TORCHVISION and backbone_name == "efficientnet_b3":
            try:
                base = tv_models.efficientnet_b3(weights=tv_models.EfficientNet_B3_Weights.DEFAULT if pretrained else None)
                # Modify first conv if input_channels != 3
                if input_channels != 3:
                    orig_conv = base.features[0][0]
                    base.features[0][0] = nn.Conv2d(
                        input_channels, orig_conv.out_channels,
                        kernel_size=orig_conv.kernel_size,
                        stride=orig_conv.stride,
                        padding=orig_conv.padding,
                        bias=False
                    )
                self.backbone = base.features
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                feature_dim = 1536
                self.target_conv_layer = self.backbone[-1]
            except Exception:
                self.backbone = LightweightBackbone(input_channels, feature_dim=512)
                self.global_pool = nn.Identity()
                feature_dim = 512
                self.target_conv_layer = self.backbone.features[-3]
        else:
            self.backbone = LightweightBackbone(input_channels, feature_dim=512)
            self.global_pool = nn.Identity()
            feature_dim = 512
            self.target_conv_layer = self.backbone.features[-3]

        # Shared representation projection
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Head 1: Dvorak Pattern (6 classes)
        self.pattern_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_patterns)
        )

        # Head 2: Dvorak T-Number Regression [1.0, 8.0]
        self.t_number_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Head 3: IMD Intensity (8 categories)
        self.intensity_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_intensities)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dimensional latent feature embeddings."""
        feat = self.backbone(x)
        feat = self.global_pool(feat)
        feat = torch.flatten(feat, 1)
        shared = self.shared_fc(feat)
        return shared

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat = self.global_pool(feat)
        feat = torch.flatten(feat, 1)
        shared = self.shared_fc(feat)

        pattern_logits = self.pattern_head(shared)
        t_raw = self.t_number_head(shared)
        intensity_logits = self.intensity_head(shared)

        # Scale T-number from [0, 1] -> [1.0, 8.0]
        t_number = t_raw * 7.0 + 1.0

        return {
            "pattern_logits": pattern_logits,
            "pattern_probs": F.softmax(pattern_logits, dim=-1),
            "t_number": t_number,
            "intensity_logits": intensity_logits,
            "intensity_probs": F.softmax(intensity_logits, dim=-1),
            "features": shared
        }


class MultiTaskLoss(_BaseModule):
    """Computes weighted loss across pattern, T-number, and IMD intensity."""
    def __init__(
        self,
        pattern_weight: float = 1.0,
        t_number_weight: float = 0.5,
        intensity_weight: float = 1.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        if TORCH_AVAILABLE:
            super().__init__()
            self.pattern_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
            self.t_number_loss = nn.SmoothL1Loss()  # Huber loss for outlier robustness
            self.intensity_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        self.weights = {
            "pattern": pattern_weight,
            "t_number": t_number_weight,
            "intensity": intensity_weight
        }

    def forward(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {"loss": 0.0}
        l_pat = self.pattern_loss(outputs["pattern_logits"], targets["pattern"])
        l_t = self.t_number_loss(outputs["t_number"].squeeze(-1), targets["t_number"])
        l_int = self.intensity_loss(outputs["intensity_logits"], targets["intensity"])

        total = (
            self.weights["pattern"] * l_pat +
            self.weights["t_number"] * l_t +
            self.weights["intensity"] * l_int
        )

        return {
            "loss": total,
            "loss_pattern": l_pat,
            "loss_t_number": l_t,
            "loss_intensity": l_int
        }
