"""
Cyclone Horizon — Vortex Identification & Center Detection Model
Locates cyclonic vortices within wide-area satellite scenes and predicts:
  1. Sub-pixel circulation center coordinates (eye fix)
  2. Bounding box extent and estimated circulation radius
  3. Cyclone detection confidence score

Architecture:
  - Multi-scale convolutional feature extractor
  - Center probability heatmap head with Gaussian focal weighting
  - Radial scale regression head for gale-force cloud shield extent
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CenterFixResult:
    """Structure holding the detected cyclone center fix."""
    pixel_x: float
    pixel_y: float
    lat: float
    lon: float
    confidence: float
    radius_km: float
    bbox_xywh_pixels: Tuple[float, float, float, float]
    is_cyclonic: bool


class CycloneVortexDetector(nn.Module):
    """
    CenterNet-style fully convolutional detector for cyclonic vortex localization.
    
    Inputs:
        x: (batch, in_channels, H, W) regional or full-disk satellite imagery.
    Outputs:
        heatmap: (batch, 1, H//4, W//4) probability map of the vortex center.
        box_wh:  (batch, 2, H//4, W//4) normalized width and height of outer shield.
        offset:  (batch, 2, H//4, W//4) sub-pixel local offset (dx, dy).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        max_detections: int = 5,
        min_confidence: float = 0.45
    ):
        super().__init__()
        self.in_channels = in_channels
        self.max_detections = max_detections
        self.min_confidence = min_confidence

        # Encoder (downsample 4x via residual conv blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.res2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Context aggregation (dilation for broad spiral cloud recognition)
        self.dilated_context = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Output Heads (feature map is H//4, W//4)
        mid_ch = base_channels * 2
        # Head 1: Center Heatmap (Sigmoid activation)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1)
        )

        # Head 2: Bounding Box Size (Width, Height normalized to [0, 1])
        self.wh_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, kernel_size=1),
            nn.Sigmoid()
        )

        # Head 3: Sub-pixel Offset [-0.5, 0.5]
        self.offset_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, mid_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 2, kernel_size=1),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Prior probability bias initialization for focal loss stability
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.res2(feat)
        feat = self.dilated_context(feat)

        heatmap_raw = self.heatmap_head(feat)
        heatmap = torch.sigmoid(heatmap_raw)
        box_wh = self.wh_head(feat)
        offset = self.offset_head(feat) * 0.5  # [-0.5, +0.5] pixel offset

        return {
            "heatmap": heatmap,
            "box_wh": box_wh,
            "offset": offset,
            "features": feat
        }

    @torch.no_grad()
    def detect(
        self,
        scene_tensor: torch.Tensor,
        geo_bounds: Optional[Dict[str, float]] = None
    ) -> List[CenterFixResult]:
        """
        Runs vortex inference on a satellite scene and returns candidate center fixes.
        
        Parameters
        ----------
        scene_tensor : (1, in_channels, H, W) or (in_channels, H, W)
        geo_bounds : dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
        
        Returns
        -------
        List of CenterFixResult objects ordered by confidence.
        """
        self.eval()
        if scene_tensor.dim() == 3:
            scene_tensor = scene_tensor.unsqueeze(0)

        _, _, H, W = scene_tensor.shape
        outputs = self.forward(scene_tensor)
        heatmap = outputs["heatmap"][0, 0]  # (H//4, W//4)
        box_wh = outputs["box_wh"][0]       # (2, H//4, W//4)
        offset = outputs["offset"][0]       # (2, H//4, W//4)

        # 3x3 Max-Pooling Non-Maximum Suppression (NMS)
        hmax = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
        keep = (heatmap.unsqueeze(0).unsqueeze(0) == hmax) & (heatmap.unsqueeze(0).unsqueeze(0) > self.min_confidence)
        keep = keep.squeeze()

        peak_indices = torch.nonzero(keep, as_tuple=False)
        if len(peak_indices) == 0:
            # If no confident peak exceeds threshold, fallback to global max
            max_val = heatmap.max().item()
            idx = (heatmap == heatmap.max()).nonzero(as_tuple=False)[0]
            peak_indices = idx.unsqueeze(0)
            confidences = [max_val]
        else:
            confidences = [heatmap[py, px].item() for py, px in peak_indices]

        # Sort peaks by confidence descending
        sorted_order = np.argsort(confidences)[::-1][:self.max_detections]
        results = []

        default_bounds = geo_bounds or {
            "lat_min": 0.0, "lat_max": 30.0,
            "lon_min": 50.0, "lon_max": 100.0
        }

        for i in sorted_order:
            py, px = peak_indices[i]
            conf = confidences[i]
            dx = offset[0, py, px].item()
            dy = offset[1, py, px].item()
            bw = box_wh[0, py, px].item()
            bh = box_wh[1, py, px].item()

            # Sub-pixel coordinates mapped to original input scene resolution
            exact_x = (px.item() + dx) * 4.0
            exact_y = (py.item() + dy) * 4.0
            pixel_w = bw * W
            pixel_h = bh * H

            # Geographic mapping
            lat = default_bounds["lat_max"] - (exact_y / H) * (default_bounds["lat_max"] - default_bounds["lat_min"])
            lon = default_bounds["lon_min"] + (exact_x / W) * (default_bounds["lon_max"] - default_bounds["lon_min"])

            # Approximate physical radius in km (assuming ~4 km/pixel for standard regional scene)
            radius_km = max(30.0, float(math.sqrt(pixel_w * pixel_h) * 2.0))

            results.append(
                CenterFixResult(
                    pixel_x=float(exact_x),
                    pixel_y=float(exact_y),
                    lat=round(float(lat), 3),
                    lon=round(float(lon), 3),
                    confidence=round(float(conf), 4),
                    radius_km=round(radius_km, 1),
                    bbox_xywh_pixels=(
                        float(exact_x - pixel_w / 2),
                        float(exact_y - pixel_h / 2),
                        float(pixel_w),
                        float(pixel_h)
                    ),
                    is_cyclonic=bool(conf >= self.min_confidence)
                )
            )

        return results
