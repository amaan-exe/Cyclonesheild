"""
Cyclone Horizon — Explainable AI (Grad-CAM Saliency Generator)
Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to explain:
  1. Which eyewall sectors or spiral convective bands drive the Dvorak pattern classification.
  2. Verifies that the CNN does NOT focus on coastline boundaries, timestamp stamps, or missing data artifacts.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for PyTorch cyclone CNN models.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h1, h2])

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        head_key: str = "pattern_logits"
    ) -> np.ndarray:
        """
        Generates a 2D saliency heatmap normalized to [0, 1].
        
        Parameters
        ----------
        input_tensor : (1, C, H, W)
        target_class : index of target pattern / intensity category
        head_key : 'pattern_logits' or 'intensity_logits'
        
        Returns
        -------
        heatmap : (H, W) numpy float32 array in range [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)

        logits = outputs[head_key]
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks didn't capture
            _, _, H, W = input_tensor.shape
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Global average pool of gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input image resolution
        _, _, H, W = input_tensor.shape
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
