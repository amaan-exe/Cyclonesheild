"""
Cyclone Horizon — Polar Coordinate Resampling (Azimuth-Radius Transform)
Transforms storm-centered Cartesian satellite crops (X, Y) into polar coordinates (Radius, Azimuth).
Enables rotation-invariant feature learning: spiral arms map to linear diagonal bands,
and eyewall symmetry becomes a horizontal gradient track.
"""

import math
from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F


def cartesian_to_polar_grid(
    height: int,
    width: int,
    radial_steps: int = 128,
    angular_steps: int = 128,
    center: Optional[Tuple[float, float]] = None,
    max_radius: Optional[float] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generates a normalized sampling grid for torch.nn.functional.grid_sample.
    
    Parameters
    ----------
    height, width : dimensions of the original Cartesian crop
    radial_steps : number of concentric radial rings (output height)
    angular_steps : number of azimuth angles 0 to 2pi (output width)
    center : (cx, cy) in pixel coordinates; defaults to image center
    max_radius : maximum sampling radius; defaults to min(H, W) / 2
    """
    if center is None:
        cx, cy = width / 2.0, height / 2.0
    else:
        cx, cy = center

    if max_radius is None:
        max_radius = min(height, width) / 2.0

    r = torch.linspace(0.0, max_radius, radial_steps, device=device)
    theta = torch.linspace(0.0, 2.0 * math.pi, angular_steps, device=device)

    # Meshgrid of (r, theta)
    grid_r, grid_theta = torch.meshgrid(r, theta, indexing="ij")

    # Map to Cartesian coordinates
    x = cx + grid_r * torch.cos(grid_theta)
    y = cy + grid_r * torch.sin(grid_theta)

    # Normalize to [-1, 1] for PyTorch grid_sample
    norm_x = (x / (width - 1)) * 2.0 - 1.0
    norm_y = (y / (height - 1)) * 2.0 - 1.0

    grid = torch.stack((norm_x, norm_y), dim=-1)  # (radial_steps, angular_steps, 2)
    return grid


class PolarResampler:
    """
    PyTorch / NumPy module for transforming storm-centered crops into polar projections.
    """

    def __init__(
        self,
        radial_steps: int = 128,
        angular_steps: int = 128
    ):
        self.radial_steps = radial_steps
        self.angular_steps = angular_steps

    def __call__(
        self,
        image_tensor: torch.Tensor,
        center: Optional[Tuple[float, float]] = None,
        max_radius: Optional[float] = None
    ) -> torch.Tensor:
        """
        Resamples (batch, channels, H, W) into (batch, channels, radial_steps, angular_steps).
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        batch, channels, H, W = image_tensor.shape
        grid = cartesian_to_polar_grid(
            H, W,
            radial_steps=self.radial_steps,
            angular_steps=self.angular_steps,
            center=center,
            max_radius=max_radius,
            device=image_tensor.device
        )
        grid_batch = grid.unsqueeze(0).expand(batch, -1, -1, -1)

        # Bilinear interpolation resampling
        polar_tensor = F.grid_sample(
            image_tensor,
            grid_batch,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )
        return polar_tensor
