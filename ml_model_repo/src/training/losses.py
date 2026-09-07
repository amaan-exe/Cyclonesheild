"""
Cyclone Horizon — Loss Functions
Focal loss, Gaussian NLL, multi-task weighted loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in intensity classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Downweights easy examples, focuses on hard ones.
    Critical for rare categories like Super Cyclonic Storm.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : (batch, num_classes) — raw logits
        targets : (batch,) — class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood loss for probabilistic predictions.
    
    Enables the model to predict both mean and variance,
    which powers the uncertainty cone in the dashboard.
    
    NLL = 0.5 * (log(σ²) + (y - μ)² / σ²)
    """
    
    def __init__(self, reduction: str = "mean", min_variance: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.min_variance = min_variance
    
    def forward(
        self,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mean : (batch, ...) — predicted means
        log_variance : (batch, ...) — predicted log variances
        target : (batch, ...) — ground truth
        """
        variance = torch.exp(log_variance) + self.min_variance
        nll = 0.5 * (log_variance + (target - mean) ** 2 / variance)
        
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        return nll


class MultiTaskLoss(nn.Module):
    """
    Weighted combination of multiple task losses.
    Supports learnable task weights (uncertainty weighting).
    """
    
    def __init__(self, task_weights: dict, learnable: bool = False):
        super().__init__()
        self.task_names = list(task_weights.keys())
        
        if learnable:
            # Learnable log-variance weights (Kendall et al., 2018)
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1))
                for name in self.task_names
            })
        else:
            self.fixed_weights = task_weights
            self.log_vars = None
    
    def forward(self, losses: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        losses : dict of task_name → scalar loss
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for name in self.task_names:
            if name not in losses:
                continue
            
            if self.log_vars is not None:
                # Uncertainty weighting: L_total = Σ (1/(2σ²)) * L_i + log(σ)
                precision = torch.exp(-self.log_vars[name])
                total = total + precision * losses[name] + self.log_vars[name]
            else:
                total = total + self.fixed_weights[name] * losses[name]
        
        return total
