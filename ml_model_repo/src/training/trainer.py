"""
Cyclone Horizon — Unified Training Loop
Supports multi-task learning, SWA, early stopping, gradient clipping.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR

from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger

logger = get_logger("training.trainer")


class CycloneTrainer:
    """
    Unified training loop for all Cyclone Horizon models.
    
    Features:
    - Multi-task loss support
    - Stochastic Weight Averaging (SWA)
    - Cosine annealing LR schedule
    - Early stopping with patience
    - Gradient clipping
    - Checkpoint saving (best + periodic)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "outputs/checkpoints",
        experiment_name: str = "cyclone_predictor",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        
        ensure_dir(checkpoint_dir)
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        gradient_clip: float = 1.0,
        teacher_forcing_ratio: float = 0.5,
    ) -> float:
        """Train for one epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            output = self.model(
                x,
                target_len=y.size(1),
                target=y,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            
            predictions = output["predictions"]
            
            if "log_variances" in output:
                from src.training.losses import GaussianNLLLoss
                nll_loss = GaussianNLLLoss()
                loss = nll_loss(predictions, output["log_variances"], y)
            else:
                loss = self.loss_fn(predictions, y)
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch in val_loader:
            x, y = batch[0].to(self.device), batch[1].to(self.device)
            
            output = self.model(x, target_len=y.size(1))
            predictions = output["predictions"]
            
            if "log_variances" in output:
                from src.training.losses import GaussianNLLLoss
                nll_loss = GaussianNLLLoss()
                loss = nll_loss(predictions, output["log_variances"], y)
            else:
                loss = self.loss_fn(predictions, y)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        gradient_clip: float = 1.0,
        swa_start_epoch: int = 70,
        lr_scheduler=None,
        teacher_forcing_decay: float = 0.02,
    ) -> Dict:
        """
        Full training loop with all bells and whistles.
        
        Parameters
        ----------
        teacher_forcing_decay : reduce teacher forcing ratio by this per epoch
        
        Returns
        -------
        Dict with training history
        """
        logger.info(f"Starting training: {epochs} epochs, patience={patience}, device={self.device}")
        
        # SWA setup
        swa_model = None
        swa_scheduler = None
        if swa_start_epoch < epochs:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(self.optimizer, swa_lr=0.0001)
        
        teacher_forcing = 0.5
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Decay teacher forcing
            teacher_forcing = max(0.0, teacher_forcing - teacher_forcing_decay)
            
            # Train
            train_loss = self.train_epoch(
                train_loader,
                gradient_clip=gradient_clip,
                teacher_forcing_ratio=teacher_forcing,
            )
            
            # Validate
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # LR scheduling
            if epoch >= swa_start_epoch and swa_scheduler:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
            elif lr_scheduler:
                lr_scheduler.step()
            
            elapsed = time.time() - start_time
            
            # Logging
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {current_lr:.6f} | TF: {teacher_forcing:.2f} | "
                f"Time: {elapsed:.1f}s"
            )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                    break
            
            # Periodic checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_loss, is_best=False)
        
        # Update batch norm for SWA model
        if swa_model and swa_start_epoch < epochs:
            logger.info("Updating SWA batch norm statistics...")
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
            swa_path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_swa.pt")
            torch.save(swa_model.state_dict(), swa_path)
            logger.info(f"SWA model saved to {swa_path}")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.train_losses),
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pt")
        else:
            path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_epoch{epoch}.pt")
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path} (val_loss={val_loss:.6f})")
    
    def load_best(self):
        """Load the best checkpoint."""
        path = os.path.join(self.checkpoint_dir, f"{self.experiment_name}_best.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best checkpoint from {path}")
        else:
            logger.warning(f"No best checkpoint found at {path}")
