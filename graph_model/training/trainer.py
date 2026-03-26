"""
Training loop for the Graph-LSTM-Attention model.

Features:
  - AdamW optimizer with cosine annealing warm restarts
  - Gradient clipping (essential for LSTM stability)
  - Early stopping on validation loss
  - Checkpoint saving on best val loss
  - TensorBoard logging
"""
import os
import time
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from graph_model.model.ed_forecast_model import EDForecastModel
from graph_model.model.model_config import ModelConfig
from graph_model.model.loss_functions import AsymmetricCongestionLoss

logger = logging.getLogger(__name__)
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")


class Trainer:
    def __init__(
        self,
        model: EDForecastModel,
        config: ModelConfig,
        device: torch.device = None,
    ):
        self.model  = model
        self.config = config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,       # restart every 10 epochs
            T_mult=2,
        )
        self.criterion = AsymmetricCongestionLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.writer = SummaryWriter(log_dir=f"runs/ed_forecast_{int(time.time())}")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self._best_val_loss = float("inf")
        self._patience_counter = 0

    def train_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            graph_seqs, targets, num_nodes = batch
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                predictions = self.model(graph_seqs, num_nodes)
                loss = self.criterion(predictions, targets)

            self.scaler.scale(loss).backward()
            # Gradient clipping — critical for LSTM stability
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.scheduler.step()
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        mae_total = 0.0
        num_batches = 0

        for batch in val_loader:
            graph_seqs, targets, num_nodes = batch
            targets = targets.to(self.device)
            predictions = self.model(graph_seqs, num_nodes)
            loss = self.criterion(predictions, targets)
            mae = (predictions - targets).abs().mean()

            total_loss += loss.item()
            mae_total  += mae.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_mae  = mae_total  / max(num_batches, 1)
        self.writer.add_scalar("Loss/val", avg_loss, epoch)
        self.writer.add_scalar("MAE/val",  avg_mae,  epoch)
        logger.info(f"Epoch {epoch} — val_loss={avg_loss:.4f}, val_mae={avg_mae:.4f}")
        return avg_loss

    def train(self, train_loader, val_loader) -> str:
        """Full training loop with early stopping. Returns path to best checkpoint."""
        best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss   = self.validate(val_loader, epoch)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": self.config,
                }, best_ckpt_path)
                logger.info(f"  ✓ New best model saved (val_loss={val_loss:.4f})")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        return best_ckpt_path
