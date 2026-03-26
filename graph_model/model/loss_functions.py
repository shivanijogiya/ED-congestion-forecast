"""
Custom loss functions for ED congestion forecasting.

1. Asymmetric weighted MSE: underpredicting high congestion is penalized 3x more
   (false negatives in an ED are clinically dangerous)

2. Threshold hinge loss: extra penalty when prediction crosses 0.8 threshold
   in the wrong direction (missing an overflow event is worse than a false alarm)
"""
import torch
import torch.nn as nn

CONGESTION_THRESHOLD = 0.8
UNDERPREDICTION_PENALTY = 3.0    # weight when pred < actual in high-congestion zone
HINGE_MARGIN = 0.05


class AsymmetricCongestionLoss(nn.Module):
    def __init__(
        self,
        threshold: float = CONGESTION_THRESHOLD,
        underprediction_penalty: float = UNDERPREDICTION_PENALTY,
        hinge_weight: float = 0.5,
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha = underprediction_penalty
        self.hinge_weight = hinge_weight

    def forward(
        self,
        predictions: torch.Tensor,  # [B, N, num_horizons]
        targets: torch.Tensor,       # [B, N, num_horizons]
        mask: torch.Tensor = None,   # optional [B, N] mask for missing data
    ) -> torch.Tensor:
        # ── Asymmetric MSE ────────────────────────────────────────────────────
        diff = predictions - targets  # positive → overprediction, negative → underprediction
        sq_err = diff ** 2

        # In high-congestion zones (actual > threshold), weight underprediction heavily
        high_congestion = (targets > self.threshold).float()
        underprediction = (diff < 0).float()
        weight = 1.0 + (self.alpha - 1.0) * high_congestion * underprediction

        weighted_mse = (weight * sq_err).mean()

        # ── Threshold hinge loss ──────────────────────────────────────────────
        # Penalize crossing the 0.8 threshold in the wrong direction
        # Case 1: actual > threshold but prediction < threshold - margin
        miss_overflow = (
            (targets > self.threshold)
            & (predictions < self.threshold - HINGE_MARGIN)
        ).float()
        # Case 2: actual < threshold but prediction > threshold + margin (false alarm)
        false_alarm = (
            (targets < self.threshold)
            & (predictions > self.threshold + HINGE_MARGIN)
        ).float()

        hinge = (
            miss_overflow * (self.threshold - predictions).clamp(min=0)
            + 0.2 * false_alarm * (predictions - self.threshold).clamp(min=0)
        ).mean()

        total = weighted_mse + self.hinge_weight * hinge

        if mask is not None:
            # Apply node-level mask (for hospitals with missing departments)
            mask_expanded = mask.unsqueeze(-1).expand_as(predictions).float()
            total = (total * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)

        return total
