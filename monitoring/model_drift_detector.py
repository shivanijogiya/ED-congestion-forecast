"""
Model drift detection using Population Stability Index (PSI).
Compares prediction distribution against actual occupancy ratios.
Triggers retraining when PSI > 0.2 or rolling MAE exceeds 2x training baseline.

Runs as a scheduled background job (every hour).
"""
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple

logger = logging.getLogger(__name__)

RETRAINING_PSI_THRESHOLD  = 0.2
MAE_MULTIPLIER_THRESHOLD  = 2.0
TRAINING_BASELINE_MAE     = 0.045   # Update after each training run
PSI_BINS = 10


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    Population Stability Index (PSI) between two distributions.
    PSI < 0.1: no significant change
    PSI 0.1-0.2: moderate change, monitor
    PSI > 0.2: significant shift, trigger retraining
    """
    epsilon = 1e-8
    # Bin using expected distribution boundaries
    breakpoints = np.linspace(0, 1, bins + 1)
    expected_fracs = np.histogram(expected, bins=breakpoints)[0] / (len(expected) + epsilon)
    actual_fracs   = np.histogram(actual,   bins=breakpoints)[0] / (len(actual)   + epsilon)

    # Avoid division by zero
    expected_fracs = np.where(expected_fracs == 0, epsilon, expected_fracs)
    actual_fracs   = np.where(actual_fracs   == 0, epsilon, actual_fracs)

    psi = np.sum((actual_fracs - expected_fracs) * np.log(actual_fracs / expected_fracs))
    return float(psi)


def compute_rolling_mae(
    predictions: List[float],
    actuals: List[float],
) -> float:
    if not predictions or not actuals:
        return 0.0
    n = min(len(predictions), len(actuals))
    return float(np.mean(np.abs(np.array(predictions[:n]) - np.array(actuals[:n]))))


class DriftDetector:
    def __init__(self, baseline_mae: float = TRAINING_BASELINE_MAE):
        self.baseline_mae = baseline_mae
        self._prediction_history: List[float] = []
        self._actual_history: List[float] = []
        self._retrain_callbacks = []

    def register_retrain_callback(self, fn):
        """Register a function to call when drift is detected."""
        self._retrain_callbacks.append(fn)

    def record(self, prediction: float, actual: float):
        self._prediction_history.append(prediction)
        self._actual_history.append(actual)
        # Keep rolling 24h window (assuming records every 5 min → 288 records)
        if len(self._prediction_history) > 288:
            self._prediction_history.pop(0)
            self._actual_history.pop(0)

    def check_drift(self) -> dict:
        if len(self._prediction_history) < 50:
            return {"status": "insufficient_data", "records": len(self._prediction_history)}

        psi = compute_psi(
            np.array(self._prediction_history),
            np.array(self._actual_history),
        )
        mae = compute_rolling_mae(self._prediction_history, self._actual_history)
        mae_ratio = mae / max(self.baseline_mae, 1e-8)

        status = "stable"
        if psi > RETRAINING_PSI_THRESHOLD or mae_ratio > MAE_MULTIPLIER_THRESHOLD:
            status = "drift_detected"
            logger.warning(
                f"Model drift detected! PSI={psi:.3f}, MAE={mae:.4f} (ratio={mae_ratio:.1f}x)"
            )
            self._trigger_retraining(psi, mae)

        return {
            "status":         status,
            "psi":            round(psi, 4),
            "rolling_mae":    round(mae, 4),
            "mae_ratio":      round(mae_ratio, 2),
            "checked_at":     datetime.utcnow().isoformat(),
            "records_checked": len(self._prediction_history),
        }

    def _trigger_retraining(self, psi: float, mae: float):
        for cb in self._retrain_callbacks:
            try:
                cb(psi=psi, mae=mae, triggered_at=datetime.utcnow())
            except Exception as e:
                logger.error(f"Retrain callback failed: {e}")
