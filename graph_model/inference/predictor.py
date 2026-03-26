"""
Real-time inference engine.
Loads a trained checkpoint and produces congestion forecasts from live features.
"""
import logging
import torch
from typing import Dict, List, Optional
from datetime import datetime

from graph_model.model.ed_forecast_model import EDForecastModel
from graph_model.model.model_config import ModelConfig
from graph_model.graph_construction.hospital_graph_builder import HospitalGraphBuilder
from simulation.hospital_topology import get_hospital_map, Hospital

logger = logging.getLogger(__name__)

HORIZON_LABELS = ["1h", "2h", "4h", "8h"]
SEVERITY_THRESHOLDS = {
    "green":  0.60,
    "amber":  0.80,
    "red":    1.01,  # everything above amber
}


def get_severity(score: float) -> str:
    if score < SEVERITY_THRESHOLDS["green"]:
        return "green"
    elif score < SEVERITY_THRESHOLDS["amber"]:
        return "amber"
    return "red"


class Predictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.hospital_map = get_hospital_map()
        self.builders: Dict[str, HospitalGraphBuilder] = {
            hid: HospitalGraphBuilder(h)
            for hid, h in self.hospital_map.items()
        }

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", ModelConfig())
        self.config = config

        self.model = EDForecastModel(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded model from {checkpoint_path} (val_loss={checkpoint.get('val_loss', 'N/A'):.4f})")

    @torch.no_grad()
    def predict(
        self,
        hospital_id: str,
        feature_sequence: List[Dict],   # T dicts: {dept_id → feature_vector}
        transfer_counts: Optional[Dict] = None,
    ) -> Dict:
        """
        Run inference for one hospital.
        Returns dict with per-department forecasts.
        """
        builder = self.builders.get(hospital_id)
        if builder is None:
            raise ValueError(f"Unknown hospital: {hospital_id}")

        graph_seq = builder.build_graph_sequence(feature_sequence, transfer_counts)
        # Wrap in batch dimension
        graph_seq_batch = [graph_seq]

        predictions = self.model(graph_seq_batch, num_nodes=builder.num_nodes)
        # predictions: [1, N, num_horizons] → squeeze batch dim
        predictions = predictions.squeeze(0).cpu()  # [N, num_horizons]

        # Build response
        result = {
            "hospital_id": hospital_id,
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "departments": [],
        }

        for i, dept in enumerate(builder.dept_list):
            scores = predictions[i].tolist()
            max_score = max(scores)
            dept_result = {
                "dept_id":   dept.dept_id,
                "dept_name": dept.dept_name,
                "dept_type": dept.dept_type,
                "forecasts": {
                    label: round(score, 4)
                    for label, score in zip(HORIZON_LABELS, scores)
                },
                "max_congestion": round(max_score, 4),
                "severity_label": get_severity(max_score),
            }
            result["departments"].append(dept_result)

        # Overall hospital severity
        all_max = [d["max_congestion"] for d in result["departments"]]
        result["hospital_max_congestion"] = round(max(all_max), 4)
        result["hospital_severity"] = get_severity(result["hospital_max_congestion"])

        return result
