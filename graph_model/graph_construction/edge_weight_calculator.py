"""
Computes dynamic edge weights combining:
  - Static base weights from topology
  - Dynamic transfer volume (from Cassandra rolling counts)
  - Normalized to [0, 1] per edge type
"""
import torch
import logging
from typing import Optional, Dict
from simulation.hospital_topology import Hospital

logger = logging.getLogger(__name__)

# Static weight by edge type
EDGE_TYPE_WEIGHTS = {
    "TRANSFER":        1.0,
    "SHARED_RESOURCE": 0.8,
    "PROXIMITY":       0.4,
}


class EdgeWeightCalculator:
    def __init__(self, hospital: Hospital):
        self.hospital = hospital
        self.dept_to_idx = {d.dept_id: i for i, d in enumerate(hospital.departments)}
        self._base_weights = self._precompute_base_weights()

    def _precompute_base_weights(self) -> Dict[tuple, float]:
        weights = {}
        for edge in self.hospital.edges:
            s = self.dept_to_idx.get(edge.source)
            t = self.dept_to_idx.get(edge.target)
            if s is None or t is None:
                continue
            type_weight = EDGE_TYPE_WEIGHTS.get(edge.edge_type, 0.5)
            w = edge.base_weight * type_weight
            weights[(s, t)] = w
            if edge.edge_type in ("SHARED_RESOURCE", "PROXIMITY"):
                weights[(t, s)] = w
        return weights

    def compute_weights(
        self,
        edge_index: torch.Tensor,
        transfer_counts: Optional[Dict[tuple, float]] = None,
    ) -> torch.Tensor:
        """
        Returns edge_attr tensor of shape [num_edges, 1].
        Blends static base weight with optional dynamic transfer volume.
        """
        num_edges = edge_index.shape[1]
        weights = torch.zeros(num_edges, 1)

        for i in range(num_edges):
            src = edge_index[0, i].item()
            tgt = edge_index[1, i].item()
            base = self._base_weights.get((src, tgt), 0.3)

            if transfer_counts is not None:
                dynamic = transfer_counts.get((src, tgt), 0.0)
                # Blend: 60% base + 40% normalized dynamic
                max_transfers = max(transfer_counts.values()) if transfer_counts else 1.0
                dynamic_norm = dynamic / max(max_transfers, 1.0)
                weight = 0.6 * base + 0.4 * dynamic_norm
            else:
                weight = base

            weights[i, 0] = weight

        # L1-normalize per source node to prevent exploding GNN aggregation
        for node in range(edge_index.max().item() + 1):
            mask = edge_index[0] == node
            if mask.sum() > 0:
                weights[mask] = weights[mask] / weights[mask].sum().clamp(min=1e-8)

        return weights
