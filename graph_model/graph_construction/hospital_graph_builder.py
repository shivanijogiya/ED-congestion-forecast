"""
Builds PyTorch Geometric Data objects from hospital topology + live features.
Each Data object represents one timestep snapshot of the hospital graph.
Returns a sequence of T Data objects for the LSTM.
"""
import torch
import logging
from typing import List, Optional
from torch_geometric.data import Data

from simulation.hospital_topology import Hospital, get_hospital_map
from graph_model.graph_construction.edge_weight_calculator import EdgeWeightCalculator

logger = logging.getLogger(__name__)

FEATURE_DIM = 10   # must match feature_engineering output


class HospitalGraphBuilder:
    def __init__(self, hospital: Hospital):
        self.hospital = hospital
        self.dept_list = hospital.departments
        self.dept_to_idx = {d.dept_id: i for i, d in enumerate(self.dept_list)}
        self.num_nodes = len(self.dept_list)
        self.edge_calculator = EdgeWeightCalculator(hospital)

        # Build static edge_index (shape [2, num_edges])
        self._static_edge_index, self._edge_types = self._build_edge_index()

    def _build_edge_index(self):
        """Convert edge list to COO format edge_index tensor."""
        src_list, dst_list, types = [], [], []
        for edge in self.hospital.edges:
            s = self.dept_to_idx.get(edge.source)
            t = self.dept_to_idx.get(edge.target)
            if s is not None and t is not None:
                src_list.append(s)
                dst_list.append(t)
                types.append(edge.edge_type)
                # Add reverse edge for undirected relationships
                if edge.edge_type in ("SHARED_RESOURCE", "PROXIMITY"):
                    src_list.append(t)
                    dst_list.append(s)
                    types.append(edge.edge_type)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return edge_index, types

    def build_graph_sequence(
        self,
        feature_sequence: List[dict],   # List of T dicts: {dept_id → feature_vector}
        transfer_counts: Optional[dict] = None,
    ) -> List[Data]:
        """
        Args:
            feature_sequence: T timesteps, each a dict mapping dept_id → 10-dim list
            transfer_counts:  optional recent transfer volume for dynamic edge weights
        Returns:
            List of T PyG Data objects
        """
        graph_seq = []
        for timestep_features in feature_sequence:
            x = self._build_node_features(timestep_features)
            edge_attr = self.edge_calculator.compute_weights(
                self._static_edge_index,
                transfer_counts=transfer_counts,
            )
            data = Data(
                x=x,
                edge_index=self._static_edge_index,
                edge_attr=edge_attr,
                num_nodes=self.num_nodes,
            )
            graph_seq.append(data)
        return graph_seq

    def _build_node_features(self, timestep_features: dict) -> torch.Tensor:
        """
        Assemble node feature matrix of shape [num_nodes, FEATURE_DIM].
        Uses zeros for departments with missing data.
        """
        x = torch.zeros(self.num_nodes, FEATURE_DIM)
        for dept_id, features in timestep_features.items():
            idx = self.dept_to_idx.get(dept_id)
            if idx is None:
                continue
            if isinstance(features, (list, tuple)) and len(features) == FEATURE_DIM:
                x[idx] = torch.tensor(features, dtype=torch.float32)
            elif isinstance(features, dict):
                x[idx] = torch.tensor([
                    features.get("occupancy_ratio", 0),
                    features.get("arrival_rate", 0),
                    features.get("severity_index", 0),
                    features.get("avg_wait_time", 0),
                    features.get("los_deviation", 0),
                    features.get("weather_score", 0),
                    features.get("flu_index", 0),
                    features.get("traffic_score", 0),
                    features.get("hour_sin", 0),
                    features.get("hour_cos", 0),
                ], dtype=torch.float32)
        return x
