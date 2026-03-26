"""
PyTorch Dataset for the Graph-LSTM model.
Reads feature windows from Cassandra and builds (graph_sequence, target) pairs.
"""
import math
import logging
import torch
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

from simulation.hospital_topology import Hospital, HOSPITALS, get_hospital_map
from graph_model.graph_construction.hospital_graph_builder import HospitalGraphBuilder
from graph_model.model.model_config import ModelConfig

logger = logging.getLogger(__name__)

# Feature normalization stats (computed from training data; update periodically)
FEATURE_MEAN = [0.5, 5.0, 0.15, 30.0, 0.0, 0.3, 3.0, 0.3, 0.0, 0.0]
FEATURE_STD  = [0.2, 3.0, 0.10, 20.0, 0.5, 0.2, 2.0, 0.2, 0.7, 0.7]


class EDForecastDataset(Dataset):
    """
    Dataset that:
    1. Queries Cassandra for T=24 hourly feature windows per hospital per dept
    2. Builds a graph sequence (list of PyG Data)
    3. Returns (graph_sequence, target_occupancy, num_nodes)

    In demo mode (no Cassandra): generates synthetic data.
    """

    def __init__(
        self,
        config: ModelConfig,
        hospital_ids: List[str] = None,
        start_time: datetime = None,
        end_time: datetime = None,
        demo_mode: bool = True,         # Use synthetic data if True
        cassandra_session=None,
    ):
        self.config = config
        self.demo_mode = demo_mode
        self.cassandra = cassandra_session
        self.hospital_map = get_hospital_map()

        if hospital_ids:
            self.hospitals = [self.hospital_map[hid] for hid in hospital_ids if hid in self.hospital_map]
        else:
            self.hospitals = list(self.hospital_map.values())

        self.graph_builders = {
            h.hospital_id: HospitalGraphBuilder(h) for h in self.hospitals
        }

        if demo_mode:
            # Generate N synthetic samples per hospital
            self._samples = self._generate_synthetic_samples(num_per_hospital=200)
        else:
            self._samples = self._build_real_samples(start_time, end_time)

    def _generate_synthetic_samples(self, num_per_hospital: int):
        """Generate synthetic (graph_sequence, target) pairs for demo/testing."""
        samples = []
        for hospital in self.hospitals:
            builder = self.graph_builders[hospital.hospital_id]
            num_nodes = builder.num_nodes
            T = self.config.sequence_len

            for _ in range(num_per_hospital):
                # Simulate a congestion scenario
                base_occupancy = np.random.uniform(0.3, 0.9)
                trend = np.random.uniform(-0.02, 0.02)

                feature_seq = []
                for t in range(T):
                    dept_features = {}
                    for dept in hospital.departments:
                        occ = float(np.clip(base_occupancy + trend * t + np.random.normal(0, 0.05), 0, 1))
                        hour = (t % 24)
                        features = [
                            occ,                                              # occupancy_ratio
                            np.random.uniform(2, 10),                         # arrival_rate
                            np.random.uniform(0.05, 0.4),                     # severity_index
                            np.random.uniform(15, 60),                        # avg_wait_time
                            np.random.normal(0, 0.3),                         # los_deviation
                            np.random.uniform(0, 0.5),                        # weather_score
                            np.random.uniform(1, 8),                          # flu_index
                            np.random.uniform(0.1, 0.7),                      # traffic_score
                            math.sin(2 * math.pi * hour / 24),               # hour_sin
                            math.cos(2 * math.pi * hour / 24),               # hour_cos
                        ]
                        # Normalize
                        features = [(f - m) / max(s, 1e-8) for f, m, s in zip(features, FEATURE_MEAN, FEATURE_STD)]
                        dept_features[dept.dept_id] = features
                    feature_seq.append(dept_features)

                graph_seq = builder.build_graph_sequence(feature_seq)

                # Target: occupancy in next 1h, 2h, 4h, 8h per node
                future_occ = float(np.clip(base_occupancy + trend * (T + 2) + np.random.normal(0, 0.08), 0, 1))
                target = torch.full((num_nodes, self.config.num_horizons), future_occ)
                for h_idx in range(self.config.num_horizons):
                    noise = np.random.normal(0, 0.05 * (h_idx + 1))
                    target[:, h_idx] = float(np.clip(future_occ + noise, 0, 1))

                samples.append((graph_seq, target, num_nodes))
        return samples

    def _build_real_samples(self, start_time, end_time):
        """Build samples from Cassandra feature store."""
        # Implementation: query feature_windows table by time range
        # and build sliding window samples
        raise NotImplementedError("Cassandra-backed dataset requires running infrastructure")

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx) -> Tuple:
        return self._samples[idx]
