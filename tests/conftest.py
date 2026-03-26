"""Shared test fixtures."""
import pytest
from simulation.hospital_topology import HOSPITALS

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def sample_hospital():
    return HOSPITALS[0]


@pytest.fixture
def sample_graph(sample_hospital):
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    n = len(sample_hospital.departments)
    e = len(sample_hospital.edges) * 2
    return Data(
        x=torch.randn(n, 10),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.rand(e, 1),
        num_nodes=n,
    )


@pytest.fixture
def sample_graph_sequence(sample_hospital):
    """Returns 24 graph snapshots for testing LSTM."""
    if not HAS_TORCH:
        pytest.skip("torch not installed")
    from graph_model.graph_construction.hospital_graph_builder import HospitalGraphBuilder
    builder = HospitalGraphBuilder(sample_hospital)
    feature_seq = [
        {d.dept_id: [0.5] * 10 for d in sample_hospital.departments}
        for _ in range(24)
    ]
    return builder.build_graph_sequence(feature_seq)
