"""
Unit tests for hospital graph construction and edge weight calculation.
"""
import pytest
import torch
from simulation.hospital_topology import HOSPITALS, get_hospital_map
from graph_model.graph_construction.hospital_graph_builder import HospitalGraphBuilder
from graph_model.graph_construction.edge_weight_calculator import EdgeWeightCalculator


class TestHospitalGraphBuilder:
    def setup_method(self):
        self.hospital = HOSPITALS[0]
        self.builder  = HospitalGraphBuilder(self.hospital)

    def test_num_nodes_matches_departments(self):
        assert self.builder.num_nodes == len(self.hospital.departments)

    def test_edge_index_shape(self):
        ei, _ = self.builder._build_edge_index()
        assert ei.shape[0] == 2, "edge_index must have shape [2, num_edges]"
        assert ei.shape[1] > 0,  "Must have at least one edge"

    def test_edge_index_valid_node_refs(self):
        ei, _ = self.builder._build_edge_index()
        assert ei.min() >= 0
        assert ei.max() < self.builder.num_nodes

    def test_build_graph_sequence(self):
        T = 24
        feature_seq = []
        for t in range(T):
            dept_features = {
                d.dept_id: [0.5] * 10
                for d in self.hospital.departments
            }
            feature_seq.append(dept_features)

        seq = self.builder.build_graph_sequence(feature_seq)
        assert len(seq) == T

        for data in seq:
            assert data.x.shape == (self.builder.num_nodes, 10)
            assert data.edge_index.shape[0] == 2
            assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    def test_missing_dept_features_default_zero(self):
        """Departments with no feature data should default to zero vectors."""
        feature_seq = [{}]  # empty — no features for any dept
        seq = self.builder.build_graph_sequence(feature_seq)
        assert torch.all(seq[0].x == 0)


class TestEdgeWeightCalculator:
    def setup_method(self):
        self.hospital = HOSPITALS[0]
        self.calc     = EdgeWeightCalculator(self.hospital)
        self.builder  = HospitalGraphBuilder(self.hospital)
        self.edge_index, _ = self.builder._build_edge_index()

    def test_weights_shape(self):
        weights = self.calc.compute_weights(self.edge_index)
        assert weights.shape == (self.edge_index.shape[1], 1)

    def test_weights_non_negative(self):
        weights = self.calc.compute_weights(self.edge_index)
        assert weights.min() >= 0

    def test_dynamic_weights_differ_from_static(self):
        transfer_counts = {(0, 1): 10.0, (1, 2): 5.0}
        static  = self.calc.compute_weights(self.edge_index, transfer_counts=None)
        dynamic = self.calc.compute_weights(self.edge_index, transfer_counts=transfer_counts)
        assert not torch.allclose(static, dynamic), "Dynamic weights should differ from static"
