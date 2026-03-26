"""
Shape and correctness tests for GNN, LSTM, Attention layers.
Uses mock tensors — no real data or infrastructure required.
"""
import pytest
import torch
from torch_geometric.data import Data, Batch


# ─── GNN Encoder ──────────────────────────────────────────────────────────────
class TestGNNEncoder:
    def setup_method(self):
        from graph_model.layers.gnn_encoder import GNNEncoder
        self.model = GNNEncoder(
            in_channels=10, hidden_channels=16, out_channels=32, num_heads=2
        )

    def _make_graph(self, num_nodes=7):
        x          = torch.randn(num_nodes, 10)
        edge_index = torch.randint(0, num_nodes, (2, 12))
        edge_attr  = torch.rand(12, 1)
        return x, edge_index, edge_attr

    def test_output_shape(self):
        x, ei, ea = self._make_graph(7)
        out = self.model(x, ei, ea)
        assert out.shape == (7, 32), f"Expected (7, 32), got {out.shape}"

    def test_output_range_reasonable(self):
        x, ei, ea = self._make_graph(7)
        out = self.model(x, ei, ea)
        assert not torch.isnan(out).any(), "NaN in GNN output"
        assert not torch.isinf(out).any(), "Inf in GNN output"

    def test_different_graph_sizes(self):
        for n in [4, 7, 10]:
            x, ei, ea = self._make_graph(n)
            out = self.model(x, ei, ea)
            assert out.shape == (n, 32)


# ─── LSTM Temporal ────────────────────────────────────────────────────────────
class TestLSTMTemporal:
    def setup_method(self):
        from graph_model.layers.lstm_temporal import LSTMTemporal
        self.model = LSTMTemporal(input_size=32, hidden_size=64, num_layers=2)

    def test_output_shape(self):
        B, T, N, F = 4, 24, 7, 32
        x   = torch.randn(B, T, N, F)
        out = self.model(x)
        assert out.shape == (B, T, N, 64), f"Expected (4,24,7,64), got {out.shape}"

    def test_temporal_causality(self):
        """Output at timestep t should not depend on t+1 inputs."""
        B, T, N, F = 2, 10, 5, 32
        x1 = torch.randn(B, T, N, F)
        x2 = x1.clone()
        x2[:, -1, :, :] = torch.zeros(B, N, F)  # zero out last timestep

        out1 = self.model(x1)
        out2 = self.model(x2)
        # First T-1 timestep outputs should be identical (unidirectional LSTM)
        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_no_nan(self):
        x   = torch.randn(2, 24, 7, 32)
        out = self.model(x)
        assert not torch.isnan(out).any()


# ─── Attention Decoder ────────────────────────────────────────────────────────
class TestAttentionDecoder:
    def setup_method(self):
        from graph_model.layers.attention_decoder import AttentionDecoder
        self.model = AttentionDecoder(hidden_size=64, num_heads=4, num_horizons=4)

    def test_output_shape(self):
        B, T, N, H = 4, 24, 7, 64
        x   = torch.randn(B, T, N, H)
        out = self.model(x)
        assert out.shape == (B, N, 4), f"Expected (4,7,4), got {out.shape}"

    def test_output_in_01(self):
        x   = torch.randn(2, 24, 7, 64)
        out = self.model(x)
        assert out.min() >= 0.0, "Predictions below 0"
        assert out.max() <= 1.0, "Predictions above 1"


# ─── Loss Function ────────────────────────────────────────────────────────────
class TestAsymmetricLoss:
    def setup_method(self):
        from graph_model.model.loss_functions import AsymmetricCongestionLoss
        self.loss_fn = AsymmetricCongestionLoss(threshold=0.8, underprediction_penalty=3.0)

    def test_underprediction_higher_loss(self):
        """Underpredicting high congestion should give higher loss than overprediction."""
        actual   = torch.full((2, 7, 4), 0.9)
        under    = torch.full((2, 7, 4), 0.6)   # underprediction
        over     = torch.full((2, 7, 4), 1.0)   # overprediction by same margin

        loss_under = self.loss_fn(under, actual)
        loss_over  = self.loss_fn(over,  actual)

        assert loss_under > loss_over, (
            f"Underprediction loss {loss_under:.4f} should exceed "
            f"overprediction loss {loss_over:.4f}"
        )

    def test_zero_loss_perfect_predictions(self):
        pred = torch.full((2, 7, 4), 0.5)
        loss = self.loss_fn(pred, pred)
        assert loss.item() < 1e-6
