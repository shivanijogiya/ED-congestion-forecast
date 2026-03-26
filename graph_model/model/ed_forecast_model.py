"""
Top-level Graph-LSTM-Attention model for ED congestion forecasting.

Pipeline per forward pass:
  1. GNNEncoder: encode spatial context per timestep
     [B, T, N, F] → [B, T, N, gnn_out]

  2. LSTMTemporal: capture temporal patterns per node
     [B, T, N, gnn_out] → [B, T, N, hidden_size]

  3. AttentionDecoder: select relevant history, project to forecasts
     [B, T, N, hidden_size] → [B, N, num_horizons]

Architecture combines:
  - GATv2Conv   → asymmetric inter-department spatial attention
  - Weight-tied LSTM → temporal autocorrelation per department
  - Multi-head self-attention → selective historical context
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import List

from graph_model.layers.gnn_encoder import GNNEncoder
from graph_model.layers.lstm_temporal import LSTMTemporal
from graph_model.layers.attention_decoder import AttentionDecoder
from graph_model.model.model_config import ModelConfig


class EDForecastModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.gnn = GNNEncoder(
            in_channels=config.feature_dim,
            hidden_channels=config.gnn_hidden,
            out_channels=config.gnn_out,
            num_heads=config.gnn_heads,
            dropout=config.dropout,
            edge_dim=1,
        )
        self.lstm = LSTMTemporal(
            input_size=config.gnn_out,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
        )
        self.decoder = AttentionDecoder(
            hidden_size=config.lstm_hidden,
            num_heads=config.attn_heads,
            num_horizons=config.num_horizons,
            dropout=config.dropout,
        )

    def forward(
        self,
        graph_sequence: List[List[Data]],   # [B] × [T] Data objects
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            graph_sequence: batch of temporal graph sequences.
                           graph_sequence[b][t] is a PyG Data for batch b, timestep t.
            num_nodes: number of department nodes (N)
        Returns:
            predictions: [B, N, num_horizons] — congestion scores ∈ [0, 1]
        """
        B = len(graph_sequence)
        T = len(graph_sequence[0])

        # ── Step 1: GNN encoding per timestep ─────────────────────────────────
        # gnn_embeddings: [B, T, N, gnn_out]
        gnn_embeddings = torch.zeros(
            B, T, num_nodes, self.config.gnn_out,
            device=next(self.parameters()).device
        )

        for t in range(T):
            # Batch all graphs at timestep t
            batch = Batch.from_data_list([graph_sequence[b][t] for b in range(B)])
            h = self.gnn(batch.x, batch.edge_index, batch.edge_attr)

            # h shape: [B*N, gnn_out] → reshape to [B, N, gnn_out]
            h = h.view(B, num_nodes, self.config.gnn_out)
            gnn_embeddings[:, t, :, :] = h

        # ── Step 2: LSTM temporal encoding ────────────────────────────────────
        # [B, T, N, gnn_out] → [B, T, N, lstm_hidden]
        lstm_out = self.lstm(gnn_embeddings)

        # ── Step 3: Attention decoding ─────────────────────────────────────────
        # [B, T, N, lstm_hidden] → [B, N, num_horizons]
        predictions = self.decoder(lstm_out)

        return predictions

    @classmethod
    def from_config(cls, config: ModelConfig) -> "EDForecastModel":
        return cls(config)
