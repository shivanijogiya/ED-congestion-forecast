"""
LSTM temporal encoder.
Processes per-node GNN embeddings across T timesteps.

Design: weight-tied across nodes — same LSTM for all departments.
Rationale: reduces overfitting on low-volume departments; temporal
dynamics (shift changes, surge patterns) are hospital-agnostic.
"""
import torch
import torch.nn as nn


class LSTMTemporal(nn.Module):
    def __init__(
        self,
        input_size: int = 64,    # GNN out_channels
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # [batch, seq_len, features]
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,  # [batch, T, num_nodes, gnn_out]
    ) -> torch.Tensor:
        """
        Processes each node's time series independently with weight-sharing.

        Input:  [batch, T, num_nodes, gnn_out]
        Output: [batch, T, num_nodes, hidden_size]
        """
        B, T, N, F = x.shape

        # Reshape: treat each (batch, node) as a separate sequence
        # → [B*N, T, gnn_out]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, F)

        lstm_out, _ = self.lstm(x_reshaped)  # [B*N, T, hidden_size]
        lstm_out = self.norm(lstm_out)

        # Reshape back: [B, N, T, hidden_size] → [B, T, N, hidden_size]
        lstm_out = lstm_out.view(B, N, T, self.hidden_size)
        lstm_out = lstm_out.permute(0, 2, 1, 3)  # [B, T, N, hidden_size]

        return lstm_out
