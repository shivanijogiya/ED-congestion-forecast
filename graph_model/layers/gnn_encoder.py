"""
GATv2Conv-based graph encoder.
Encodes spatial relationships between ED departments into node embeddings.

GATv2 chosen over GATv1 for dynamic attention:
  - GATv1 computes e_ij = a^T [Wh_i || Wh_j] (static for fixed inputs)
  - GATv2 computes e_ij = a^T LeakyReLU(W[h_i || h_j]) (truly dynamic)
  This captures asymmetric patient flow (Triage→Resus ≠ Resus→Triage).
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        out_channels: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        edge_dim: int = 1,
    ):
        super().__init__()

        # Layer 1: in_channels → hidden_channels * num_heads
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=True,
        )

        # Layer 2: hidden_channels * num_heads → out_channels (single head)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=False,
        )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_channels * num_heads)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: torch.Tensor,           # [num_nodes, in_channels]
        edge_index: torch.Tensor,  # [2, num_edges]
        edge_attr: torch.Tensor,   # [num_edges, edge_dim]
    ) -> torch.Tensor:
        """
        Returns: node embeddings [num_nodes, out_channels]
        """
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.norm1(h)
        h = self.activation(h)
        h = self.dropout(h)

        # Layer 2
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(h)
        h = self.activation(h)

        return h
