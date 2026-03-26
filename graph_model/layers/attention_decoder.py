"""
Multi-head temporal self-attention decoder.

Query  = most recent timestep's hidden state (what we want to predict from)
Keys   = all T timesteps (what historical moments are relevant)
Values = all T timesteps (what we extract from those moments)

This learns WHICH historical patterns matter for near-future congestion.
Example: a surge event 3 hours ago predicts overflow NOW.
"""
import torch
import torch.nn as nn


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,     # LSTM hidden_size
        num_heads: int = 8,
        num_horizons: int = 4,      # predict: 1h, 2h, 4h, 8h
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_horizons = num_horizons

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Per-node output projection: hidden → num_horizons congestion scores
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_horizons),
            nn.Sigmoid(),   # congestion score ∈ [0, 1]
        )

    def forward(
        self,
        lstm_out: torch.Tensor,   # [B, T, N, hidden_size]
    ) -> torch.Tensor:
        """
        Returns: predictions [B, N, num_horizons] (congestion scores per horizon)
        """
        B, T, N, H = lstm_out.shape

        # Process each node independently
        # Reshape: [B*N, T, H]
        x = lstm_out.permute(0, 2, 1, 3).contiguous().view(B * N, T, H)

        # Query: last timestep [B*N, 1, H]
        query = x[:, -1:, :]

        # Self-attention over all T timesteps
        context, attn_weights = self.attention(
            query=query,
            key=x,
            value=x,
        )
        context = self.dropout(context)
        context = self.norm(context + query)  # residual

        # context shape: [B*N, 1, H] → squeeze → [B*N, H]
        context = context.squeeze(1)

        # Project to predictions [B*N, num_horizons]
        predictions = self.output_proj(context)

        # Reshape to [B, N, num_horizons]
        predictions = predictions.view(B, N, self.num_horizons)

        return predictions
