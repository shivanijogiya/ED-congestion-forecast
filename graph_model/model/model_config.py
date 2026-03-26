"""Model hyperparameter configuration."""
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    # Input
    feature_dim:   int   = 10     # Number of input features per node

    # GNN
    gnn_hidden:    int   = 32     # Hidden channels per head in GATv2 layer 1
    gnn_out:       int   = 64     # Output embedding dim (GATv2 layer 2)
    gnn_heads:     int   = 4      # Attention heads in GATv2 layer 1

    # LSTM
    lstm_hidden:   int   = 128    # LSTM hidden state size
    lstm_layers:   int   = 2      # Number of LSTM layers

    # Attention
    attn_heads:    int   = 8      # Multi-head attention heads in decoder

    # Output
    num_horizons:  int   = 4      # Forecast horizons: 1h, 2h, 4h, 8h
    horizon_hours: list  = field(default_factory=lambda: [1, 2, 4, 8])

    # Regularization
    dropout:       float = 0.2
    weight_decay:  float = 1e-4

    # Training
    learning_rate: float = 1e-3
    batch_size:    int   = 32
    max_epochs:    int   = 100
    patience:      int   = 10     # Early stopping patience
    grad_clip:     float = 1.0    # Gradient clipping norm

    # Temporal window
    sequence_len:  int   = 24     # T: number of hourly timesteps in input

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
