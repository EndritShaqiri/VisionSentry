from __future__ import annotations

import torch
from torch import nn


class DroneRangeHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        ordinal_bins: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)
        self.ordinal_head = nn.Linear(hidden_dim, ordinal_bins)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        mean = self.mean_head(encoded)
        log_var = self.log_var_head(encoded).clamp(min=-6.0, max=6.0)
        ordinal_logits = self.ordinal_head(encoded)
        return mean, log_var, ordinal_logits


class TemporalRangeRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.delta_head = nn.Linear(output_dim, 1)
        self.log_var_head = nn.Linear(output_dim, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.rnn(sequence)
        delta = self.delta_head(encoded)
        log_var = self.log_var_head(encoded).clamp(min=-6.0, max=6.0)
        return delta, log_var
