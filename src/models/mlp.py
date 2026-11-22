from __future__ import annotations

import torch
from torch import nn


class FraudMLP(nn.Module):
    """Small tabular MLP that returns one logit per row."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

        dimensions = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for in_features, out_features in zip(dimensions[:-1], dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(dimensions[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        if logits.shape != (features.shape[0], 1):
            raise RuntimeError(
                f"Expected logits shape {(features.shape[0], 1)}, got {tuple(logits.shape)}."
            )
        return logits
