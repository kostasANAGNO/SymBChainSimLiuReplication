"""Candidate-conditioned scalar Q-network.

Architecture: concat(state_enc, action_enc) → MLP → scalar Q(S, A)

This architecture is OUR_RECONSTRUCTION_OF_Q_FUNCTION_PARAMETERIZATION.
Liu specifies Q(S,A) conceptually (Eq.12) but does not provide a practical
neural architecture for the combinatorial action space.

Design choices:
  - Simple MLP: Linear → ReLU → Linear → ReLU → Linear(1) [scalar output]
  - Intentionally small for interpretability and CPU runtime
  - No batch norm, dropout, or attention — keep it minimal

Classification: OUR_RECONSTRUCTION_OF_Q_FUNCTION_PARAMETERIZATION
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DQNConfig, DEFAULT_CONFIG


class QNetwork(nn.Module):
    """Q(S, A) scalar network: input = concat(state_enc, action_enc)."""

    def __init__(self, state_dim: int, action_dim: int, cfg: DQNConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights(cfg)

    def _init_weights(self, cfg: DQNConfig) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim) or (state_dim,)
            action: (B, action_dim) or (action_dim,)
        Returns:
            (B, 1) or (1,) Q values
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def q_values_batch(self, state_enc: torch.Tensor, action_batch: torch.Tensor) -> torch.Tensor:
        """Score all candidate actions for a single state.

        Args:
            state_enc:    (state_dim,) — encoded single state
            action_batch: (A, action_dim) — all candidate action encodings
        Returns:
            (A,) Q values
        """
        B = action_batch.shape[0]
        state_expanded = state_enc.unsqueeze(0).expand(B, -1)  # (A, state_dim)
        return self.forward(state_expanded, action_batch).squeeze(1)  # (A,)
