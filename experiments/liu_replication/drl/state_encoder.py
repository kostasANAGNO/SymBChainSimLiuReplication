"""State encoder for Liu S_t = [chi, Upsilon, x, c, R].

Preserves Liu's state definition (Eq.10):
  S_t = [chi, Upsilon, x, c, R(t)]

where:
  chi        — transaction size (1 scalar)
  Upsilon    — stake vector (N scalars)
  x          — position vector (N × 2 scalars)
  c          — capability vector (N scalars)
  R          — full directed link-rate matrix (N*(N-1) scalars, row-major, skip diagonal)

Normalization is an IMPLEMENTATION_DETAIL, not a semantic modification.
All normalization constants are centralized here.

Classification: OUR_RECONSTRUCTION (exact state composition) / PAPER_EXACT (state variables per Eq.10)
"""
from __future__ import annotations

import numpy as np
import torch

from config import DQNConfig, DEFAULT_CONFIG


class StateEncoder:
    """Encodes a state dict into a fixed-length float32 tensor.

    The state dict must contain:
      chi_bytes          — transaction size in bytes (scalar)
      stakes             — tuple/list of N stake values
      positions_km       — tuple/list of N (x,y) positions
      capabilities_ghz   — tuple/list of N capability values
      link_rows_mbps     — tuple of N tuples (row i: link rates from i to all j≠i)

    Ordering in the output tensor (row-major, deterministic):
      [chi_norm | stakes_norm(N) | pos_x_norm(N) | pos_y_norm(N) | caps_norm(N) | links_norm(N*(N-1))]

    dim = 1 + N + 2*N + N + N*(N-1) = 1 + 4*N + N*(N-1) = 1 + N*(N+3)
    """

    def __init__(self, n: int, cfg: DQNConfig = DEFAULT_CONFIG) -> None:
        self.n = n
        self.cfg = cfg
        self.dim = 1 + n + 2 * n + n + n * (n - 1)

    def encode(self, state: dict) -> torch.Tensor:
        """Encode a state dict into a float32 tensor of shape (dim,)."""
        n = self.n
        cfg = self.cfg

        chi = float(state["chi_bytes"]) / cfg.max_tx_size_bytes

        stakes = [float(s) / cfg.max_stake for s in state["stakes"]]
        assert len(stakes) == n

        positions = state["positions_km"]
        pos_x = [float(p[0]) for p in positions]
        pos_y = [float(p[1]) for p in positions]
        assert len(pos_x) == n

        caps = [float(c) / cfg.max_capability_ghz for c in state["capabilities_ghz"]]
        assert len(caps) == n

        # Link rates: row-major, skip diagonal (i==j)
        link_rows = state["link_rows_mbps"]
        links = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rate = link_rows[i][j]
                    links.append(float(rate) / cfg.max_link_rate_mbps if rate is not None else 0.0)
        assert len(links) == n * (n - 1)

        vec = [chi] + stakes + pos_x + pos_y + caps + links
        assert len(vec) == self.dim
        return torch.tensor(vec, dtype=torch.float32)

    def encode_batch(self, states: list[dict]) -> torch.Tensor:
        """Encode a list of state dicts into shape (B, dim)."""
        return torch.stack([self.encode(s) for s in states])

    @staticmethod
    def from_env_observation(obs: dict, n: int,
                              stakes: tuple, positions_km: tuple,
                              capabilities_ghz: tuple,
                              link_rows_mbps: tuple,
                              chi_bytes: float) -> dict:
        """Build a state dict from the env observation + fixed population.

        The dynamic component is link_rows_mbps (current FSMC state).
        The static components are stakes, positions, capabilities, chi.
        """
        return {
            "chi_bytes": chi_bytes,
            "stakes": stakes,
            "positions_km": positions_km,
            "capabilities_ghz": capabilities_ghz,
            "link_rows_mbps": link_rows_mbps,
        }
