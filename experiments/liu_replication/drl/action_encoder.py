"""Action encoder for Liu A = [a, delta, S_B, T_I].

Encoding:
  validator_mask (N binary)  — 1 if node i is in the validator set
  protocol_onehot (3)        — [PBFT=1 ZYZZYVA=0 QUORUM=0] etc.
  S_B_norm (1)               — block_size_mb / max_block_size_mb
  T_I_norm (1)               — block_interval_s / max_block_interval_s

Total: N + 3 + 1 + 1 = N + 5

Preserves validator structure (not just an index) so the network can learn
which validators are chosen, consistent with Liu's S_t including node attributes.

Classification: OUR_RECONSTRUCTION_OF_ACTION_ENCODING
The paper specifies the action components but not a tensor encoding.
"""
from __future__ import annotations

import torch

from config import DQNConfig, DEFAULT_CONFIG

_PROTOCOL_ORDER = ["PBFT", "ZYZZYVA", "LIU_QUORUM"]  # fixed index for one-hot


class ActionEncoder:
    """Encodes an action dict into a fixed-length float32 tensor.

    The action dict must contain:
      validator_mask    — list/tuple of N binary values
      protocol          — string: "PBFT" / "ZYZZYVA" / "LIU_QUORUM"
      block_size_mb     — float
      block_interval_s  — float
    """

    def __init__(self, n: int, cfg: DQNConfig = DEFAULT_CONFIG) -> None:
        self.n = n
        self.cfg = cfg
        self.dim = n + 3 + 1 + 1  # mask + onehot + S_B + T_I

    def encode(self, action: dict) -> torch.Tensor:
        """Encode one action dict into shape (dim,)."""
        mask = [float(v) for v in action["validator_mask"]]
        assert len(mask) == self.n

        proto_idx = _PROTOCOL_ORDER.index(action["protocol"])
        onehot = [0.0, 0.0, 0.0]
        onehot[proto_idx] = 1.0

        sb = float(action["block_size_mb"]) / self.cfg.max_block_size_mb
        ti = float(action["block_interval_s"]) / self.cfg.max_block_interval_s

        vec = mask + onehot + [sb, ti]
        assert len(vec) == self.dim
        return torch.tensor(vec, dtype=torch.float32)

    def encode_batch(self, actions: list[dict]) -> torch.Tensor:
        """Encode a list of action dicts into shape (B, dim)."""
        return torch.stack([self.encode(a) for a in actions])
