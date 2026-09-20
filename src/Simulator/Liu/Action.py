"""Immutable Liu action A(t) = [a, delta, S^B, T^I]."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from Liu.Protocol import LiuConsensusProtocol, require_liu_protocol
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer


BLOCK_SIZE_STEP_MB = Decimal("0.2")
BLOCK_INTERVAL_STEP_S = Decimal("0.5")


def require_block_size_mb(value: object) -> float:
    numeric = require_finite_number(value, "block_size_mb", positive=True)
    if Decimal(str(numeric)) % BLOCK_SIZE_STEP_MB != 0:
        raise ValueError("block_size_mb must use the Liu 0.2 MB action step")
    return numeric


def require_block_interval_s(value: object) -> float:
    numeric = require_finite_number(value, "block_interval_s", positive=True)
    if Decimal(str(numeric)) % BLOCK_INTERVAL_STEP_S != 0:
        raise ValueError("block_interval_s must use the Liu 0.5 second action step")
    return numeric


@dataclass(frozen=True, slots=True)
class LiuAction(CanonicalSerializable):
    node_count: int
    validator_count: int
    validator_ids: tuple[int, ...]
    consensus_protocol: LiuConsensusProtocol
    block_size_mb: float
    block_interval_s: float

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        validator_count = require_integer(self.validator_count, "validator_count", minimum=1)
        if validator_count > node_count:
            raise ValueError("validator_count K cannot exceed node_count N")

        validator_ids = tuple(self.validator_ids)
        if len(validator_ids) != validator_count:
            raise ValueError("validator_ids must contain exactly K selected validators")
        if any(isinstance(node_id, bool) or not isinstance(node_id, int) for node_id in validator_ids):
            raise ValueError("validator IDs must be integers")
        if len(set(validator_ids)) != len(validator_ids):
            raise ValueError("validator IDs must be unique")
        if any(node_id < 0 or node_id >= node_count for node_id in validator_ids):
            raise ValueError("validator ID is outside the configured N nodes")

        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "validator_count", validator_count)
        object.__setattr__(self, "validator_ids", tuple(sorted(validator_ids)))
        object.__setattr__(self, "consensus_protocol", require_liu_protocol(self.consensus_protocol))
        object.__setattr__(self, "block_size_mb", require_block_size_mb(self.block_size_mb))
        object.__setattr__(self, "block_interval_s", require_block_interval_s(self.block_interval_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_interval_s": self.block_interval_s,
            "block_size_mb": self.block_size_mb,
            "consensus_protocol": self.consensus_protocol.value,
            "node_count": self.node_count,
            "validator_count": self.validator_count,
            "validator_ids": list(self.validator_ids),
        }
