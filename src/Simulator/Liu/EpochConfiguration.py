"""Immutable configuration selected for one Liu decision epoch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Chain.ValidatorSet import ValidatorSet
from Liu.Action import require_block_interval_s, require_block_size_mb
from Liu.Protocol import LiuConsensusProtocol, require_liu_protocol
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_integer


@dataclass(frozen=True, slots=True)
class EpochConfiguration(CanonicalSerializable):
    epoch_id: int
    validator_set: ValidatorSet
    consensus_protocol: LiuConsensusProtocol
    block_size_mb: float
    block_interval_s: float
    activation_height: int | None = None
    previous_epoch_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.validator_set, ValidatorSet):
            raise ValueError("validator_set must be an immutable ValidatorSet")
        object.__setattr__(self, "epoch_id", require_integer(self.epoch_id, "epoch_id"))
        object.__setattr__(self, "consensus_protocol", require_liu_protocol(self.consensus_protocol))
        object.__setattr__(self, "block_size_mb", require_block_size_mb(self.block_size_mb))
        object.__setattr__(self, "block_interval_s", require_block_interval_s(self.block_interval_s))
        if self.activation_height is not None:
            object.__setattr__(
                self,
                "activation_height",
                require_integer(self.activation_height, "activation_height", minimum=1),
            )
        if self.previous_epoch_hash is not None:
            if not isinstance(self.previous_epoch_hash, str) or len(self.previous_epoch_hash) != 64:
                raise ValueError("previous_epoch_hash must be a SHA-256 hexadecimal string")
            try:
                int(self.previous_epoch_hash, 16)
            except ValueError as error:
                raise ValueError("previous_epoch_hash must be a SHA-256 hexadecimal string") from error

    @property
    def epoch_hash(self) -> str:
        return self.deterministic_hash()

    def to_dict(self) -> dict[str, Any]:
        data = {
            "block_interval_s": self.block_interval_s,
            "block_size_mb": self.block_size_mb,
            "consensus_protocol": self.consensus_protocol.value,
            "epoch_id": self.epoch_id,
            "validator_ids": list(self.validator_set.ids),
        }
        # Omitted lifecycle metadata preserves the hashes of the Phase-1 domain
        # values and all pre-dynamic runtime fixtures.
        if self.activation_height is not None:
            data["activation_height"] = self.activation_height
        if self.previous_epoch_hash is not None:
            data["previous_epoch_hash"] = self.previous_epoch_hash
        return data
