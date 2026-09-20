"""Immutable metadata attached to Liu events outside simulated payload bytes."""

from dataclasses import dataclass

from Chain.ValidatorSet import ValidatorSet
from Liu.Serialization import CanonicalSerializable
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity


@dataclass(frozen=True, slots=True)
class LiuEventContext(CanonicalSerializable):
    protocol: str
    epoch_id: int
    epoch_hash: str
    validator_set_snapshot: ValidatorSet
    height: int
    view: int
    block_identity: LiuBlockIdentity
    logical_message_id: str

    def to_dict(self) -> dict:
        return {
            "block_identity": self.block_identity.to_dict(),
            "epoch_hash": self.epoch_hash,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "logical_message_id": self.logical_message_id,
            "protocol": self.protocol,
            "validator_ids": list(self.validator_set_snapshot.ids),
            "view": self.view,
        }

