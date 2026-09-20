"""Canonical collision-resistant identities for Liu runtime blocks/messages."""

from dataclasses import dataclass
import hashlib
import json

from Liu.Serialization import CanonicalSerializable


def _sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parent_digest(block) -> str:
    identity = block.extra_data.get("liu_identity")
    if identity is not None:
        return identity["block_digest"]
    return _sha256(
        {
            "depth": block.depth,
            "id": block.id,
            "previous": block.previous,
            "transaction_ids": [transaction.id for transaction in block.transactions],
        }
    )


@dataclass(frozen=True, slots=True)
class LiuBlockIdentity(CanonicalSerializable):
    epoch_id: int
    height: int
    parent_digest: str
    block_digest: str

    @classmethod
    def create(cls, epoch_id: int, height: int, parent: str, client_id: int, transactions: tuple) -> "LiuBlockIdentity":
        digest = _sha256(
            {
                "client_id": client_id,
                "epoch_id": epoch_id,
                "height": height,
                "parent_digest": parent,
                "transactions": [{"id": tx.id, "size_mb": tx.size} for tx in transactions],
            }
        )
        return cls(epoch_id, height, parent, digest)

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "parent_digest": self.parent_digest,
        }


def logical_message_id(
    identity: LiuBlockIdentity,
    message_type: str,
    sender_id: int,
    receiver_id: int,
    view: int = 0,
) -> str:
    return _sha256(
        {
            "block_identity": identity.to_dict(),
            "message_type": message_type,
            "receiver_id": receiver_id,
            "sender_id": sender_id,
            "view": view,
        }
    )
