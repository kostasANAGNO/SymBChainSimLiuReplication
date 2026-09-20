"""Mutable per-node execution state for the LiuQuorum DES protocol."""

from dataclasses import dataclass, field
from enum import Enum


class LiuQuorumPhase(str, Enum):
    IDLE = "idle"
    OBSERVER = "observer"
    WAITING_REQUEST = "waiting_request"
    WAITING_REPLIES = "waiting_replies"
    PROCESSING_REQUEST = "processing_request"
    REPLICA_ACCEPTED = "replica_accepted"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass(slots=True)
class LiuQuorumState:
    phase: LiuQuorumPhase = LiuQuorumPhase.IDLE
    height: int = 1
    client_id: int = -1
    replica_ids: tuple[int, ...] = ()
    block: object | None = None
    block_identity: object | None = None
    request_sent_at: float | None = None
    replies: dict[int, str] = field(default_factory=dict)
    finalized: bool = False
