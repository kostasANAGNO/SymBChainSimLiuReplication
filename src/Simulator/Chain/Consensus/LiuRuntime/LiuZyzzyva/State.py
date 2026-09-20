"""Per-node state for one LiuZyzzyva fast/recovery height."""

from dataclasses import dataclass, field
from enum import Enum


class LiuZyzzyvaPhase(str, Enum):
    IDLE = "idle"
    OBSERVER = "observer"
    CLIENT_READY = "client_ready"
    CLIENT_WAITING_REPLIES = "client_waiting_replies"
    PRIMARY_WAITING_REQUEST = "primary_waiting_request"
    PRIMARY_PROCESSING_ORDER = "primary_processing_order"
    ORDERED = "ordered"
    BACKUP_WAITING_ORDER = "backup_waiting_order"
    PROCESSING_SPECULATIVE = "processing_speculative"
    SPECULATIVELY_EXECUTED = "speculatively_executed"
    FAST_FINALIZED = "fast_finalized"
    PENDING_RECOVERY = "pending_recovery"
    RECOVERY_CERTIFICATE_CREATED = "recovery_certificate_created"
    RECOVERY_PROCESSING = "recovery_processing"
    LOCAL_COMMITTED = "local_committed"
    RECOVERY_FINALIZED = "recovery_finalized"
    VIEW_CHANGE_WAITING = "view_change_waiting"
    NEW_VIEW_ESTABLISHED = "new_view_established"
    FAILED = "failed"


@dataclass(slots=True)
class LiuZyzzyvaState:
    phase: LiuZyzzyvaPhase = LiuZyzzyvaPhase.IDLE
    height: int = 1
    current_view: int = 0
    client_id: int = -1
    primary_id: int = -1
    replica_ids: tuple[int, ...] = ()
    backup_ids: tuple[int, ...] = ()
    block: object | None = None
    block_identity: object | None = None
    request_sent_at: float | None = None
    accepted_order: str | None = None
    speculative_execution_state: str | None = None
    speculative_replies: dict[str, dict[int, str]] = field(default_factory=dict)
    seen_speculative_reply_signers: set[int] = field(default_factory=set)
    conflicting_speculative_replies: dict[str, dict[int, str]] = field(default_factory=dict)
    sent_speculative_reply: bool = False
    fast_finalized_height: int | None = None
    recovery_finalized_height: int | None = None
    failure_reason: str | None = None
    recovery_commit_certificate: object | None = None
    recovery_local_commits: dict[str, dict[int, str]] = field(default_factory=dict)
    local_commit_digest: str | None = None
    local_commit_certificate: object | None = None
    local_committed_height: int | None = None
    sent_local_commit: bool = False
    highest_speculative_evidence: object | None = None
    locked_digest: str | None = None
    locked_block: object | None = None
    strongest_safety_evidence: object | None = None
    view_change_evidence: dict[int, dict[int, object]] = field(default_factory=dict)
    view_change_blocks: dict[int, dict[str, object]] = field(default_factory=dict)
    new_view_certificate: object | None = None
    pending_target_view: int | None = None
    safety_failure: str | None = None
