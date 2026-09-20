"""Per-node safety state for one LiuPBFT epoch/height/view."""

from dataclasses import dataclass, field
from enum import Enum


class LiuPBFTPhase(str, Enum):
    IDLE = "idle"
    OBSERVER = "observer"
    CLIENT_WAITING = "client_waiting"
    WAITING_PREPREPARE = "waiting_preprepare"
    PROCESSING_REQUEST = "processing_request"
    PROCESSING_PREPREPARE = "processing_preprepare"
    PREPARE_COLLECTING = "prepare_collecting"
    PREPARE_PROCESSING = "prepare_processing"
    PREPARED = "prepared"
    COMMIT_COLLECTING = "commit_collecting"
    COMMIT_PROCESSING = "commit_processing"
    LOCAL_COMMITTED = "local_committed"
    REPLIED = "replied"
    PROTOCOL_FINALIZED = "protocol_finalized"
    FAILED = "failed"
    VIEW_CHANGE_WAITING = "view_change_waiting"
    NEW_VIEW_PRIMARY = "new_view_primary"


@dataclass(slots=True)
class LiuPBFTState:
    phase: LiuPBFTPhase = LiuPBFTPhase.IDLE
    height: int = 1
    current_view: int = 0
    height_start_view: int = 0
    highest_observed_view: int = 0
    timeout_backoff_count: int = 0
    maximum_effective_timeout_s: float = 0.0
    client_id: int = -1
    primary_id: int = -1
    replica_ids: tuple[int, ...] = ()
    backup_ids: tuple[int, ...] = ()
    block: object | None = None
    block_identity: object | None = None
    request_sent_at: float | None = None
    accepted_preprepare: str | None = None
    prepare_votes: dict[str, dict[int, str]] = field(default_factory=dict)
    commit_votes: dict[str, dict[int, str]] = field(default_factory=dict)
    prepared_certificate: object | None = None
    local_commit_certificate: object | None = None
    client_reply_votes: dict[str, dict[int, object]] = field(default_factory=dict)
    sent_prepare: bool = False
    sent_commit: bool = False
    committed_height: int | None = None
    prepare_processing_started: bool = False
    commit_processing_started: bool = False
    highest_prepared_certificate: object | None = None
    locked_digest: str | None = None
    locked_block: object | None = None
    view_change_evidence: dict[int, dict[int, object]] = field(default_factory=dict)
    view_change_blocks: dict[int, dict[str, object]] = field(default_factory=dict)
    new_view_certificate: object | None = None
    pending_target_view: int | None = None
    safety_failure: str | None = None
    catchup_responses: set[tuple[str, int]] = field(default_factory=set)
    proposal_retransmissions: set[tuple[int, int, int, str, int]] = field(default_factory=set)
