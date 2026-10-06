"""Protocol trace of the Liu runtime protocols.

`protocol_finalities` is functional: it is the DES finality measurement (loop termination, T_C and the
epoch-boundary check). The other lists are the append-only protocol trace the protocol tests assert on.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ProtocolMessageRecord:
    logical_message_id: str
    protocol: str
    message_type: str
    epoch_id: int
    height: int
    block_digest: str
    parent_digest: str
    sender_id: int
    receiver_id: int
    sent_at: float
    arrival_at: float
    rate_mbps: float
    payload_size_mb: float
    serialization_delay_s: float
    propagation_delay_s: float


@dataclass(frozen=True, slots=True)
class ProtocolPhaseTransitionRecord:
    protocol: str
    epoch_id: int
    height: int
    block_digest: str
    node_id: int
    previous_phase: str
    new_phase: str
    transition_time: float
    trigger_message_id: str
    processing_started_at: float | None = None
    processing_completed_at: float | None = None
    processing_work_name: str | None = None
    signature_operations: int | None = None
    mac_operations: int | None = None


@dataclass(frozen=True, slots=True)
class ProtocolCertificateRecord:
    protocol: str
    epoch_id: int
    height: int
    block_digest: str
    certificate_hash: str
    certificate_type: str
    signer_ids: tuple[int, ...]
    threshold: int
    creation_time: float


@dataclass(frozen=True, slots=True)
class ProtocolFinalityRecord:
    protocol: str
    epoch_id: int
    height: int
    block_digest: str
    parent_digest: str
    client_id: int
    request_sent_at: float
    finality_time: float
    finality_path: str
    certificate_hash: str

    @property
    def consensus_latency_s(self) -> float:
        return self.finality_time - self.request_sent_at


@dataclass(frozen=True, slots=True)
class ProtocolViewChangeRecord:
    protocol: str
    epoch_id: int
    height: int
    node_id: int
    from_view: int
    target_view: int
    event_type: str
    status: str
    event_time: float
    signer_count: int = 0
    carried_prepared_digest: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class ProtocolTimeoutAdaptationRecord:
    protocol: str
    policy_version: str
    epoch_id: int
    height: int
    node_id: int
    view: int
    phase: str
    event_type: str
    event_time: float
    consecutive_timeout_view_changes: int
    effective_timeout_s: float
    initial_timeout_s: float
    maximum_timeout_s: float
    backoff_factor: float


class LiuRuntimeInstrumentationCollector:
    protocol_messages: ClassVar[list[ProtocolMessageRecord]] = []
    phase_transitions: ClassVar[list[ProtocolPhaseTransitionRecord]] = []
    certificates: ClassVar[list[ProtocolCertificateRecord]] = []
    protocol_finalities: ClassVar[list[ProtocolFinalityRecord]] = []
    view_changes: ClassVar[list[ProtocolViewChangeRecord]] = []
    timeout_adaptations: ClassVar[list[ProtocolTimeoutAdaptationRecord]] = []

    @classmethod
    def reset(cls) -> None:
        cls.protocol_messages = []
        cls.phase_transitions = []
        cls.certificates = []
        cls.protocol_finalities = []
        cls.view_changes = []
        cls.timeout_adaptations = []

    @classmethod
    def snapshot(cls) -> dict[str, tuple]:
        return {
            "protocol_messages": tuple(cls.protocol_messages),
            "phase_transitions": tuple(cls.phase_transitions),
            "certificates": tuple(cls.certificates),
            "protocol_finalities": tuple(cls.protocol_finalities),
            "view_changes": tuple(cls.view_changes),
            "timeout_adaptations": tuple(cls.timeout_adaptations),
        }

    @classmethod
    def restore(cls, snapshot: dict[str, tuple]) -> None:
        for name in (
            "protocol_messages",
            "phase_transitions",
            "certificates",
            "protocol_finalities",
            "view_changes",
            "timeout_adaptations",
        ):
            setattr(cls, name, list(snapshot[name]))

    @classmethod
    def deterministic_hash(cls) -> str:
        data = {
            "certificates": [asdict(record) for record in cls.certificates],
            "phase_transitions": [asdict(record) for record in cls.phase_transitions],
            "protocol_finalities": [asdict(record) for record in cls.protocol_finalities],
            "protocol_messages": [asdict(record) for record in cls.protocol_messages],
        }
        # Preserve pre-extension normal-path hashes when optional recovery/view
        # change streams are empty, while binding those records when present.
        if cls.view_changes:
            data["view_changes"] = [asdict(record) for record in cls.view_changes]
        if cls.timeout_adaptations:
            data["timeout_adaptations"] = [asdict(record) for record in cls.timeout_adaptations]
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
