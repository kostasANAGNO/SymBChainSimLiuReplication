"""Independent append-only instrumentation for Liu runtime protocols."""

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
class ProtocolFailureRecord:
    protocol: str
    epoch_id: int
    height: int
    view: int
    node_id: int
    phase: str
    reason: str
    event_time: float
    block_digest: str


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


@dataclass(frozen=True, slots=True)
class QuorumRequestTimeoutRecord:
    """The genuine pre-dispatch LiuQuorum request-timer decision for one height.

    Recorded when the client dispatches the request, from observable S_t/A_t only.
    ``observed_des_t_c_s`` is intentionally absent here; it is a post-execution DES
    outcome joined for diagnostics only, never fed back into the timer.
    """

    protocol: str
    policy_version: str
    epoch_id: int
    height: int
    client_id: int
    estimated_round_trip_s: float
    scheduled_timeout_s: float
    safety_factor: float
    floor_s: float
    ceiling_s: float
    event_time: float


@dataclass(frozen=True, slots=True)
class EpochLifecycleRecord:
    event_type: str
    event_time: float
    epoch_id: int
    activation_height: int
    epoch_hash: str
    previous_epoch_hash: str | None
    node_id: int | None = None
    validator_ids: tuple[int, ...] = ()
    consensus_protocol: str | None = None
    block_size_mb: float | None = None
    block_interval_s: float | None = None
    action_hash: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeLiuStateRecord:
    epoch_id: int
    state_hash: str
    chi_transaction_size_bytes: float
    chi_policy_version: str
    transaction_size_unit_policy: str
    chi_source: str
    chi_sample_count: int
    link_state_hash: str
    action_hash: str | None
    previous_state_hash: str | None
    observation_time: float


@dataclass(frozen=True, slots=True)
class RuntimeLiuEpochEvaluationRecord:
    epoch_id: int
    state_hash: str
    action_hash: str
    execution_measurement_hash: str
    constraint_result_hash: str
    reward_result_hash: str
    next_state_hash: str
    evaluation_time: float


class LiuRuntimeInstrumentationCollector:
    protocol_messages: ClassVar[list[ProtocolMessageRecord]] = []
    phase_transitions: ClassVar[list[ProtocolPhaseTransitionRecord]] = []
    certificates: ClassVar[list[ProtocolCertificateRecord]] = []
    protocol_finalities: ClassVar[list[ProtocolFinalityRecord]] = []
    view_changes: ClassVar[list[ProtocolViewChangeRecord]] = []
    protocol_failures: ClassVar[list[ProtocolFailureRecord]] = []
    timeout_adaptations: ClassVar[list[ProtocolTimeoutAdaptationRecord]] = []
    quorum_request_timeouts: ClassVar[list[QuorumRequestTimeoutRecord]] = []
    epoch_lifecycle: ClassVar[list[EpochLifecycleRecord]] = []
    runtime_states: ClassVar[list[RuntimeLiuStateRecord]] = []
    epoch_evaluations: ClassVar[list[RuntimeLiuEpochEvaluationRecord]] = []

    @classmethod
    def reset(cls) -> None:
        cls.protocol_messages = []
        cls.phase_transitions = []
        cls.certificates = []
        cls.protocol_finalities = []
        cls.view_changes = []
        cls.protocol_failures = []
        cls.timeout_adaptations = []
        cls.quorum_request_timeouts = []
        cls.epoch_lifecycle = []
        cls.runtime_states = []
        cls.epoch_evaluations = []

    @classmethod
    def snapshot(cls) -> dict[str, tuple]:
        return {
            "protocol_messages": tuple(cls.protocol_messages),
            "phase_transitions": tuple(cls.phase_transitions),
            "certificates": tuple(cls.certificates),
            "protocol_finalities": tuple(cls.protocol_finalities),
            "view_changes": tuple(cls.view_changes),
            "protocol_failures": tuple(cls.protocol_failures),
            "timeout_adaptations": tuple(cls.timeout_adaptations),
            "quorum_request_timeouts": tuple(cls.quorum_request_timeouts),
            "epoch_lifecycle": tuple(cls.epoch_lifecycle),
            "runtime_states": tuple(cls.runtime_states),
            "epoch_evaluations": tuple(cls.epoch_evaluations),
        }

    @classmethod
    def restore(cls, snapshot: dict[str, tuple]) -> None:
        for name in (
            "protocol_messages",
            "phase_transitions",
            "certificates",
            "protocol_finalities",
            "view_changes",
            "protocol_failures",
            "timeout_adaptations",
            "quorum_request_timeouts",
            "epoch_lifecycle",
            "runtime_states",
            "epoch_evaluations",
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
        if cls.protocol_failures:
            data["protocol_failures"] = [asdict(record) for record in cls.protocol_failures]
        if cls.timeout_adaptations:
            data["timeout_adaptations"] = [asdict(record) for record in cls.timeout_adaptations]
        if cls.epoch_lifecycle:
            data["epoch_lifecycle"] = [asdict(record) for record in cls.epoch_lifecycle]
        if cls.runtime_states:
            data["runtime_states"] = [asdict(record) for record in cls.runtime_states]
        if cls.epoch_evaluations:
            data["epoch_evaluations"] = [asdict(record) for record in cls.epoch_evaluations]
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
