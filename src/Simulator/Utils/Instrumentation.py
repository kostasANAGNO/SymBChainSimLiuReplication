"""Passive, append-only records for research instrumentation.

This module deliberately has no dependency on the simulation engine. Recording
an observation must not schedule events, consume randomness, or mutate model
objects.
"""

from dataclasses import asdict, dataclass
from typing import ClassVar

from Chain.NodeProfile import NodeProfile


@dataclass(frozen=True, slots=True)
class TransactionCreationRecord:
    transaction_id: int
    creator_node_id: int
    original_creation_time: float
    transaction_size: float


@dataclass(frozen=True, slots=True)
class BlockProposalRecord:
    block_id: int
    block_depth: int
    proposer: int
    consensus_protocol: str
    round: int
    configuration_depth: int
    proposal_time: float
    block_size: float
    transaction_count: int
    transaction_ids: tuple[int, ...]
    configured_block_size: float
    configured_block_time: float


@dataclass(frozen=True, slots=True)
class LocalConsensusDecisionRecord:
    block_id: int
    node_id: int
    decision_time: float
    consensus_protocol: str
    round: int
    configuration_depth: int
    decision_path: str
    quorum_size: int
    validator_count: int


@dataclass(frozen=True, slots=True)
class BlockObservationRecord:
    block_id: int
    block_depth: int
    node_id: int
    observation_time: float
    consensus_protocol: str
    round: int
    configuration_depth: int
    cause: str


@dataclass(frozen=True, slots=True)
class RunProfileContext:
    """Static run metadata exported separately from append-only raw records."""

    validator_ids: tuple[int, ...]
    node_profiles: tuple[NodeProfile, ...]
    computational_capability_reference_ghz: float
    capability_scaling_enabled: bool
    capability_scaling_formula: str
    capability_scaling_scope: tuple[str, ...]


class InstrumentationCollector:
    """Run-scoped passive instrumentation stored as immutable records."""

    transaction_creations: ClassVar[list[TransactionCreationRecord]] = []
    block_proposals: ClassVar[list[BlockProposalRecord]] = []
    local_consensus_decisions: ClassVar[list[LocalConsensusDecisionRecord]] = []
    block_observations: ClassVar[list[BlockObservationRecord]] = []
    run_profile_context: ClassVar[RunProfileContext | None] = None

    @classmethod
    def reset(cls) -> None:
        """Start a fresh collection scope before a simulation is configured."""
        cls.transaction_creations = []
        cls.block_proposals = []
        cls.local_consensus_decisions = []
        cls.block_observations = []
        cls.run_profile_context = None

    @classmethod
    def snapshot(cls) -> dict:
        """Capture an immutable-by-convention run scope for environment isolation."""
        return {
            "transaction_creations": tuple(cls.transaction_creations),
            "block_proposals": tuple(cls.block_proposals),
            "local_consensus_decisions": tuple(cls.local_consensus_decisions),
            "block_observations": tuple(cls.block_observations),
            "run_profile_context": cls.run_profile_context,
        }

    @classmethod
    def restore(cls, snapshot: dict) -> None:
        """Restore a previously captured run scope without mutating its tuples."""
        cls.transaction_creations = list(snapshot["transaction_creations"])
        cls.block_proposals = list(snapshot["block_proposals"])
        cls.local_consensus_decisions = list(snapshot["local_consensus_decisions"])
        cls.block_observations = list(snapshot["block_observations"])
        cls.run_profile_context = snapshot["run_profile_context"]

    @classmethod
    def configure_run_context(cls, context: RunProfileContext) -> None:
        cls.run_profile_context = context

    @classmethod
    def export_run_context(cls) -> dict:
        """Return a serialization-ready copy without altering raw records."""
        return {} if cls.run_profile_context is None else asdict(cls.run_profile_context)

    @classmethod
    def record_transaction_creation(cls, record: TransactionCreationRecord) -> None:
        cls.transaction_creations.append(record)

    @classmethod
    def record_block_proposal(cls, record: BlockProposalRecord) -> None:
        cls.block_proposals.append(record)

    @classmethod
    def record_local_consensus_decision(cls, record: LocalConsensusDecisionRecord) -> None:
        cls.local_consensus_decisions.append(record)

    @classmethod
    def record_block_observation(cls, record: BlockObservationRecord) -> None:
        cls.block_observations.append(record)
