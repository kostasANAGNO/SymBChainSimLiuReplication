"""Pure aggregation of passive instrumentation into research metrics.

The analysis layer neither schedules events nor mutates the collector or its
records.  ``LogicalBlockKey`` is sufficient for the simulator's current
single-successful-proposal-per-round execution model.  It is deliberately
documented as *not fork-proof*: an adversarial/forking model would need a
collision-resistant block identity (for example a block hash or parent-linked
proposal digest).
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median, pstdev

from Utils.Instrumentation import (
    BlockObservationRecord,
    BlockProposalRecord,
    InstrumentationCollector,
    LocalConsensusDecisionRecord,
    TransactionCreationRecord,
)


@dataclass(frozen=True, slots=True, order=True)
class LogicalBlockKey:
    configuration_depth: int
    consensus_protocol: str
    round: int
    block_id: int


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    standard_deviation: float

    @classmethod
    def from_values(cls, values: list[float]) -> "DistributionSummary":
        if not values:
            raise ValueError("A distribution requires at least one value")
        return cls(len(values), mean(values), median(values), min(values), max(values), pstdev(values))


@dataclass(frozen=True, slots=True)
class FirstQuorumEvidence:
    logical_block_key: LogicalBlockKey
    decision_time: float
    node_id: int
    decision_path: str
    quorum_size: int
    validator_count: int


@dataclass(frozen=True, slots=True)
class FinalizedBlockMetric:
    logical_block_key: LogicalBlockKey
    block_depth: int
    proposer: int
    proposal_time: float
    first_quorum_evidence: FirstQuorumEvidence
    quorum_adoption_time: float | None
    consensus_latency: float
    transaction_ids: tuple[int, ...]
    actual_block_size: float
    configured_block_size: float
    configured_block_time: float

    @property
    def first_consensus_decision_time(self) -> float:
        return self.first_quorum_evidence.decision_time


@dataclass(frozen=True, slots=True)
class TransactionQuorumTTF:
    transaction_id: int
    logical_block_key: LogicalBlockKey
    original_creation_time: float
    first_consensus_decision_time: float
    quorum_ttf: float


@dataclass(frozen=True, slots=True)
class ObservedInterBlockTime:
    previous_block: LogicalBlockKey
    current_block: LogicalBlockKey
    observed_inter_block_time: float
    configured_block_time: float


@dataclass(frozen=True, slots=True)
class SimulationThroughput:
    measurement_start: float
    measurement_end: float
    measurement_duration: float
    unique_transaction_count: int
    transactions_per_second: float


@dataclass(frozen=True, slots=True)
class AnalyticalCapacityEstimate:
    empirical_mean_transaction_size: float
    per_protocol: dict[str, DistributionSummary]


@dataclass(frozen=True, slots=True)
class ObservationLatency:
    logical_block_key: LogicalBlockKey
    node_id: int
    cause: str
    observation_time: float
    latency_from_first_consensus_decision: float


@dataclass(frozen=True, slots=True)
class InstrumentationMetricsResult:
    finalized_blocks: tuple[FinalizedBlockMetric, ...]
    consensus_latency_by_protocol: dict[str, DistributionSummary]
    transaction_quorum_ttf: tuple[TransactionQuorumTTF, ...]
    transaction_quorum_ttf_by_protocol: dict[str, DistributionSummary]
    observed_inter_block_times: tuple[ObservedInterBlockTime, ...]
    observed_inter_block_time_by_protocol: dict[str, DistributionSummary]
    simulation_throughput: SimulationThroughput | None
    analytical_capacity_estimate: AnalyticalCapacityEstimate | None
    observation_latencies: tuple[ObservationLatency, ...]
    duplicate_transaction_ids: tuple[int, ...]


class InstrumentationMetrics:
    """Build derived metrics from immutable raw instrumentation records."""

    @staticmethod
    def logical_block_key(record: BlockProposalRecord | LocalConsensusDecisionRecord | BlockObservationRecord) -> LogicalBlockKey:
        return LogicalBlockKey(record.configuration_depth, record.consensus_protocol, record.round, record.block_id)

    @classmethod
    def from_collector(
        cls,
        collector: type[InstrumentationCollector] = InstrumentationCollector,
        measurement_start: float | None = None,
        measurement_end: float | None = None,
    ) -> InstrumentationMetricsResult:
        return cls.calculate(
            collector.transaction_creations,
            collector.block_proposals,
            collector.local_consensus_decisions,
            collector.block_observations,
            measurement_start,
            measurement_end,
        )

    @classmethod
    def calculate(
        cls,
        transaction_creations: list[TransactionCreationRecord],
        block_proposals: list[BlockProposalRecord],
        local_consensus_decisions: list[LocalConsensusDecisionRecord],
        block_observations: list[BlockObservationRecord],
        measurement_start: float | None = None,
        measurement_end: float | None = None,
    ) -> InstrumentationMetricsResult:
        creations = cls._unique_by_id(transaction_creations, "transaction")
        proposals = cls._unique_by_key(block_proposals)
        decisions = cls._decisions_by_key_and_node(local_consensus_decisions)

        finalized: list[FinalizedBlockMetric] = []
        for key, node_decisions in decisions.items():
            if key not in proposals:
                raise ValueError(f"Consensus decision without proposal for {key}")
            proposal = proposals[key]
            if proposal.transaction_count != len(proposal.transaction_ids):
                raise ValueError(f"Transaction count does not match membership for {key}")
            if len(set(proposal.transaction_ids)) != len(proposal.transaction_ids):
                raise ValueError(f"Duplicate transaction inside proposal {key}")

            ordered_decisions = sorted(node_decisions.values(), key=lambda record: (record.decision_time, record.node_id))
            first = ordered_decisions[0]
            validator_counts = {record.validator_count for record in ordered_decisions}
            if len(validator_counts) != 1:
                raise ValueError(f"Inconsistent validator count for {key}")
            # BigFoot validators may decide the same block through different
            # valid paths (fast q=K-1 or slow q=2f+1). Adoption is therefore
            # defined relative to the quorum carried by the first evidence.
            quorum_size = first.quorum_size
            if quorum_size <= 0:
                raise ValueError(f"Invalid quorum size for {key}")
            adoption_time = ordered_decisions[quorum_size - 1].decision_time - first.decision_time if len(ordered_decisions) >= quorum_size else None
            evidence = FirstQuorumEvidence(key, first.decision_time, first.node_id, first.decision_path, first.quorum_size, first.validator_count)
            finalized.append(
                FinalizedBlockMetric(
                    key,
                    proposal.block_depth,
                    proposal.proposer,
                    proposal.proposal_time,
                    evidence,
                    adoption_time,
                    first.decision_time - proposal.proposal_time,
                    proposal.transaction_ids,
                    proposal.block_size,
                    proposal.configured_block_size,
                    proposal.configured_block_time,
                )
            )

        finalized.sort(key=lambda block: (block.block_depth, block.first_consensus_decision_time, block.logical_block_key))
        cls._validate_canonical_depths(finalized)

        consensus_values: dict[str, list[float]] = defaultdict(list)
        for block in finalized:
            consensus_values[block.logical_block_key.consensus_protocol].append(block.consensus_latency)

        transaction_ttf, duplicate_ids = cls._transaction_ttf(finalized, creations)
        ttf_values: dict[str, list[float]] = defaultdict(list)
        for item in transaction_ttf:
            ttf_values[item.logical_block_key.consensus_protocol].append(item.quorum_ttf)

        inter_block = cls._inter_block_times(finalized)
        inter_block_values: dict[str, list[float]] = defaultdict(list)
        for item in inter_block:
            protocol = item.current_block.consensus_protocol
            inter_block_values[protocol].append(item.observed_inter_block_time)

        throughput, window_ids = cls._throughput(finalized, measurement_start, measurement_end)
        analytical = cls._analytical_capacity(finalized, creations, window_ids)
        observation_latencies = cls._observation_latencies(block_observations, finalized)

        return InstrumentationMetricsResult(
            tuple(finalized),
            cls._summaries(consensus_values),
            tuple(transaction_ttf),
            cls._summaries(ttf_values),
            tuple(inter_block),
            cls._summaries(inter_block_values),
            throughput,
            analytical,
            tuple(observation_latencies),
            tuple(sorted(duplicate_ids)),
        )

    @staticmethod
    def _unique_by_id(records: list[TransactionCreationRecord], label: str) -> dict[int, TransactionCreationRecord]:
        result: dict[int, TransactionCreationRecord] = {}
        for record in records:
            previous = result.get(record.transaction_id)
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting {label} records for id {record.transaction_id}")
            result[record.transaction_id] = record
        return result

    @classmethod
    def _unique_by_key(cls, records: list[BlockProposalRecord]) -> dict[LogicalBlockKey, BlockProposalRecord]:
        result: dict[LogicalBlockKey, BlockProposalRecord] = {}
        for record in records:
            key = cls.logical_block_key(record)
            previous = result.get(key)
            if previous is not None and previous != record:
                raise ValueError(f"Logical block key collision for {key}; the current key is not fork-proof")
            result[key] = record
        return result

    @classmethod
    def _decisions_by_key_and_node(
        cls, records: list[LocalConsensusDecisionRecord]
    ) -> dict[LogicalBlockKey, dict[int, LocalConsensusDecisionRecord]]:
        result: dict[LogicalBlockKey, dict[int, LocalConsensusDecisionRecord]] = defaultdict(dict)
        for record in records:
            key = cls.logical_block_key(record)
            previous = result[key].get(record.node_id)
            if previous is not None and previous != record:
                raise ValueError(f"Conflicting decisions by node {record.node_id} for {key}")
            result[key][record.node_id] = record
        return result

    @staticmethod
    def _validate_canonical_depths(finalized: list[FinalizedBlockMetric]) -> None:
        by_depth: dict[int, LogicalBlockKey] = {}
        for block in finalized:
            previous = by_depth.get(block.block_depth)
            if previous is not None and previous != block.logical_block_key:
                raise ValueError(f"Multiple finalized logical blocks at depth {block.block_depth}; canonical-chain selection is undefined")
            by_depth[block.block_depth] = block.logical_block_key

    @staticmethod
    def _transaction_ttf(
        finalized: list[FinalizedBlockMetric], creations: dict[int, TransactionCreationRecord]
    ) -> tuple[list[TransactionQuorumTTF], set[int]]:
        memberships: dict[int, list[FinalizedBlockMetric]] = defaultdict(list)
        for block in finalized:
            for transaction_id in block.transaction_ids:
                memberships[transaction_id].append(block)

        result: list[TransactionQuorumTTF] = []
        duplicates: set[int] = set()
        for transaction_id, blocks in memberships.items():
            if transaction_id not in creations:
                raise ValueError(f"Missing creation record for finalized transaction {transaction_id}")
            if len(blocks) > 1:
                duplicates.add(transaction_id)
            block = min(blocks, key=lambda item: (item.first_consensus_decision_time, item.logical_block_key))
            creation = creations[transaction_id]
            result.append(
                TransactionQuorumTTF(
                    transaction_id,
                    block.logical_block_key,
                    creation.original_creation_time,
                    block.first_consensus_decision_time,
                    block.first_consensus_decision_time - creation.original_creation_time,
                )
            )
        result.sort(key=lambda item: item.transaction_id)
        return result, duplicates

    @staticmethod
    def _inter_block_times(finalized: list[FinalizedBlockMetric]) -> list[ObservedInterBlockTime]:
        result: list[ObservedInterBlockTime] = []
        for previous, current in zip(finalized, finalized[1:]):
            result.append(
                ObservedInterBlockTime(
                    previous.logical_block_key,
                    current.logical_block_key,
                    current.first_consensus_decision_time - previous.first_consensus_decision_time,
                    current.configured_block_time,
                )
            )
        return result

    @staticmethod
    def _throughput(
        finalized: list[FinalizedBlockMetric], measurement_start: float | None, measurement_end: float | None
    ) -> tuple[SimulationThroughput | None, set[int]]:
        if not finalized:
            return None, set()
        start = finalized[0].first_consensus_decision_time if measurement_start is None else measurement_start
        end = finalized[-1].first_consensus_decision_time if measurement_end is None else measurement_end
        if end <= start:
            raise ValueError("Measurement end must be later than measurement start")

        transaction_ids = {
            transaction_id
            for block in finalized
            if start < block.first_consensus_decision_time <= end
            for transaction_id in block.transaction_ids
        }
        duration = end - start
        return SimulationThroughput(start, end, duration, len(transaction_ids), len(transaction_ids) / duration), transaction_ids

    @staticmethod
    def _analytical_capacity(
        finalized: list[FinalizedBlockMetric], creations: dict[int, TransactionCreationRecord], window_ids: set[int]
    ) -> AnalyticalCapacityEstimate | None:
        if not window_ids:
            return None
        empirical_mean_size = mean(creations[transaction_id].transaction_size for transaction_id in window_ids)
        values: dict[str, list[float]] = defaultdict(list)
        for block in finalized:
            if window_ids.intersection(block.transaction_ids):
                if block.configured_block_time <= 0 or empirical_mean_size <= 0:
                    raise ValueError("Analytical capacity requires positive transaction size and configured block time")
                values[block.logical_block_key.consensus_protocol].append(
                    block.configured_block_size / empirical_mean_size / block.configured_block_time
                )
        return AnalyticalCapacityEstimate(empirical_mean_size, InstrumentationMetrics._summaries(values))

    @classmethod
    def _observation_latencies(
        cls, observations: list[BlockObservationRecord], finalized: list[FinalizedBlockMetric]
    ) -> list[ObservationLatency]:
        first_times = {block.logical_block_key: block.first_consensus_decision_time for block in finalized}
        result = []
        seen = set()
        for observation in observations:
            key = cls.logical_block_key(observation)
            if key not in first_times:
                continue
            identity = (key, observation.node_id, observation.cause, observation.observation_time)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(
                ObservationLatency(
                    key,
                    observation.node_id,
                    observation.cause,
                    observation.observation_time,
                    observation.observation_time - first_times[key],
                )
            )
        result.sort(key=lambda item: (item.observation_time, item.logical_block_key, item.node_id, item.cause))
        return result

    @staticmethod
    def _summaries(values: dict[str, list[float]]) -> dict[str, DistributionSummary]:
        return {protocol: DistributionSummary.from_values(items) for protocol, items in sorted(values.items()) if items}
