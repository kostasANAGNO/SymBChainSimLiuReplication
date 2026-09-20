"""Post-execution Liu constraints and reward evaluation over DES evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import floor, isclose
from typing import Iterable

from Liu.Action import LiuAction
from Liu.AnalyticalConsensus import AnalyticalConsensusResult
from Liu.Feasibility import ConstraintResult, FeasibilityResult
from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.SpatialIntensity import GridSpatialIntensityModel
from Liu.State import LiuState
from Liu.Threat import ThreatScenario
from Liu.Validation import require_finite_number, require_integer
from Utils.DecentralizationMetrics import canonical_pairwise_gini
from Utils.LiuRuntimeInstrumentation import ProtocolFinalityRecord


_RUNTIME_PROTOCOL_NAMES = {
    LiuConsensusProtocol.LIU_QUORUM: "LiuQuorum",
    LiuConsensusProtocol.PBFT: "LiuPBFT",
    LiuConsensusProtocol.ZYZZYVA: "LiuZyzzyva",
}


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a SHA-256 hexadecimal string") from error
    return value


@dataclass(frozen=True, slots=True)
class RuntimeFinalizedBlockMeasurement(CanonicalSerializable):
    """Height-scoped protocol finality joined to immutable transaction identity."""

    epoch_id: int
    height: int
    block_digest: str
    request_started_at: float
    protocol_finality_at: float
    transaction_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        epoch_id = require_integer(self.epoch_id, "epoch_id")
        height = require_integer(self.height, "height", minimum=1)
        digest = _require_sha256(self.block_digest, "block_digest")
        started = require_finite_number(self.request_started_at, "request_started_at", non_negative=True)
        finalized = require_finite_number(self.protocol_finality_at, "protocol_finality_at", non_negative=True)
        if finalized < started:
            raise ValueError("protocol_finality_at cannot precede request_started_at")
        transaction_ids = tuple(self.transaction_ids)
        if any(isinstance(tx_id, bool) or not isinstance(tx_id, int) for tx_id in transaction_ids):
            raise ValueError("transaction_ids must contain integers")
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError("a finalized block cannot contain duplicate transaction IDs")
        object.__setattr__(self, "epoch_id", epoch_id)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "block_digest", digest)
        object.__setattr__(self, "request_started_at", started)
        object.__setattr__(self, "protocol_finality_at", finalized)
        object.__setattr__(self, "transaction_ids", tuple(sorted(transaction_ids)))

    @property
    def consensus_latency_s(self) -> float:
        return self.protocol_finality_at - self.request_started_at

    def to_dict(self) -> dict:
        return {
            "block_digest": self.block_digest,
            "epoch_id": self.epoch_id,
            "height": self.height,
            "protocol_finality_at": self.protocol_finality_at,
            "request_started_at": self.request_started_at,
            "transaction_ids": list(self.transaction_ids),
        }


@dataclass(frozen=True, slots=True)
class LiuRuntimeExecutionMeasurement(CanonicalSerializable):
    """One action epoch's measured DES outcome using a half-open time window."""

    epoch_id: int
    protocol: LiuConsensusProtocol
    epoch_start_time: float
    epoch_end_time: float
    finalized_blocks: tuple[RuntimeFinalizedBlockMeasurement, ...]
    execution_failure_reason: str | None = None
    boundary_policy_version: str = "liu_epoch_half_open_finality_window_v1"
    consensus_start_policy_version: str = "liu_initial_request_start_v1"

    def __post_init__(self) -> None:
        epoch_id = require_integer(self.epoch_id, "epoch_id")
        if not isinstance(self.protocol, LiuConsensusProtocol):
            raise ValueError("protocol must be a LiuConsensusProtocol")
        start = require_finite_number(self.epoch_start_time, "epoch_start_time", non_negative=True)
        end = require_finite_number(self.epoch_end_time, "epoch_end_time", non_negative=True)
        if end <= start:
            raise ValueError("epoch_end_time must be after epoch_start_time")
        blocks = tuple(self.finalized_blocks)
        if not all(isinstance(block, RuntimeFinalizedBlockMeasurement) for block in blocks):
            raise ValueError("finalized_blocks must contain RuntimeFinalizedBlockMeasurement values")
        if any(block.epoch_id != epoch_id for block in blocks):
            raise ValueError("all finalized blocks must belong to the measured epoch")
        if any(not (start <= block.protocol_finality_at < end) for block in blocks):
            raise ValueError("protocol finality must lie in the half-open epoch interval [start,end)")
        if any(block.request_started_at < start for block in blocks):
            raise ValueError("consensus-cycle request start must belong to the measured epoch")
        keys = tuple((block.height, block.block_digest) for block in blocks)
        if len(set(keys)) != len(keys):
            raise ValueError("finalized block measurements must be unique")
        reason = self.execution_failure_reason
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("execution_failure_reason must be None or a non-empty string")
        if reason is None and not blocks:
            raise ValueError("a successful action epoch requires protocol-finality evidence")
        heights = tuple(block.height for block in blocks)
        if reason is None and len(set(heights)) != len(heights):
            raise ValueError("multiple protocol-finalized digests at one height are ambiguous")
        for field_name in ("boundary_policy_version", "consensus_start_policy_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "epoch_id", epoch_id)
        object.__setattr__(self, "epoch_start_time", start)
        object.__setattr__(self, "epoch_end_time", end)
        object.__setattr__(
            self,
            "finalized_blocks",
            tuple(sorted(blocks, key=lambda item: (item.height, item.block_digest))),
        )

    @property
    def execution_succeeded(self) -> bool:
        return self.execution_failure_reason is None

    @property
    def duration_s(self) -> float:
        return self.epoch_end_time - self.epoch_start_time

    @property
    def unique_finalized_transaction_ids(self) -> tuple[int, ...]:
        return tuple(sorted({tx_id for block in self.finalized_blocks for tx_id in block.transaction_ids}))

    @property
    def des_consensus_latency_s(self) -> float | None:
        if not self.execution_succeeded:
            return None
        # C3 must hold for every finalized height in an epoch; the maximum is
        # the deterministic scalar sufficient statistic for that conjunction.
        return max(block.consensus_latency_s for block in self.finalized_blocks)

    def to_dict(self) -> dict:
        return {
            "boundary_policy_version": self.boundary_policy_version,
            "consensus_start_policy_version": self.consensus_start_policy_version,
            "epoch_end_time": self.epoch_end_time,
            "epoch_id": self.epoch_id,
            "epoch_start_time": self.epoch_start_time,
            "execution_failure_reason": self.execution_failure_reason,
            "finalized_blocks": [block.to_dict() for block in self.finalized_blocks],
            "protocol": self.protocol.value,
            "unique_finalized_transaction_count": len(self.unique_finalized_transaction_ids),
        }

    @classmethod
    def from_runtime(
        cls,
        *,
        epoch_id: int,
        protocol: LiuConsensusProtocol,
        epoch_start_time: float,
        epoch_end_time: float,
        nodes: Iterable,
        finality_records: Iterable[ProtocolFinalityRecord],
        execution_failure_reason: str | None = None,
    ) -> "LiuRuntimeExecutionMeasurement":
        """Join finality evidence to existing node blocks without mutating either."""
        runtime_name = _RUNTIME_PROTOCOL_NAMES[protocol]
        records = tuple(
            record
            for record in finality_records
            if record.epoch_id == epoch_id
            and record.protocol == runtime_name
            and epoch_start_time <= record.finality_time < epoch_end_time
        )
        unique_records: dict[tuple[int, str], ProtocolFinalityRecord] = {}
        for record in records:
            key = (record.height, record.block_digest)
            existing = unique_records.get(key)
            if existing is not None and existing != record:
                raise ValueError("conflicting protocol-finality records for one logical block")
            unique_records[key] = record

        transaction_ids_by_digest: dict[str, tuple[int, ...]] = {}
        for node in tuple(nodes):
            for block in node.blockchain:
                identity = block.extra_data.get("liu_identity")
                if identity is None:
                    continue
                digest = identity["block_digest"]
                tx_ids = tuple(sorted(transaction.id for transaction in block.transactions))
                existing = transaction_ids_by_digest.get(digest)
                if existing is not None and existing != tx_ids:
                    raise ValueError("nodes disagree on transaction membership for a finalized digest")
                transaction_ids_by_digest[digest] = tx_ids

        blocks = []
        missing = []
        for (_, digest), record in sorted(unique_records.items()):
            if digest not in transaction_ids_by_digest:
                missing.append(digest)
                continue
            blocks.append(
                RuntimeFinalizedBlockMeasurement(
                    record.epoch_id,
                    record.height,
                    digest,
                    record.request_sent_at,
                    record.finality_time,
                    transaction_ids_by_digest[digest],
                )
            )
        reason = execution_failure_reason
        if reason is None and missing:
            reason = "protocol finality could not be joined to finalized block transaction membership"
        if reason is None and not blocks:
            reason = "epoch did not reach valid protocol finality"
        if reason is None and len({block.height for block in blocks}) != len(blocks):
            reason = "multiple competing protocol-finalized blocks exist at one height"
        return cls(epoch_id, protocol, epoch_start_time, epoch_end_time, tuple(blocks), reason)


@dataclass(frozen=True, slots=True)
class LiuRuntimeConstraintInput:
    epoch_id: int
    state: LiuState
    action: LiuAction
    threat_scenario: ThreatScenario
    geographic_model: GridSpatialIntensityModel
    execution_measurement: LiuRuntimeExecutionMeasurement
    analytical_result: AnalyticalConsensusResult
    stake_gini_threshold: float
    geographic_gini_threshold: float
    finality_multiplier_omega: float

    def __post_init__(self) -> None:
        epoch_id = require_integer(self.epoch_id, "epoch_id")
        if not isinstance(self.state, LiuState) or not isinstance(self.action, LiuAction):
            raise ValueError("state and action must be immutable Liu domain values")
        if self.state.node_count != self.action.node_count:
            raise ValueError("state and action must describe the same N nodes")
        if not isinstance(self.threat_scenario, ThreatScenario) or self.threat_scenario.node_count != self.state.node_count:
            raise ValueError("threat_scenario must describe the state node population")
        if not isinstance(self.geographic_model, GridSpatialIntensityModel):
            raise ValueError("geographic_model must be a GridSpatialIntensityModel")
        if self.geographic_model.spatial_profiles != self.state.spatial_profiles:
            raise ValueError("geographic model must use the state's immutable spatial profiles")
        if not isinstance(self.execution_measurement, LiuRuntimeExecutionMeasurement):
            raise ValueError("execution_measurement must be a LiuRuntimeExecutionMeasurement")
        if (
            self.execution_measurement.epoch_id != epoch_id
            or self.execution_measurement.protocol is not self.action.consensus_protocol
        ):
            raise ValueError("execution measurement must align with the executed epoch/action protocol")
        if not isinstance(self.analytical_result, AnalyticalConsensusResult):
            raise ValueError("analytical_result must be an AnalyticalConsensusResult")
        if self.analytical_result.protocol is not self.action.consensus_protocol:
            raise ValueError("analytical result protocol must match the action")
        if not isclose(
            self.analytical_result.block_interval_s,
            self.action.block_interval_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("analytical result block interval must match the action")
        object.__setattr__(self, "epoch_id", epoch_id)
        object.__setattr__(
            self,
            "stake_gini_threshold",
            require_finite_number(self.stake_gini_threshold, "stake_gini_threshold", non_negative=True),
        )
        object.__setattr__(
            self,
            "geographic_gini_threshold",
            require_finite_number(self.geographic_gini_threshold, "geographic_gini_threshold", non_negative=True),
        )
        object.__setattr__(
            self,
            "finality_multiplier_omega",
            require_finite_number(self.finality_multiplier_omega, "finality_multiplier_omega", positive=True),
        )
        if self.stake_gini_threshold > 1 or self.geographic_gini_threshold > 1:
            raise ValueError("Gini thresholds must be in [0,1]")


@dataclass(frozen=True, slots=True)
class LiuRuntimeConstraintResult(CanonicalSerializable):
    epoch_id: int
    state_hash: str
    action_hash: str
    stake_gini: float | None
    stake_gini_threshold: float
    stake_constraint_passed: bool
    geographic_gini: float
    geographic_gini_threshold: float
    geographic_constraint_passed: bool
    malicious_validator_count: int
    tolerated_fault_count: int
    security_constraint_passed: bool
    des_consensus_latency_s: float | None
    des_finality_latency_s: float | None
    finality_limit_s: float
    finality_constraint_passed: bool
    failed_constraints: tuple[str, ...]
    feasible: bool
    feasibility_result: FeasibilityResult
    policy_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_sha256(self.state_hash, "state_hash")
        _require_sha256(self.action_hash, "action_hash")
        if self.feasible != self.feasibility_result.feasible:
            raise ValueError("feasible must equal the nested FeasibilityResult")
        expected_failed = tuple(
            result.constraint_name for result in self.feasibility_result.constraint_results if not result.satisfied
        )
        if tuple(self.failed_constraints) != expected_failed:
            raise ValueError("failed_constraints must match the failed component constraints")

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "des_consensus_latency_s": self.des_consensus_latency_s,
            "des_finality_latency_s": self.des_finality_latency_s,
            "epoch_id": self.epoch_id,
            "failed_constraints": list(self.failed_constraints),
            "feasibility_result": self.feasibility_result.to_dict(),
            "feasible": self.feasible,
            "finality_constraint_passed": self.finality_constraint_passed,
            "finality_limit_s": self.finality_limit_s,
            "geographic_constraint_passed": self.geographic_constraint_passed,
            "geographic_gini": self.geographic_gini,
            "geographic_gini_threshold": self.geographic_gini_threshold,
            "malicious_validator_count": self.malicious_validator_count,
            "policy_versions": [{"name": name, "version": version} for name, version in self.policy_versions],
            "security_constraint_passed": self.security_constraint_passed,
            "stake_constraint_passed": self.stake_constraint_passed,
            "stake_gini": self.stake_gini,
            "stake_gini_threshold": self.stake_gini_threshold,
            "state_hash": self.state_hash,
            "tolerated_fault_count": self.tolerated_fault_count,
        }


class LiuRewardStatus(str, Enum):
    FEASIBLE = "FEASIBLE"
    ACTION_INFEASIBLE = "ACTION_INFEASIBLE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass(frozen=True, slots=True)
class LiuRewardResult(CanonicalSerializable):
    epoch_id: int
    state_hash: str
    action_hash: str
    execution_measurement_hash: str
    constraint_result_hash: str
    status: LiuRewardStatus
    feasible: bool
    liu_throughput: float
    reward: float
    observed_des_tps: float
    analytical_consensus_latency: float
    analytical_finality_latency: float
    reward_policy_version: str
    observed_tps_policy_version: str
    created_at: float

    def __post_init__(self) -> None:
        for field_name in ("state_hash", "action_hash", "execution_measurement_hash", "constraint_result_hash"):
            _require_sha256(getattr(self, field_name), field_name)
        if not isinstance(self.status, LiuRewardStatus):
            raise ValueError("status must be a LiuRewardStatus")
        if self.feasible != (self.status is LiuRewardStatus.FEASIBLE):
            raise ValueError("feasible status is inconsistent")
        for field_name in (
            "liu_throughput", "reward", "observed_des_tps",
            "analytical_consensus_latency", "analytical_finality_latency", "created_at",
        ):
            object.__setattr__(self, field_name, require_finite_number(getattr(self, field_name), field_name, non_negative=True))
        if self.status is not LiuRewardStatus.FEASIBLE and self.reward != 0.0:
            raise ValueError("infeasible or failed executions must have zero reward")
        if not isinstance(self.reward_policy_version, str) or not self.reward_policy_version:
            raise ValueError("reward_policy_version must be a non-empty string")
        if not isinstance(self.observed_tps_policy_version, str) or not self.observed_tps_policy_version:
            raise ValueError("observed_tps_policy_version must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "analytical_consensus_latency": self.analytical_consensus_latency,
            "analytical_finality_latency": self.analytical_finality_latency,
            "constraint_result_hash": self.constraint_result_hash,
            "created_at": self.created_at,
            "epoch_id": self.epoch_id,
            "execution_measurement_hash": self.execution_measurement_hash,
            "feasible": self.feasible,
            "liu_throughput": self.liu_throughput,
            "observed_des_tps": self.observed_des_tps,
            "observed_tps_policy_version": self.observed_tps_policy_version,
            "reward": self.reward,
            "reward_policy_version": self.reward_policy_version,
            "state_hash": self.state_hash,
            "status": self.status.value,
        }

    @property
    def liu_throughput_tps(self) -> float:
        return self.liu_throughput

    @property
    def analytical_consensus_latency_s(self) -> float:
        return self.analytical_consensus_latency

    @property
    def analytical_finality_latency_s(self) -> float:
        return self.analytical_finality_latency


@dataclass(frozen=True, slots=True)
class LiuRuntimeEpochEvaluation(CanonicalSerializable):
    execution_measurement: LiuRuntimeExecutionMeasurement
    constraint_result: LiuRuntimeConstraintResult
    reward_result: LiuRewardResult

    def __post_init__(self) -> None:
        if self.reward_result.execution_measurement_hash != self.execution_measurement.deterministic_hash():
            raise ValueError("reward does not bind the supplied execution measurement")
        if self.reward_result.constraint_result_hash != self.constraint_result.deterministic_hash():
            raise ValueError("reward does not bind the supplied constraint result")

    def to_dict(self) -> dict:
        return {
            "constraint_result": self.constraint_result.to_dict(),
            "execution_measurement": self.execution_measurement.to_dict(),
            "reward_result": self.reward_result.to_dict(),
        }


class LiuRuntimeConstraintEvaluator:
    """Pure post-execution evaluator for Liu C1/C2/C3 and Equation-10 reward."""

    DECENTRALIZATION_POLICY = "liu_selected_validator_gini_constraints_v1"
    SECURITY_POLICY = "liu_protocol_fault_bound_v1"
    FINALITY_POLICY = "liu_des_finality_constraint_v1"
    EPOCH_LATENCY_AGGREGATION_POLICY = "liu_epoch_max_height_consensus_latency_v1"
    REWARD_POLICY = "liu_throughput_reward_v1"
    OBSERVED_TPS_POLICY = "liu_epoch_protocol_finality_tps_v1"

    @staticmethod
    def tolerated_faults(protocol: LiuConsensusProtocol, validator_count: int) -> int:
        if protocol is LiuConsensusProtocol.LIU_QUORUM:
            return 0
        return (validator_count - 1) // 3

    def evaluate_constraints(self, evaluation_input: LiuRuntimeConstraintInput) -> LiuRuntimeConstraintResult:
        action = evaluation_input.action
        state = evaluation_input.state
        selected_stakes = tuple(state.stakes_tokens[node_id] for node_id in action.validator_ids)
        try:
            stake_gini = canonical_pairwise_gini(selected_stakes)
            stake_passed = stake_gini <= evaluation_input.stake_gini_threshold
            stake_reason = None
        except ValueError as error:
            stake_gini = None
            stake_passed = False
            stake_reason = str(error)
        geographic = evaluation_input.geographic_model.evaluate(action.validator_ids)
        geographic_gini = geographic.geographic_gini
        geographic_passed = geographic_gini <= evaluation_input.geographic_gini_threshold

        malicious = evaluation_input.threat_scenario.malicious_validator_count(action.validator_ids)
        tolerated = self.tolerated_faults(action.consensus_protocol, action.validator_count)
        security_passed = malicious <= tolerated

        measurement = evaluation_input.execution_measurement
        des_consensus = measurement.des_consensus_latency_s
        des_finality = None if des_consensus is None else action.block_interval_s + des_consensus
        finality_limit = evaluation_input.finality_multiplier_omega * action.block_interval_s
        finality_passed = des_finality is not None and des_finality <= finality_limit

        components = FeasibilityResult(
            (
                ConstraintResult(
                    "stake_gini",
                    stake_passed,
                    stake_gini,
                    evaluation_input.stake_gini_threshold,
                    stake_reason,
                ),
                ConstraintResult(
                    "geographic_gini",
                    geographic_passed,
                    geographic_gini,
                    evaluation_input.geographic_gini_threshold,
                ),
                ConstraintResult("security", security_passed, float(malicious), float(tolerated)),
                ConstraintResult(
                    "des_finality",
                    finality_passed,
                    des_finality,
                    finality_limit,
                    measurement.execution_failure_reason if des_finality is None else None,
                ),
            )
        )
        failed = tuple(result.constraint_name for result in components.constraint_results if not result.satisfied)
        return LiuRuntimeConstraintResult(
            evaluation_input.epoch_id,
            state.deterministic_hash(),
            action.deterministic_hash(),
            stake_gini,
            evaluation_input.stake_gini_threshold,
            stake_passed,
            geographic_gini,
            evaluation_input.geographic_gini_threshold,
            geographic_passed,
            malicious,
            tolerated,
            security_passed,
            des_consensus,
            des_finality,
            finality_limit,
            finality_passed,
            failed,
            components.feasible,
            components,
            tuple(sorted((
                ("decentralization", self.DECENTRALIZATION_POLICY),
                ("security", self.SECURITY_POLICY),
                ("finality", self.FINALITY_POLICY),
                ("epoch_latency_aggregation", self.EPOCH_LATENCY_AGGREGATION_POLICY),
                ("geography", geographic.estimator_version),
            ))),
        )

    def evaluate(self, evaluation_input: LiuRuntimeConstraintInput) -> LiuRuntimeEpochEvaluation:
        constraints = self.evaluate_constraints(evaluation_input)
        measurement = evaluation_input.execution_measurement
        capacity = floor(
            evaluation_input.action.block_size_mb
            * 1_000_000.0
            / evaluation_input.state.transaction_size_bytes
        )
        liu_throughput = capacity / evaluation_input.action.block_interval_s
        observed_tps = len(measurement.unique_finalized_transaction_ids) / measurement.duration_s
        if not measurement.execution_succeeded:
            status = LiuRewardStatus.EXECUTION_FAILED
        elif constraints.feasible:
            status = LiuRewardStatus.FEASIBLE
        else:
            status = LiuRewardStatus.ACTION_INFEASIBLE
        reward = liu_throughput if status is LiuRewardStatus.FEASIBLE else 0.0
        analytical = evaluation_input.analytical_result
        reward_result = LiuRewardResult(
            evaluation_input.epoch_id,
            constraints.state_hash,
            constraints.action_hash,
            measurement.deterministic_hash(),
            constraints.deterministic_hash(),
            status,
            status is LiuRewardStatus.FEASIBLE,
            liu_throughput,
            reward,
            observed_tps,
            analytical.consensus_delay_s,
            analytical.finality_delay_s,
            self.REWARD_POLICY,
            self.OBSERVED_TPS_POLICY,
            measurement.epoch_end_time,
        )
        return LiuRuntimeEpochEvaluation(measurement, constraints, reward_result)
