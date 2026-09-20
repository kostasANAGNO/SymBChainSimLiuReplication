"""Epoch-boundary FSMC evolution and runtime Liu state observation."""

from __future__ import annotations

from Chain.Consensus.LiuRuntime.Common.ActionApplicator import LiuRuntimeActionApplicator
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.RuntimeEvaluation import (
    LiuRuntimeConstraintEvaluator,
    LiuRuntimeConstraintInput,
    LiuRuntimeEpochEvaluation,
    LiuRuntimeExecutionMeasurement,
)
from Chain.Consensus.LiuRuntime.Common.StateBuilder import LiuRuntimeStateBuildResult, LiuRuntimeStateBuilder
from Liu.Action import LiuAction
from Liu.AnalyticalConsensus import AnalyticalConsensusResult
from Liu.LinkFSMC import LinkFSMCState
from Liu.Validation import require_finite_number
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import (
    LiuRuntimeInstrumentationCollector,
    RuntimeLiuEpochEvaluationRecord,
)


class LiuRuntimeStateEvolution:
    """Coordinate S_t --A_t--> S_(t+1) at finalized epoch boundaries.

    State index is a decision-step index. The action between two recorded states
    atomically creates the latter state's active epoch configuration.
    """

    def __init__(
        self,
        action_applicator: LiuRuntimeActionApplicator,
        state_builder: LiuRuntimeStateBuilder,
        link_fsmc_state: LinkFSMCState,
        rng,
        *,
        initial_epoch_start_time: float = 0.0,
    ) -> None:
        if not isinstance(action_applicator, LiuRuntimeActionApplicator):
            raise ValueError("action_applicator must be a LiuRuntimeActionApplicator")
        if not isinstance(state_builder, LiuRuntimeStateBuilder):
            raise ValueError("state_builder must be a LiuRuntimeStateBuilder")
        if not isinstance(link_fsmc_state, LinkFSMCState):
            raise ValueError("link_fsmc_state must be a LinkFSMCState")
        if rng is None or not callable(getattr(rng, "random", None)):
            raise ValueError("a dedicated external FSMC RNG is required")
        if link_fsmc_state.current_links != action_applicator.current_epoch.link_state_matrix:
            raise ValueError("initial FSMC links must equal the active runtime epoch links")
        self.action_applicator = action_applicator
        self.state_builder = state_builder
        self.link_fsmc_state = link_fsmc_state
        self.rng = rng
        self._initial_epoch_start_time = require_finite_number(
            initial_epoch_start_time,
            "initial_epoch_start_time",
            non_negative=True,
        )
        self._executing_epoch_started_at: float | None = None
        self._executing_action_hash: str | None = None
        self._executing_action: LiuAction | None = None
        self.current_state: LiuRuntimeStateBuildResult | None = None

    def observe_initial_state(self, observation_time: float = 0.0) -> LiuRuntimeStateBuildResult:
        if self.current_state is not None:
            raise ValueError("initial runtime state has already been observed")
        observation_time = require_finite_number(observation_time, "observation_time", non_negative=True)
        if observation_time < self._initial_epoch_start_time:
            raise ValueError("initial observation precedes the configured environment start")
        self.current_state = self.state_builder.build(
            self.action_applicator.current_epoch,
            observation_time=observation_time,
            is_initial_state=True,
        )
        return self.current_state

    def _completed_epoch_transactions(self, boundary_time: float):
        assert self._executing_epoch_started_at is not None
        return tuple(
            record
            for record in InstrumentationCollector.transaction_creations
            if self._executing_epoch_started_at <= record.original_creation_time < boundary_time
        )

    def begin_epoch(
        self,
        action: LiuAction,
        *,
        current_finalized_height: int,
        boundary_time: float,
    ):
        if self.current_state is None:
            raise ValueError("observe_initial_state() must be called before beginning an epoch")
        if self._executing_action_hash is not None:
            raise ValueError("the current action epoch must complete before another action is applied")
        boundary_time = require_finite_number(boundary_time, "boundary_time", non_negative=True)
        if boundary_time < self._initial_epoch_start_time:
            raise ValueError("action boundary precedes the environment start")
        activation = self.action_applicator.apply(
            action,
            current_finalized_height,
            boundary_time,
            self.link_fsmc_state.current_links,
        )
        self._executing_epoch_started_at = boundary_time
        self._executing_action_hash = activation.action_hash
        self._executing_action = action
        return activation

    def evaluate_epoch(
        self,
        evaluator: LiuRuntimeConstraintEvaluator,
        analytical_result: AnalyticalConsensusResult,
        *,
        boundary_time: float,
        stake_gini_threshold: float,
        geographic_gini_threshold: float,
        finality_multiplier_omega: float,
        execution_failure_reason: str | None = None,
    ) -> LiuRuntimeEpochEvaluation:
        """Measure and evaluate the active action epoch before environment evolution."""
        if self.current_state is None or self._executing_action is None or self._executing_epoch_started_at is None:
            raise ValueError("begin_epoch() must apply an action before epoch evaluation")
        if not isinstance(evaluator, LiuRuntimeConstraintEvaluator):
            raise ValueError("evaluator must be a LiuRuntimeConstraintEvaluator")
        boundary_time = require_finite_number(boundary_time, "boundary_time", non_negative=True)
        active = self.action_applicator.current_epoch
        measurement = LiuRuntimeExecutionMeasurement.from_runtime(
            epoch_id=active.epoch_id,
            protocol=self._executing_action.consensus_protocol,
            epoch_start_time=self._executing_epoch_started_at,
            epoch_end_time=boundary_time,
            nodes=self.action_applicator.nodes,
            finality_records=LiuRuntimeInstrumentationCollector.protocol_finalities,
            execution_failure_reason=execution_failure_reason,
        )
        return evaluator.evaluate(
            LiuRuntimeConstraintInput(
                active.epoch_id,
                self.current_state.state,
                self._executing_action,
                active.threat_scenario,
                self.state_builder.geographic_model,
                measurement,
                analytical_result,
                stake_gini_threshold,
                geographic_gini_threshold,
                finality_multiplier_omega,
            )
        )

    def complete_epoch(
        self,
        *,
        current_finalized_height: int,
        boundary_time: float,
        evaluation: LiuRuntimeEpochEvaluation | None = None,
    ) -> LiuRuntimeStateBuildResult:
        if self.current_state is None or self._executing_action_hash is None:
            raise ValueError("begin_epoch() must apply an action before epoch completion")
        boundary_time = require_finite_number(boundary_time, "boundary_time", non_negative=True)
        if boundary_time <= self._executing_epoch_started_at:
            raise ValueError("epoch completion time must be after its action boundary")
        if evaluation is not None:
            if not isinstance(evaluation, LiuRuntimeEpochEvaluation):
                raise ValueError("evaluation must be a LiuRuntimeEpochEvaluation")
            measurement = evaluation.execution_measurement
            if not measurement.execution_succeeded:
                raise ValueError("an execution-failed epoch cannot advance to S_(t+1)")
            if measurement.epoch_id != self.action_applicator.current_epoch.epoch_id:
                raise ValueError("evaluation epoch does not match the active runtime epoch")
            if measurement.epoch_start_time != self._executing_epoch_started_at or measurement.epoch_end_time != boundary_time:
                raise ValueError("evaluation measurement window does not match the executing epoch")
            if evaluation.reward_result.state_hash != self.current_state.state_hash:
                raise ValueError("evaluation is not bound to the current S_t")
            if evaluation.reward_result.action_hash != self._executing_action_hash:
                raise ValueError("evaluation is not bound to the executing A_t")
        samples = self._completed_epoch_transactions(boundary_time)
        if not samples:
            raise ValueError("completed action epoch has no transaction samples for empirical chi")
        self.action_applicator.validate_finalized_boundary(current_finalized_height)
        next_fsmc = self.link_fsmc_state.advance(self.rng)
        active = self.action_applicator.current_epoch
        next_state_context = LiuRuntimeEpochContext(
            active.epoch_configuration,
            active.node_profiles,
            next_fsmc.current_links,
            active.threat_scenario,
        )
        next_state = self.state_builder.build(
            next_state_context,
            observation_time=boundary_time,
            is_initial_state=False,
            previous_epoch_transactions=samples,
            action_hash=self._executing_action_hash,
            previous_state_hash=self.current_state.state_hash,
        )
        self.link_fsmc_state = next_fsmc
        self.current_state = next_state
        if evaluation is not None:
            LiuRuntimeInstrumentationCollector.epoch_evaluations.append(
                RuntimeLiuEpochEvaluationRecord(
                    evaluation.execution_measurement.epoch_id,
                    evaluation.reward_result.state_hash,
                    evaluation.reward_result.action_hash,
                    evaluation.execution_measurement.deterministic_hash(),
                    evaluation.constraint_result.deterministic_hash(),
                    evaluation.reward_result.deterministic_hash(),
                    next_state.state_hash,
                    boundary_time,
                )
            )
        self._executing_epoch_started_at = None
        self._executing_action_hash = None
        self._executing_action = None
        return next_state
