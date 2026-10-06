"""Deterministic reset/step coordinator for Liu runtime decision epochs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
from typing import Callable

from Chain.Consensus.LiuRuntime.Common.RuntimeEvaluation import (
    LiuRewardStatus,
    LiuRuntimeConstraintEvaluator,
    LiuRuntimeConstraintInput,
    LiuRuntimeEpochEvaluation,
    LiuRuntimeExecutionMeasurement,
)
from Chain.Consensus.LiuRuntime.Common.StateBuilder import LiuRuntimeStateBuildResult
from Chain.Consensus.LiuRuntime.Common.StateEvolution import LiuRuntimeStateEvolution
from Liu.Action import LiuAction
from Liu.Serialization import CanonicalSerializable
from Liu.State import LiuState
from Liu.Validation import require_finite_number, require_integer
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector
from Utils.LiuRuntimeInstrumentation import ProtocolFinalityRecord


@dataclass(frozen=True, slots=True)
class LiuEnvironmentConfig(CanonicalSerializable):
    runtime_seed: int
    fsmc_seed: int
    candidate_seed: int
    stake_gini_threshold: float
    geographic_gini_threshold: float
    finality_multiplier_omega: float
    max_decision_steps: int | None = None
    max_runtime_time_s: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("runtime_seed", "fsmc_seed", "candidate_seed"):
            object.__setattr__(self, field_name, require_integer(getattr(self, field_name), field_name))
        for field_name in ("stake_gini_threshold", "geographic_gini_threshold"):
            value = require_finite_number(getattr(self, field_name), field_name, non_negative=True)
            if value > 1:
                raise ValueError(f"{field_name} must be in [0,1]")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "finality_multiplier_omega",
            require_finite_number(self.finality_multiplier_omega, "finality_multiplier_omega", positive=True),
        )
        if self.max_decision_steps is not None:
            object.__setattr__(
                self,
                "max_decision_steps",
                require_integer(self.max_decision_steps, "max_decision_steps", minimum=1),
            )
        if self.max_runtime_time_s is not None:
            object.__setattr__(
                self,
                "max_runtime_time_s",
                require_finite_number(self.max_runtime_time_s, "max_runtime_time_s", positive=True),
            )

    def to_dict(self) -> dict:
        return {
            "candidate_seed": self.candidate_seed,
            "finality_multiplier_omega": self.finality_multiplier_omega,
            "fsmc_seed": self.fsmc_seed,
            "geographic_gini_threshold": self.geographic_gini_threshold,
            "max_decision_steps": self.max_decision_steps,
            "max_runtime_time_s": self.max_runtime_time_s,
            "runtime_seed": self.runtime_seed,
            "stake_gini_threshold": self.stake_gini_threshold,
        }


@dataclass(frozen=True, slots=True)
class LiuEpochExecutionOutcome:
    end_time: float
    finalized_height: int
    finality_records: tuple[ProtocolFinalityRecord, ...]
    execution_failure_reason: str | None = None
    externally_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "end_time", require_finite_number(self.end_time, "end_time", non_negative=True))
        object.__setattr__(
            self,
            "finalized_height",
            require_integer(self.finalized_height, "finalized_height"),
        )
        records = tuple(self.finality_records)
        if not all(isinstance(record, ProtocolFinalityRecord) for record in records):
            raise ValueError("finality_records must contain ProtocolFinalityRecord values")
        object.__setattr__(self, "finality_records", records)
        if self.execution_failure_reason is not None and (
            not isinstance(self.execution_failure_reason, str) or not self.execution_failure_reason.strip()
        ):
            raise ValueError("execution_failure_reason must be None or a non-empty string")
        if not isinstance(self.externally_truncated, bool):
            raise ValueError("externally_truncated must be boolean")


class LiuEnvironmentRuntimeAdapter(ABC):
    """Narrow bridge to SymBChainSim; the environment never processes events."""

    @property
    @abstractmethod
    def state_evolution(self) -> LiuRuntimeStateEvolution:
        pass

    @property
    @abstractmethod
    def nodes(self) -> tuple:
        pass

    @property
    @abstractmethod
    def current_time(self) -> float:
        pass

    @property
    @abstractmethod
    def current_finalized_height(self) -> int:
        pass

    @abstractmethod
    def execute_one_decision_epoch(
        self,
        target_height: int,
        simulation_deadline: float | None,
    ) -> LiuEpochExecutionOutcome:
        """Drive existing SymBChainSim events until one height finishes or a limit is reached."""

    def activate(self) -> None:
        """Restore adapter-owned compatibility globals, if its simulator requires them."""

    def deactivate(self) -> None:
        """Capture adapter-owned compatibility globals after a coordinated call."""


class CallbackLiuEnvironmentRuntimeAdapter(LiuEnvironmentRuntimeAdapter):
    """Bind an existing SymBChainSim harness to the environment without duplicating it."""

    def __init__(
        self,
        state_evolution: LiuRuntimeStateEvolution,
        nodes: tuple,
        current_time: Callable[[], float],
        current_finalized_height: Callable[[], int],
        execute_one_decision_epoch: Callable[[int, float | None], LiuEpochExecutionOutcome],
        activate: Callable[[], None] | None = None,
        deactivate: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(state_evolution, LiuRuntimeStateEvolution):
            raise ValueError("state_evolution must be a LiuRuntimeStateEvolution")
        callbacks = (current_time, current_finalized_height, execute_one_decision_epoch)
        if not all(callable(callback) for callback in callbacks):
            raise ValueError("runtime adapter providers must be callable")
        self._state_evolution = state_evolution
        self._nodes = tuple(nodes)
        self._current_time_provider = current_time
        self._height_provider = current_finalized_height
        self._execution_provider = execute_one_decision_epoch
        self._activate_hook = activate
        self._deactivate_hook = deactivate

    @property
    def state_evolution(self) -> LiuRuntimeStateEvolution:
        return self._state_evolution

    @property
    def nodes(self) -> tuple:
        return self._nodes

    @property
    def current_time(self) -> float:
        return self._current_time_provider()

    @property
    def current_finalized_height(self) -> int:
        return self._height_provider()

    def execute_one_decision_epoch(self, target_height: int, simulation_deadline: float | None):
        return self._execution_provider(target_height, simulation_deadline)

    def activate(self) -> None:
        if self._activate_hook is not None:
            self._activate_hook()

    def deactivate(self) -> None:
        if self._deactivate_hook is not None:
            self._deactivate_hook()


@dataclass(frozen=True, slots=True)
class LiuResetResult(CanonicalSerializable):
    observation: LiuState
    state_hash: str
    metadata: tuple[tuple[str, str | int | float | bool | None], ...]

    def to_dict(self) -> dict:
        return {
            "metadata": [{"name": name, "value": value} for name, value in self.metadata],
            "observation": self.observation.to_dict(),
            "state_hash": self.state_hash,
        }


@dataclass(frozen=True, slots=True)
class LiuStepResult(CanonicalSerializable):
    observation: LiuState | None
    reward: float
    terminated: bool
    truncated: bool
    status: LiuRewardStatus
    state_t_hash: str
    action_t_hash: str
    evaluation_hash: str
    state_t_plus_1_hash: str | None
    diagnostics: tuple[tuple[str, str | int | float | bool | None], ...]

    def __post_init__(self) -> None:
        reward = require_finite_number(self.reward, "reward", non_negative=True)
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated and truncated must be boolean")
        if not isinstance(self.status, LiuRewardStatus):
            raise ValueError("status must be a LiuRewardStatus")
        if self.observation is None:
            if self.state_t_plus_1_hash is not None:
                raise ValueError("missing observation cannot have a next-state hash")
        elif self.state_t_plus_1_hash != self.observation.deterministic_hash():
            raise ValueError("state_t_plus_1_hash must bind the returned observation")
        object.__setattr__(self, "reward", reward)

    def to_dict(self) -> dict:
        return {
            "action_t_hash": self.action_t_hash,
            "diagnostics": [{"name": name, "value": value} for name, value in self.diagnostics],
            "evaluation_hash": self.evaluation_hash,
            "observation": None if self.observation is None else self.observation.to_dict(),
            "reward": self.reward,
            "state_t_hash": self.state_t_hash,
            "state_t_plus_1_hash": self.state_t_plus_1_hash,
            "status": self.status.value,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class LiuRuntimeEnvironment:
    """RL-ready deterministic coordinator with no policy or learning logic."""

    API_VERSION = "liu_runtime_environment_v1"

    def __init__(
        self,
        config: LiuEnvironmentConfig,
        runtime_factory: Callable[[LiuEnvironmentConfig], LiuEnvironmentRuntimeAdapter],
        evaluator: LiuRuntimeConstraintEvaluator | None = None,
    ) -> None:
        if not isinstance(config, LiuEnvironmentConfig):
            raise ValueError("config must be a LiuEnvironmentConfig")
        if not callable(runtime_factory):
            raise ValueError("runtime_factory must be callable")
        self.config = config
        self.runtime_factory = runtime_factory
        self.evaluator = evaluator or LiuRuntimeConstraintEvaluator()
        self.candidate_rng = random.Random(config.candidate_seed)
        self._runtime: LiuEnvironmentRuntimeAdapter | None = None
        self._current_state: LiuRuntimeStateBuildResult | None = None
        self._episode_started_at: float | None = None
        self._decision_steps = 0
        self._done = False
        self._raw_instrumentation_snapshot: dict | None = None
        self._liu_instrumentation_snapshot: dict | None = None
        self._last_evaluation = None

    @property
    def last_evaluation(self):
        """Most recent immutable evaluation, exposed only for diagnostics."""
        return self._last_evaluation

    @property
    def runtime(self) -> LiuEnvironmentRuntimeAdapter:
        if self._runtime is None:
            raise ValueError("reset() must be called before accessing the runtime")
        return self._runtime

    @property
    def current_state(self) -> LiuRuntimeStateBuildResult:
        if self._current_state is None:
            raise ValueError("reset() must be called before accessing state")
        return self._current_state

    def reset(self) -> LiuResetResult:
        runtime = self.runtime_factory(self.config)
        if not isinstance(runtime, LiuEnvironmentRuntimeAdapter):
            raise ValueError("runtime_factory must return a LiuEnvironmentRuntimeAdapter")
        self._runtime = runtime
        runtime.activate()
        self.candidate_rng = random.Random(self.config.candidate_seed)
        self._decision_steps = 0
        self._done = False
        self._last_evaluation = None
        initial_time = require_finite_number(runtime.current_time, "runtime.current_time", non_negative=True)
        self._episode_started_at = initial_time
        self._current_state = runtime.state_evolution.observe_initial_state(initial_time)
        self._raw_instrumentation_snapshot = InstrumentationCollector.snapshot()
        self._liu_instrumentation_snapshot = LiuRuntimeInstrumentationCollector.snapshot()
        runtime.deactivate()
        return LiuResetResult(
            self._current_state.state,
            self._current_state.state_hash,
            (
                ("api_version", self.API_VERSION),
                ("candidate_seed", self.config.candidate_seed),
                ("fsmc_seed", self.config.fsmc_seed),
                ("runtime_seed", self.config.runtime_seed),
            ),
        )

    def _runtime_deadline(self) -> float | None:
        if self.config.max_runtime_time_s is None:
            return None
        assert self._episode_started_at is not None
        return self._episode_started_at + self.config.max_runtime_time_s

    def step(self, action: LiuAction) -> LiuStepResult:
        if self._runtime is None or self._current_state is None or self._episode_started_at is None:
            raise ValueError("reset() must be called before step()")
        if self._done:
            raise ValueError("episode is complete; call reset() before another step")
        if not isinstance(action, LiuAction):
            raise ValueError("action must be an immutable LiuAction")

        runtime = self.runtime
        runtime.activate()
        if self._raw_instrumentation_snapshot is not None:
            InstrumentationCollector.restore(self._raw_instrumentation_snapshot)
        if self._liu_instrumentation_snapshot is not None:
            LiuRuntimeInstrumentationCollector.restore(self._liu_instrumentation_snapshot)
        state_t = self.current_state
        start_time = require_finite_number(runtime.current_time, "runtime.current_time", non_negative=True)
        start_height = require_integer(runtime.current_finalized_height, "runtime.current_finalized_height")
        target_height = start_height + 1
        action_hash = action.deterministic_hash()
        runtime.state_evolution.begin_epoch(
            action,
            current_finalized_height=start_height,
            boundary_time=start_time,
        )
        outcome = runtime.execute_one_decision_epoch(target_height, self._runtime_deadline())
        if outcome.end_time <= start_time:
            raise ValueError("runtime adapter must advance simulated time during step()")
        reason = outcome.execution_failure_reason
        if reason is None and outcome.externally_truncated and outcome.finalized_height != target_height:
            reason = "external runtime limit before decision epoch completion"
        if reason is None and not outcome.externally_truncated and outcome.finalized_height != target_height:
            reason = "runtime did not finalize exactly the target decision height"
        measurement = LiuRuntimeExecutionMeasurement.from_runtime(
            epoch_id=runtime.state_evolution.action_applicator.current_epoch.epoch_id,
            protocol=action.consensus_protocol,
            epoch_start_time=start_time,
            epoch_end_time=outcome.end_time,
            nodes=runtime.nodes,
            finality_records=outcome.finality_records,
            execution_failure_reason=reason,
        )
        evaluation = self.evaluator.evaluate(
            LiuRuntimeConstraintInput(
                measurement.epoch_id,
                state_t.state,
                action,
                runtime.state_evolution.action_applicator.current_epoch.threat_scenario,
                runtime.state_evolution.state_builder.geographic_model,
                measurement,
                self.config.stake_gini_threshold,
                self.config.geographic_gini_threshold,
                self.config.finality_multiplier_omega,
            )
        )
        self._last_evaluation = evaluation

        self._decision_steps += 1
        reward_result = evaluation.reward_result
        terminated = reward_result.status is LiuRewardStatus.EXECUTION_FAILED and not outcome.externally_truncated
        truncated = outcome.externally_truncated
        next_state = None
        next_hash = None
        if measurement.execution_succeeded:
            self._current_state = runtime.state_evolution.complete_epoch(
                current_finalized_height=target_height,
                boundary_time=outcome.end_time,
                evaluation=evaluation,
            )
            next_state = self._current_state.state
            next_hash = self._current_state.state_hash
            if self.config.max_decision_steps is not None and self._decision_steps >= self.config.max_decision_steps:
                terminated = True
            deadline = self._runtime_deadline()
            if deadline is not None and outcome.end_time >= deadline:
                truncated = True

        self._done = terminated or truncated
        self._raw_instrumentation_snapshot = InstrumentationCollector.snapshot()
        self._liu_instrumentation_snapshot = LiuRuntimeInstrumentationCollector.snapshot()
        runtime.deactivate()
        evaluation_hash = evaluation.deterministic_hash()
        diagnostics = (
            ("decision_step", self._decision_steps - 1),
            ("epoch_id", measurement.epoch_id),
            ("execution_measurement_hash", measurement.deterministic_hash()),
            ("constraint_result_hash", evaluation.constraint_result.deterministic_hash()),
            ("reward_result_hash", reward_result.deterministic_hash()),
            ("pre_execution_screening_applied", False),
        )
        return LiuStepResult(
            next_state,
            reward_result.reward,
            terminated,
            truncated,
            reward_result.status,
            state_t.state_hash,
            action_hash,
            evaluation_hash,
            next_hash,
            diagnostics,
        )

    def screen_candidate(self, generator, action: LiuAction):
        """Convenience screening hook; step() never calls or enforces it."""
        return generator.screen(action, self.current_state.state)
