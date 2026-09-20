"""Deterministic candidate-conditioned DQN training infrastructure.

The trainer composes the frozen Liu environment, candidate generator, tensor
encoders, replay buffer, and Q network.  It contains no paper experiment or
simulator-specific policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import random
from statistics import fmean
from time import perf_counter
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from Liu.Action import LiuAction
from Liu.ActionCandidates import LiuActionCandidateGenerator, LiuCandidateRepairDiagnostic
from Liu.DQN import (
    LiuActionTensorEncoder,
    LiuActionTensorEncoding,
    LiuBellmanTargetConfig,
    LiuCandidateActionMasker,
    LiuCandidateGenerationContext,
    LiuCandidateQNetwork,
    LiuCandidateQNetworkConfig,
    LiuCandidateSetConfig,
    LiuReplayBuffer,
    LiuReplayTransition,
    LiuStateTensorEncoder,
    LiuStateTensorEncoding,
    compute_bellman_targets,
    generate_candidate_actions,
    select_greedy,
    select_uniform_candidate,
)
from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.State import LiuState
from Liu.Validation import require_finite_number, require_integer


TRAINER_VERSION = "liu_candidate_dqn_trainer_v1"
EPSILON_POLICY_VERSION = "liu_dqn_linear_epsilon_v1"
LOSS_MSE_VERSION = "liu_dqn_mse_loss_v1"
LOSS_HUBER_VERSION = "liu_dqn_huber_loss_v1"
TARGET_SYNC_VERSION = "liu_dqn_hard_target_sync_v1"
NO_FEASIBLE_POLICY_VERSION = "liu_dqn_no_feasible_candidate_terminate_episode_v1"
CHECKPOINT_SCHEMA_VERSION = "liu_dqn_checkpoint_v1"


@dataclass(frozen=True, slots=True)
class LiuEpsilonSchedule(CanonicalSerializable):
    epsilon_start: float
    epsilon_end: float
    decay_steps: int
    version: str = EPSILON_POLICY_VERSION

    def __post_init__(self) -> None:
        start = require_finite_number(self.epsilon_start, "epsilon_start", non_negative=True)
        end = require_finite_number(self.epsilon_end, "epsilon_end", non_negative=True)
        if start > 1 or end > 1 or end > start:
            raise ValueError("epsilon must satisfy 0 <= epsilon_end <= epsilon_start <= 1")
        object.__setattr__(self, "epsilon_start", start)
        object.__setattr__(self, "epsilon_end", end)
        object.__setattr__(self, "decay_steps", require_integer(self.decay_steps, "decay_steps", minimum=1))

    def value(self, environment_step: int) -> float:
        step = require_integer(environment_step, "environment_step")
        return max(
            self.epsilon_end,
            self.epsilon_start - step * (self.epsilon_start - self.epsilon_end) / self.decay_steps,
        )

    def to_dict(self) -> dict:
        return {
            "decay_steps": self.decay_steps,
            "epsilon_end": self.epsilon_end,
            "epsilon_start": self.epsilon_start,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNTrainingConfig(CanonicalSerializable):
    gamma: float
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float | None
    loss: str
    replay_capacity: int
    replay_warmup_steps: int
    batch_size: int
    update_every_steps: int
    target_sync_steps: int
    candidates_per_step: int
    max_steps_per_episode: int
    epsilon_schedule: LiuEpsilonSchedule
    exploration_seed: int
    replay_seed: int
    candidate_seed: int
    torch_seed: int
    optimizer: str = "adam"
    trainer_version: str = TRAINER_VERSION

    def __post_init__(self) -> None:
        gamma = require_finite_number(self.gamma, "gamma", non_negative=True)
        if gamma > 1:
            raise ValueError("gamma must be in [0,1]")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "learning_rate", require_finite_number(self.learning_rate, "learning_rate", positive=True))
        object.__setattr__(self, "weight_decay", require_finite_number(self.weight_decay, "weight_decay", non_negative=True))
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "gradient_clip_norm", require_finite_number(self.gradient_clip_norm, "gradient_clip_norm", positive=True))
        if self.loss not in {"mse", "huber"}:
            raise ValueError("loss must be mse or huber")
        for field_name, minimum in (
            ("replay_capacity", 1),
            ("replay_warmup_steps", 0),
            ("batch_size", 1),
            ("update_every_steps", 1),
            ("target_sync_steps", 1),
            ("candidates_per_step", 1),
            ("max_steps_per_episode", 1),
        ):
            object.__setattr__(self, field_name, require_integer(getattr(self, field_name), field_name, minimum=minimum))
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if not isinstance(self.epsilon_schedule, LiuEpsilonSchedule):
            raise ValueError("epsilon_schedule must be a LiuEpsilonSchedule")
        for field_name in ("exploration_seed", "replay_seed", "candidate_seed", "torch_seed"):
            object.__setattr__(self, field_name, require_integer(getattr(self, field_name), field_name))
        if self.optimizer != "adam":
            raise ValueError("this milestone supports the configured Adam baseline only")

    @property
    def loss_policy_version(self) -> str:
        return LOSS_MSE_VERSION if self.loss == "mse" else LOSS_HUBER_VERSION

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "candidate_seed": self.candidate_seed,
            "candidates_per_step": self.candidates_per_step,
            "epsilon_schedule": self.epsilon_schedule.to_dict(),
            "exploration_seed": self.exploration_seed,
            "gamma": self.gamma,
            "gradient_clip_norm": self.gradient_clip_norm,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "loss_policy_version": self.loss_policy_version,
            "max_steps_per_episode": self.max_steps_per_episode,
            "optimizer": self.optimizer,
            "replay_capacity": self.replay_capacity,
            "replay_seed": self.replay_seed,
            "replay_warmup_steps": self.replay_warmup_steps,
            "target_sync_policy_version": TARGET_SYNC_VERSION,
            "target_sync_steps": self.target_sync_steps,
            "torch_seed": self.torch_seed,
            "trainer_version": self.trainer_version,
            "update_every_steps": self.update_every_steps,
            "weight_decay": self.weight_decay,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNTrainingState(CanonicalSerializable):
    episode_index: int
    environment_step: int
    optimizer_step: int
    target_sync_count: int
    replay_size: int

    def to_dict(self) -> dict:
        return {
            "environment_step": self.environment_step,
            "episode_index": self.episode_index,
            "optimizer_step": self.optimizer_step,
            "replay_size": self.replay_size,
            "target_sync_count": self.target_sync_count,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNUpdateResult(CanonicalSerializable):
    optimizer_step: int
    loss: float
    mean_selected_q: float
    mean_target_q: float
    mean_td_error: float
    gradient_norm_before_clip: float
    gradient_norm_after_clip: float
    target_synchronized: bool

    def to_dict(self) -> dict:
        return {
            "gradient_norm_after_clip": self.gradient_norm_after_clip,
            "gradient_norm_before_clip": self.gradient_norm_before_clip,
            "loss": self.loss,
            "mean_selected_q": self.mean_selected_q,
            "mean_target_q": self.mean_target_q,
            "mean_td_error": self.mean_td_error,
            "optimizer_step": self.optimizer_step,
            "target_synchronized": self.target_synchronized,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNTrainingStepRecord(CanonicalSerializable):
    episode_index: int
    episode_step: int
    environment_step: int
    epsilon: float
    selected_action_hash: str | None
    selection_mode: str
    candidate_count: int
    screened_out_count: int
    reward: float | None
    status: str
    selected_q: float | None
    loss: float | None
    mean_target_q: float | None
    mean_td_error: float | None
    replay_size: int
    optimizer_step: int
    target_synchronized: bool
    diagnostic_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "candidate_count": self.candidate_count,
            "diagnostic_reason": self.diagnostic_reason,
            "environment_step": self.environment_step,
            "episode_index": self.episode_index,
            "episode_step": self.episode_step,
            "epsilon": self.epsilon,
            "loss": self.loss,
            "mean_target_q": self.mean_target_q,
            "mean_td_error": self.mean_td_error,
            "optimizer_step": self.optimizer_step,
            "replay_size": self.replay_size,
            "reward": self.reward,
            "screened_out_count": self.screened_out_count,
            "selected_action_hash": self.selected_action_hash,
            "selected_q": self.selected_q,
            "selection_mode": self.selection_mode,
            "status": self.status,
            "target_synchronized": self.target_synchronized,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNSelectionDiagnostic(CanonicalSerializable):
    """Read-only detail for experiment reporting; it does not affect training."""

    action: LiuAction
    maximum_feasible_candidate_q: float
    c1_failure_count: int = 0
    c2_failure_count: int = 0
    feasible_candidate_count: int = 0

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "c1_failure_count": self.c1_failure_count,
            "c2_failure_count": self.c2_failure_count,
            "feasible_candidate_count": self.feasible_candidate_count,
            "maximum_feasible_candidate_q": self.maximum_feasible_candidate_q,
        }


@dataclass(frozen=True, slots=True)
class LiuDQNWallClockProfile:
    environment_step: int
    candidate_generation_s: float
    candidate_screening_s: float
    q_scoring_s: float
    environment_execution_s: float
    optimizer_update_s: float
    total_train_step_s: float


@dataclass(frozen=True, slots=True)
class LiuDQNTrainingEpisodeResult(CanonicalSerializable):
    episode_index: int
    total_reward: float
    mean_reward: float
    feasible_steps: int
    infeasible_steps: int
    failed_steps: int
    episode_length: int
    mean_loss: float | None
    termination_reason: str
    step_records: tuple[LiuDQNTrainingStepRecord, ...]

    def to_dict(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "episode_length": self.episode_length,
            "failed_steps": self.failed_steps,
            "feasible_steps": self.feasible_steps,
            "infeasible_steps": self.infeasible_steps,
            "mean_loss": self.mean_loss,
            "mean_reward": self.mean_reward,
            "step_records": [record.to_dict() for record in self.step_records],
            "termination_reason": self.termination_reason,
            "total_reward": self.total_reward,
        }


class LiuDQNCheckpointEnvironment:
    """Non-invasive checkpoint hooks around a frozen reset/step environment."""

    def __init__(
        self,
        environment,
        snapshot_provider: Callable[[], dict[str, Any]],
        restore_provider: Callable[[dict[str, Any]], LiuState],
        metadata_provider: Callable[[], dict[str, Any]],
    ) -> None:
        if not callable(snapshot_provider) or not callable(restore_provider) or not callable(metadata_provider):
            raise ValueError("checkpoint environment providers must be callable")
        self.environment = environment
        self._snapshot_provider = snapshot_provider
        self._restore_provider = restore_provider
        self._metadata_provider = metadata_provider

    def reset(self):
        return self.environment.reset()

    def step(self, action: LiuAction):
        return self.environment.step(action)

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot_provider()

    def restore(self, snapshot: dict[str, Any]) -> LiuState:
        state = self._restore_provider(snapshot)
        if not isinstance(state, LiuState):
            raise ValueError("environment restore must return its current LiuState")
        return state

    def metadata(self) -> dict[str, Any]:
        return self._metadata_provider()


def _gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    squares = [float(torch.sum(parameter.grad.detach() ** 2).item()) for parameter in parameters if parameter.grad is not None]
    return sqrt(sum(squares))


def _checkpoint_encode(value: Any) -> Any:
    """JSON-only tagged encoding; deliberately avoids pickle/torch.save."""
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__kind__": "tensor",
            "data": tensor.tolist(),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
        }
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_checkpoint_encode(item) for item in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_checkpoint_encode(item) for item in value]}
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": [[_checkpoint_encode(key), _checkpoint_encode(item)] for key, item in value.items()],
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"checkpoint value type is not supported: {type(value).__name__}")


def _checkpoint_decode(value: Any) -> Any:
    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value["__kind__"]
    if kind == "tensor":
        dtype = getattr(torch, value["dtype"])
        return torch.tensor(value["data"], dtype=dtype).reshape(tuple(value["shape"]))
    if kind == "tuple":
        return tuple(_checkpoint_decode(item) for item in value["items"])
    if kind == "list":
        return [_checkpoint_decode(item) for item in value["items"]]
    if kind == "dict":
        return {_checkpoint_decode(key): _checkpoint_decode(item) for key, item in value["items"]}
    raise ValueError(f"unknown checkpoint value kind: {kind}")


def _action_from_dict(value: dict[str, Any]) -> LiuAction:
    return LiuAction(
        value["node_count"],
        value["validator_count"],
        tuple(value["validator_ids"]),
        LiuConsensusProtocol(value["consensus_protocol"]),
        value["block_size_mb"],
        value["block_interval_s"],
    )


def _state_encoding_from_dict(value: dict[str, Any]) -> LiuStateTensorEncoding:
    return LiuStateTensorEncoding(
        tuple(value["values"]),
        value["node_count"],
        value["source_state_hash"],
        value["normalization_hash"],
        value["version"],
    )


def _action_encoding_from_dict(value: dict[str, Any]) -> LiuActionTensorEncoding:
    return LiuActionTensorEncoding(
        tuple(value["values"]),
        value["node_count"],
        value["source_action_hash"],
        value["version"],
    )


def _candidate_context_from_dict(value: dict[str, Any] | None) -> LiuCandidateGenerationContext | None:
    if value is None:
        return None
    return LiuCandidateGenerationContext(
        value["state_hash"],
        value["candidate_seed"],
        value["candidates_per_step"],
        value["candidate_domain_hash"],
        value["generator_policy_version"],
    )


def _transition_from_dict(value: dict[str, Any]) -> LiuReplayTransition:
    return LiuReplayTransition(
        _state_encoding_from_dict(value["state_encoding"]),
        _action_encoding_from_dict(value["action_encoding"]),
        value["reward"],
        None if value["next_state_encoding"] is None else _state_encoding_from_dict(value["next_state_encoding"]),
        value["terminated"],
        value["truncated"],
        tuple(_action_from_dict(action) for action in value["next_candidate_actions"]),
        tuple(value["next_candidate_mask"]),
        _candidate_context_from_dict(value["next_candidate_context"]),
    )


def _step_record_from_dict(value: dict[str, Any]) -> LiuDQNTrainingStepRecord:
    return LiuDQNTrainingStepRecord(
        value["episode_index"],
        value["episode_step"],
        value["environment_step"],
        value["epsilon"],
        value["selected_action_hash"],
        value["selection_mode"],
        value["candidate_count"],
        value["screened_out_count"],
        value["reward"],
        value["status"],
        value["selected_q"],
        value["loss"],
        value["mean_target_q"],
        value["mean_td_error"],
        value["replay_size"],
        value["optimizer_step"],
        value["target_synchronized"],
        value["diagnostic_reason"],
    )


def _update_result_from_dict(value: dict[str, Any]) -> LiuDQNUpdateResult:
    return LiuDQNUpdateResult(
        value["optimizer_step"],
        value["loss"],
        value["mean_selected_q"],
        value["mean_target_q"],
        value["mean_td_error"],
        value["gradient_norm_before_clip"],
        value["gradient_norm_after_clip"],
        value["target_synchronized"],
    )


class LiuDQNTrainer:
    """Modular deterministic trainer for candidate-conditioned Q(S,A)."""

    def __init__(
        self,
        config: LiuDQNTrainingConfig,
        environment: LiuDQNCheckpointEnvironment,
        candidate_generator: LiuActionCandidateGenerator,
        state_encoder: LiuStateTensorEncoder,
        action_encoder: LiuActionTensorEncoder,
        network_config: LiuCandidateQNetworkConfig,
    ) -> None:
        if not isinstance(config, LiuDQNTrainingConfig):
            raise ValueError("config must be a LiuDQNTrainingConfig")
        if not isinstance(environment, LiuDQNCheckpointEnvironment):
            raise ValueError("environment must be a LiuDQNCheckpointEnvironment")
        if not isinstance(candidate_generator, LiuActionCandidateGenerator):
            raise ValueError("candidate_generator must be a LiuActionCandidateGenerator")
        if state_encoder.dimension != network_config.state_dimension or action_encoder.dimension != network_config.action_dimension:
            raise ValueError("encoder dimensions must match the Q-network configuration")
        if state_encoder.normalization.node_count != action_encoder.node_count:
            raise ValueError("state and action encoders must use the same N")
        if network_config.device != "cpu":
            raise ValueError("this deterministic training milestone is CPU-only")

        self.config = config
        self.environment = environment
        self.candidate_generator = candidate_generator
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.network_config = network_config
        self.online_network = LiuCandidateQNetwork(network_config)
        self.target_network = LiuCandidateQNetwork(network_config)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_function = nn.MSELoss() if config.loss == "mse" else nn.SmoothL1Loss()
        self.replay_buffer = LiuReplayBuffer(config.replay_capacity)
        self.exploration_rng = random.Random(config.exploration_seed)
        self.replay_rng = random.Random(config.replay_seed)
        self.candidate_rng = random.Random(config.candidate_seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(config.torch_seed)
        self.candidate_config = LiuCandidateSetConfig(config.candidates_per_step)
        self.masker = LiuCandidateActionMasker()
        self.bellman_config = LiuBellmanTargetConfig(config.gamma, True)

        self._episode_index = 0
        self._environment_step = 0
        self._optimizer_step = 0
        self._target_sync_count = 0
        self._current_observation: LiuState | None = None
        self._episode_step = 0
        self._episode_records: list[LiuDQNTrainingStepRecord] = []
        self._all_step_records: list[LiuDQNTrainingStepRecord] = []
        self._update_results: list[LiuDQNUpdateResult] = []
        self._pending_candidates: tuple[LiuAction, ...] | None = None
        self._pending_mask: tuple[bool, ...] | None = None
        self._pending_context: LiuCandidateGenerationContext | None = None
        self._pending_repair_diagnostic: LiuCandidateRepairDiagnostic | None = None
        self._pending_repair_wall_s = 0.0
        self._last_selection_diagnostic: LiuDQNSelectionDiagnostic | None = None
        self._selection_diagnostics: list[LiuDQNSelectionDiagnostic] = []
        self._wall_clock_profiles: list[LiuDQNWallClockProfile] = []
        self._candidate_repair_diagnostics: list[LiuCandidateRepairDiagnostic | None] = []
        self._candidate_repair_wall_profiles: list[float] = []
        self._last_candidate_generation_s = 0.0
        self._last_candidate_screening_s = 0.0

    @property
    def training_state(self) -> LiuDQNTrainingState:
        return LiuDQNTrainingState(
            self._episode_index,
            self._environment_step,
            self._optimizer_step,
            self._target_sync_count,
            len(self.replay_buffer),
        )

    @property
    def step_records(self) -> tuple[LiuDQNTrainingStepRecord, ...]:
        return tuple(self._all_step_records)

    @property
    def update_results(self) -> tuple[LiuDQNUpdateResult, ...]:
        return tuple(self._update_results)

    @property
    def last_selection_diagnostic(self) -> LiuDQNSelectionDiagnostic | None:
        return self._last_selection_diagnostic

    @property
    def selection_diagnostics(self) -> tuple[LiuDQNSelectionDiagnostic, ...]:
        return tuple(self._selection_diagnostics)

    @property
    def wall_clock_profiles(self) -> tuple[LiuDQNWallClockProfile, ...]:
        return tuple(self._wall_clock_profiles)

    @property
    def candidate_repair_diagnostics(self) -> tuple[LiuCandidateRepairDiagnostic | None, ...]:
        return tuple(self._candidate_repair_diagnostics)

    @property
    def candidate_repair_wall_profiles(self) -> tuple[float, ...]:
        return tuple(self._candidate_repair_wall_profiles)

    def _begin_episode(self) -> None:
        result = self.environment.reset()
        if not isinstance(result.observation, LiuState):
            raise ValueError("environment reset must return a LiuState observation")
        self._current_observation = result.observation
        self._episode_step = 0
        self._episode_records = []
        self._pending_candidates = None
        self._pending_mask = None
        self._pending_context = None
        self._pending_repair_diagnostic = None
        self._pending_repair_wall_s = 0.0
        self._last_selection_diagnostic = None

    def _candidate_set(self, state: LiuState) -> tuple[
        tuple[LiuAction, ...], tuple[bool, ...], LiuCandidateGenerationContext, int,
        LiuCandidateRepairDiagnostic | None, float,
    ]:
        if self._pending_candidates is not None:
            assert self._pending_mask is not None and self._pending_context is not None
            candidates, mask, context = self._pending_candidates, self._pending_mask, self._pending_context
            repair_diagnostic = self._pending_repair_diagnostic
            repair_wall_s = self._pending_repair_wall_s
            self._pending_candidates = self._pending_mask = self._pending_context = None
            self._pending_repair_diagnostic = None
            self._pending_repair_wall_s = 0.0
            self._last_candidate_generation_s = 0.0
            self._last_candidate_screening_s = 0.0
            return candidates, mask, context, sum(not item for item in mask), repair_diagnostic, repair_wall_s
        step_seed = self.candidate_rng.randrange(2**63)
        started = perf_counter()
        candidates, context = generate_candidate_actions(
            self.candidate_generator,
            state,
            self.candidate_config,
            candidate_seed=step_seed,
        )
        self._last_candidate_generation_s = perf_counter() - started
        started = perf_counter()
        screening = self.masker.screen(self.candidate_generator, state, candidates)
        self._last_candidate_screening_s = perf_counter() - started
        return (
            candidates, screening.pre_execution_mask, context, screening.screened_out_count,
            getattr(self.candidate_generator, "last_repair_diagnostic", None),
            float(getattr(self.candidate_generator, "last_repair_wall_s", 0.0)),
        )

    def _score(self, state: LiuState, candidates: Sequence[LiuAction]) -> Tensor:
        state_tensor = self.state_encoder.encode(state).as_tensor(
            dtype=self.online_network.model[0].weight.dtype,
            device=self.online_network.model[0].weight.device,
        )
        action_tensors = torch.stack(
            tuple(
                self.action_encoder.encode(action).as_tensor(dtype=state_tensor.dtype, device=state_tensor.device)
                for action in candidates
            )
        )
        with torch.no_grad():
            return self.online_network.score_candidates(state_tensor, action_tensors)

    def _target_candidate_values(self, transitions: Sequence[LiuReplayTransition]) -> tuple[Tensor, Tensor]:
        batch_size = len(transitions)
        maximum = max((len(transition.next_candidate_actions) for transition in transitions), default=0)
        maximum = max(maximum, 1)
        dtype = self.online_network.model[0].weight.dtype
        device = self.online_network.model[0].weight.device
        values = torch.zeros((batch_size, maximum), dtype=dtype, device=device)
        mask = torch.zeros((batch_size, maximum), dtype=torch.bool, device=device)
        with torch.no_grad():
            for row, transition in enumerate(transitions):
                if transition.next_state_encoding is None or not transition.next_candidate_actions:
                    continue
                state = transition.next_state_encoding.as_tensor(dtype=dtype, device=device)
                actions = torch.stack(
                    tuple(
                        self.action_encoder.encode(action).as_tensor(dtype=dtype, device=device)
                        for action in transition.next_candidate_actions
                    )
                )
                scored = self.target_network.score_candidates(state, actions)
                count = len(transition.next_candidate_actions)
                values[row, :count] = scored
                mask[row, :count] = torch.tensor(transition.next_candidate_mask, dtype=torch.bool, device=device)
        return values, mask

    def _update(self) -> LiuDQNUpdateResult:
        transitions = self.replay_buffer.sample(self.config.batch_size, self.replay_rng)
        dtype = self.online_network.model[0].weight.dtype
        device = self.online_network.model[0].weight.device
        state_actions = torch.stack(
            tuple(
                torch.cat(
                    (
                        transition.state_encoding.as_tensor(dtype=dtype, device=device),
                        transition.action_encoding.as_tensor(dtype=dtype, device=device),
                    )
                )
                for transition in transitions
            )
        )
        selected_q = self.online_network(state_actions)
        rewards = torch.tensor(tuple(transition.reward for transition in transitions), dtype=dtype, device=device)
        terminated = torch.tensor(tuple(transition.terminated for transition in transitions), dtype=torch.bool, device=device)
        truncated = torch.tensor(tuple(transition.truncated for transition in transitions), dtype=torch.bool, device=device)
        next_values, next_mask = self._target_candidate_values(transitions)
        with torch.no_grad():
            targets = compute_bellman_targets(rewards, terminated, truncated, next_values, next_mask, self.bellman_config)
        loss = self.loss_function(selected_q, targets)
        if not torch.isfinite(loss):
            raise ValueError("DQN loss is not finite")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = tuple(self.online_network.parameters())
        before = _gradient_norm(parameters)
        if self.config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm)
        after = _gradient_norm(parameters)
        self.optimizer.step()
        self._optimizer_step += 1
        synchronized = False
        if self._optimizer_step % self.config.target_sync_steps == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
            self.target_network.requires_grad_(False)
            self._target_sync_count += 1
            synchronized = True
        td = targets - selected_q.detach()
        return LiuDQNUpdateResult(
            self._optimizer_step,
            float(loss.detach().item()),
            float(selected_q.detach().mean().item()),
            float(targets.mean().item()),
            float(td.mean().item()),
            before,
            after,
            synchronized,
        )

    def train_step(self) -> LiuDQNTrainingStepRecord:
        train_step_started = perf_counter()
        if self._current_observation is None:
            self._begin_episode()
        assert self._current_observation is not None
        state = self._current_observation
        epsilon = self.config.epsilon_schedule.value(self._environment_step)
        candidates, mask, context, screened_out, repair_diagnostic, repair_wall_s = self._candidate_set(state)
        self._candidate_repair_diagnostics.append(repair_diagnostic)
        self._candidate_repair_wall_profiles.append(repair_wall_s)
        candidate_generation_elapsed = self._last_candidate_generation_s
        candidate_screening_elapsed = self._last_candidate_screening_s
        if not any(mask):
            self._last_selection_diagnostic = None
            record = LiuDQNTrainingStepRecord(
                self._episode_index,
                self._episode_step,
                self._environment_step,
                epsilon,
                None,
                "none",
                len(candidates),
                screened_out,
                None,
                "NO_FEASIBLE_CANDIDATE",
                None,
                None,
                None,
                None,
                len(self.replay_buffer),
                self._optimizer_step,
                False,
                NO_FEASIBLE_POLICY_VERSION,
            )
            self._episode_records.append(record)
            self._all_step_records.append(record)
            self._current_observation = None
            self._wall_clock_profiles.append(
                LiuDQNWallClockProfile(
                    record.environment_step,
                    candidate_generation_elapsed,
                    candidate_screening_elapsed,
                    0.0,
                    0.0,
                    0.0,
                    perf_counter() - train_step_started,
                )
            )
            return record

        q_started = perf_counter()
        q_values = self._score(state, candidates)
        q_elapsed = perf_counter() - q_started
        exploratory = self.exploration_rng.random() < epsilon
        selection = (
            select_uniform_candidate(candidates, mask, self.exploration_rng)
            if exploratory
            else select_greedy(candidates, q_values, mask)
        )
        action = candidates[selection.candidate_index]
        selected_q = float(q_values[selection.candidate_index].item())
        screenings = tuple(self.candidate_generator.screen(candidate, state) for candidate in candidates)
        self._last_selection_diagnostic = LiuDQNSelectionDiagnostic(
            action,
            max(float(q_values[index].item()) for index, allowed in enumerate(mask) if allowed),
            sum(not item.c1_passed for item in screenings),
            sum(not item.c2_passed for item in screenings),
            sum(item.pre_execution_feasible for item in screenings),
        )
        self._selection_diagnostics.append(self._last_selection_diagnostic)
        environment_started = perf_counter()
        result = self.environment.step(action)
        environment_elapsed = perf_counter() - environment_started
        next_state = result.observation
        horizon_truncated = self._episode_step + 1 >= self.config.max_steps_per_episode
        terminated = bool(result.terminated)
        truncated = bool(result.truncated or horizon_truncated)

        next_candidates: tuple[LiuAction, ...] = ()
        next_mask: tuple[bool, ...] = ()
        next_context: LiuCandidateGenerationContext | None = None
        next_encoding: LiuStateTensorEncoding | None = None
        if next_state is not None:
            next_encoding = self.state_encoder.encode(next_state)
        if next_state is not None and not terminated:
            next_candidate_started = perf_counter()
            next_seed = self.candidate_rng.randrange(2**63)
            next_candidates, next_context = generate_candidate_actions(
                self.candidate_generator,
                next_state,
                self.candidate_config,
                candidate_seed=next_seed,
            )
            next_repair_diagnostic = getattr(self.candidate_generator, "last_repair_diagnostic", None)
            next_repair_wall_s = float(getattr(self.candidate_generator, "last_repair_wall_s", 0.0))
            candidate_generation_elapsed += perf_counter() - next_candidate_started
            next_screening_started = perf_counter()
            next_screening = self.masker.screen(self.candidate_generator, next_state, next_candidates)
            candidate_screening_elapsed += perf_counter() - next_screening_started
            next_mask = next_screening.pre_execution_mask
            if not truncated:
                self._pending_candidates = next_candidates
                self._pending_mask = next_mask
                self._pending_context = next_context
                self._pending_repair_diagnostic = next_repair_diagnostic
                self._pending_repair_wall_s = next_repair_wall_s

        transition = LiuReplayTransition(
            self.state_encoder.encode(state),
            self.action_encoder.encode(action),
            result.reward,
            next_encoding,
            terminated,
            truncated,
            next_candidates,
            next_mask,
            next_context,
        )
        self.replay_buffer.append(transition)
        self._environment_step += 1
        self._episode_step += 1

        update = None
        optimizer_elapsed = 0.0
        warm = self._environment_step >= self.config.replay_warmup_steps
        cadence = self._environment_step % self.config.update_every_steps == 0
        enough = len(self.replay_buffer) >= self.config.batch_size
        if warm and cadence and enough:
            optimizer_started = perf_counter()
            update = self._update()
            optimizer_elapsed = perf_counter() - optimizer_started
            self._update_results.append(update)

        status = result.status.value if hasattr(result.status, "value") else str(result.status)
        reason = None
        if terminated:
            reason = "environment_terminated"
        elif result.truncated:
            reason = "environment_truncated"
        elif horizon_truncated:
            reason = "trainer_episode_horizon"
        record = LiuDQNTrainingStepRecord(
            self._episode_index,
            self._episode_step - 1,
            self._environment_step - 1,
            epsilon,
            action.deterministic_hash(),
            "exploratory" if exploratory else "greedy",
            len(candidates),
            screened_out,
            result.reward,
            status,
            selected_q,
            None if update is None else update.loss,
            None if update is None else update.mean_target_q,
            None if update is None else update.mean_td_error,
            len(self.replay_buffer),
            self._optimizer_step,
            False if update is None else update.target_synchronized,
            reason,
        )
        self._episode_records.append(record)
        self._all_step_records.append(record)
        self._current_observation = None if terminated or truncated else next_state
        self._wall_clock_profiles.append(
            LiuDQNWallClockProfile(
                record.environment_step,
                candidate_generation_elapsed,
                candidate_screening_elapsed,
                q_elapsed,
                environment_elapsed,
                optimizer_elapsed,
                perf_counter() - train_step_started,
            )
        )
        return record

    def _finish_episode(self, reason: str) -> LiuDQNTrainingEpisodeResult:
        rewards = tuple(record.reward for record in self._episode_records if record.reward is not None)
        losses = tuple(record.loss for record in self._episode_records if record.loss is not None)
        feasible = sum(record.status == "feasible" for record in self._episode_records)
        infeasible = sum(record.status == "infeasible" for record in self._episode_records)
        failed = sum(record.status in {"execution_failed", "NO_FEASIBLE_CANDIDATE"} for record in self._episode_records)
        result = LiuDQNTrainingEpisodeResult(
            self._episode_index,
            sum(rewards),
            fmean(rewards) if rewards else 0.0,
            feasible,
            infeasible,
            failed,
            len(rewards),
            fmean(losses) if losses else None,
            reason,
            tuple(self._episode_records),
        )
        self._episode_index += 1
        self._current_observation = None
        self._episode_records = []
        self._episode_step = 0
        self._pending_candidates = self._pending_mask = self._pending_context = None
        self._pending_repair_diagnostic = None
        self._pending_repair_wall_s = 0.0
        return result

    def train_episode(self) -> LiuDQNTrainingEpisodeResult:
        if self._current_observation is None:
            self._begin_episode()
        while True:
            record = self.train_step()
            if record.status == "NO_FEASIBLE_CANDIDATE":
                return self._finish_episode(NO_FEASIBLE_POLICY_VERSION)
            if record.diagnostic_reason is not None:
                return self._finish_episode(record.diagnostic_reason)

    def train(self, num_episodes: int) -> tuple[LiuDQNTrainingEpisodeResult, ...]:
        count = require_integer(num_episodes, "num_episodes", minimum=1)
        return tuple(self.train_episode() for _ in range(count))

    def _checkpoint_manifest(self) -> dict[str, Any]:
        pending = None
        if self._pending_candidates is not None:
            assert self._pending_mask is not None and self._pending_context is not None
            pending = {
                "actions": [action.to_dict() for action in self._pending_candidates],
                "context": self._pending_context.to_dict(),
                "mask": list(self._pending_mask),
            }
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "config_hash": self.config.deterministic_hash(),
            "environment_metadata": self.environment.metadata(),
            "environment_snapshot": self.environment.snapshot(),
            "all_step_records": [record.to_dict() for record in self._all_step_records],
            "candidate_repair_diagnostics": [
                None if value is None else value.to_dict()
                for value in self._candidate_repair_diagnostics
            ],
            "episode_records": [record.to_dict() for record in self._episode_records],
            "network_config_hash": self.network_config.deterministic_hash(),
            "online_network": _checkpoint_encode(self.online_network.state_dict()),
            "optimizer": _checkpoint_encode(self.optimizer.state_dict()),
            "pending_candidate_set": pending,
            "replay": [transition.to_dict() for transition in self.replay_buffer.snapshot()],
            "rng_states": {
                "candidate": _checkpoint_encode(self.candidate_rng.getstate()),
                "exploration": _checkpoint_encode(self.exploration_rng.getstate()),
                "replay": _checkpoint_encode(self.replay_rng.getstate()),
                "trainer_torch_cpu": _checkpoint_encode(self.torch_rng.get_state()),
            },
            "target_network": _checkpoint_encode(self.target_network.state_dict()),
            "update_results": [result.to_dict() for result in self._update_results],
            "trainer_state": {
                **self.training_state.to_dict(),
                "episode_step": self._episode_step,
                "has_active_observation": self._current_observation is not None,
            },
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            self._checkpoint_manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        if manifest.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported Liu DQN checkpoint schema")
        if manifest.get("config_hash") != self.config.deterministic_hash():
            raise ValueError("checkpoint training configuration hash mismatch")
        if manifest.get("network_config_hash") != self.network_config.deterministic_hash():
            raise ValueError("checkpoint network configuration hash mismatch")
        if manifest.get("environment_metadata") != self.environment.metadata():
            raise ValueError("checkpoint environment reset/run metadata mismatch")

        self.online_network.load_state_dict(_checkpoint_decode(manifest["online_network"]))
        self.target_network.load_state_dict(_checkpoint_decode(manifest["target_network"]))
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer.load_state_dict(_checkpoint_decode(manifest["optimizer"]))

        self.replay_buffer = LiuReplayBuffer(self.config.replay_capacity)
        for value in manifest["replay"]:
            self.replay_buffer.append(_transition_from_dict(value))
        self.candidate_rng.setstate(_checkpoint_decode(manifest["rng_states"]["candidate"]))
        self.exploration_rng.setstate(_checkpoint_decode(manifest["rng_states"]["exploration"]))
        self.replay_rng.setstate(_checkpoint_decode(manifest["rng_states"]["replay"]))
        self.torch_rng.set_state(_checkpoint_decode(manifest["rng_states"]["trainer_torch_cpu"]))

        state = manifest["trainer_state"]
        self._episode_index = state["episode_index"]
        self._environment_step = state["environment_step"]
        self._optimizer_step = state["optimizer_step"]
        self._target_sync_count = state["target_sync_count"]
        self._episode_step = state["episode_step"]
        self._episode_records = [_step_record_from_dict(value) for value in manifest["episode_records"]]
        self._all_step_records = [_step_record_from_dict(value) for value in manifest["all_step_records"]]
        self._candidate_repair_diagnostics = [
            None if value is None else LiuCandidateRepairDiagnostic(
                value["invoked"], value["succeeded"], value["replaced_candidate_index"],
                value["before_stake_gini"], value["after_stake_gini"],
                value["before_geographic_gini"], value["after_geographic_gini"],
                tuple(tuple(item) for item in value["validator_swaps"]),
                value["repair_iterations"], value["start_candidates_examined"],
                tuple(value.get("repaired_validator_ids", ())),
                value.get("repaired_action_hash"), value["policy_version"],
            )
            for value in manifest.get("candidate_repair_diagnostics", [])
        ]
        self._candidate_repair_wall_profiles = [0.0] * len(self._candidate_repair_diagnostics)
        self._update_results = [_update_result_from_dict(value) for value in manifest["update_results"]]
        restored_state = self.environment.restore(manifest["environment_snapshot"])
        self._current_observation = restored_state if state["has_active_observation"] else None
        pending = manifest["pending_candidate_set"]
        if pending is None:
            self._pending_candidates = self._pending_mask = self._pending_context = None
            self._pending_repair_diagnostic = None
            self._pending_repair_wall_s = 0.0
        else:
            self._pending_candidates = tuple(_action_from_dict(value) for value in pending["actions"])
            self._pending_mask = tuple(pending["mask"])
            self._pending_context = _candidate_context_from_dict(pending["context"])
            self._pending_repair_diagnostic = None
            self._pending_repair_wall_s = 0.0
