"""Candidate-conditioned DQN representations and training infrastructure.

This module deliberately contains no optimizer or episode-training loop.  The
candidate-conditioned Q(S, A) interface is a SymBChainSim reconstruction for
the Liu action space; it is not claimed as the neural architecture of the
paper.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from Liu.Action import LiuAction
from Liu.ActionCandidates import LiuActionCandidateGenerator, LiuActionScreeningResult
from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.State import LiuState
from Liu.Validation import require_finite_number, require_integer


STATE_NORMALIZATION_VERSION = "liu_state_normalization_v1"
ACTION_ENCODING_VERSION = "liu_action_encoding_v1"
CANDIDATE_MASKING_POLICY_VERSION = "liu_candidate_mask_c1_c2_negative_infinity_v1"
Q_NETWORK_ARCHITECTURE_VERSION = "liu_candidate_q_mlp_v1"
BELLMAN_TARGET_POLICY_VERSION = "liu_bellman_target_bootstrap_truncation_v1"


def _finite_tuple(values: Iterable[float], field_name: str) -> tuple[float, ...]:
    return tuple(require_finite_number(value, f"{field_name}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True, slots=True)
class LiuStateNormalizationConfig(CanonicalSerializable):
    """Reset-time bounds used by the fixed-shape state representation."""

    node_count: int
    max_transaction_size_bytes: float
    max_stake_tokens: float
    region_width_km: float
    region_height_km: float
    max_computational_capability_ghz: float
    max_link_rate_mbps: float
    diagonal_link_sentinel: float = 0.0
    version: str = STATE_NORMALIZATION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_count", require_integer(self.node_count, "node_count", minimum=1))
        for field_name in (
            "max_transaction_size_bytes",
            "max_stake_tokens",
            "region_width_km",
            "region_height_km",
            "max_computational_capability_ghz",
            "max_link_rate_mbps",
        ):
            object.__setattr__(self, field_name, require_finite_number(getattr(self, field_name), field_name, positive=True))
        object.__setattr__(
            self,
            "diagonal_link_sentinel",
            require_finite_number(self.diagonal_link_sentinel, "diagonal_link_sentinel"),
        )
        if self.version != STATE_NORMALIZATION_VERSION:
            raise ValueError(f"normalization version must be {STATE_NORMALIZATION_VERSION}")

    def to_dict(self) -> dict:
        return {
            "diagonal_link_sentinel": self.diagonal_link_sentinel,
            "max_computational_capability_ghz": self.max_computational_capability_ghz,
            "max_link_rate_mbps": self.max_link_rate_mbps,
            "max_stake_tokens": self.max_stake_tokens,
            "max_transaction_size_bytes": self.max_transaction_size_bytes,
            "node_count": self.node_count,
            "region_height_km": self.region_height_km,
            "region_width_km": self.region_width_km,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LiuStateTensorEncoding(CanonicalSerializable):
    values: tuple[float, ...]
    node_count: int
    source_state_hash: str
    normalization_hash: str
    version: str = STATE_NORMALIZATION_VERSION

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        values = _finite_tuple(self.values, "values")
        expected = 1 + 4 * node_count + node_count * node_count
        if len(values) != expected:
            raise ValueError(f"state encoding must contain exactly {expected} values for N={node_count}")
        if not isinstance(self.source_state_hash, str) or not self.source_state_hash:
            raise ValueError("source_state_hash must be a non-empty string")
        if not isinstance(self.normalization_hash, str) or not self.normalization_hash:
            raise ValueError("normalization_hash must be a non-empty string")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "values", values)

    @property
    def dimension(self) -> int:
        return len(self.values)

    def as_tensor(self, *, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> Tensor:
        return torch.tensor(self.values, dtype=dtype, device=device)

    def to_dict(self) -> dict:
        return {
            "node_count": self.node_count,
            "normalization_hash": self.normalization_hash,
            "source_state_hash": self.source_state_hash,
            "values": list(self.values),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LiuStateTensorEncoder:
    """Flatten S=[chi,Upsilon,x,c,R] without losing canonical node order."""

    normalization: LiuStateNormalizationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.normalization, LiuStateNormalizationConfig):
            raise ValueError("normalization must be a LiuStateNormalizationConfig")

    @property
    def dimension(self) -> int:
        n = self.normalization.node_count
        return 1 + 4 * n + n * n

    def encode(self, state: LiuState) -> LiuStateTensorEncoding:
        if not isinstance(state, LiuState) or state.node_count != self.normalization.node_count:
            raise ValueError("state must match the configured node_count")
        cfg = self.normalization
        if state.spatial_profiles.region_width_km != cfg.region_width_km or state.spatial_profiles.region_height_km != cfg.region_height_km:
            raise ValueError("state region must match the reset-time normalization bounds")
        if state.transaction_size_bytes > cfg.max_transaction_size_bytes:
            raise ValueError("transaction size exceeds the configured normalization bound")
        if any(value > cfg.max_stake_tokens for value in state.stakes_tokens):
            raise ValueError("stake exceeds the configured normalization bound")
        if any(value > cfg.max_computational_capability_ghz for value in state.computational_capabilities_ghz):
            raise ValueError("capability exceeds the configured normalization bound")
        if any(
            rate is not None and rate > cfg.max_link_rate_mbps
            for row in state.link_state_matrix.rates_mbps
            for rate in row
        ):
            raise ValueError("link rate exceeds the configured normalization bound")
        values: list[float] = [state.transaction_size_bytes / cfg.max_transaction_size_bytes]
        values.extend(value / cfg.max_stake_tokens for value in state.stakes_tokens)
        values.extend(profile.x_km / cfg.region_width_km for profile in state.spatial_profiles.profiles)
        values.extend(profile.y_km / cfg.region_height_km for profile in state.spatial_profiles.profiles)
        values.extend(value / cfg.max_computational_capability_ghz for value in state.computational_capabilities_ghz)
        for sender_id, row in enumerate(state.link_state_matrix.rates_mbps):
            for receiver_id, rate in enumerate(row):
                values.append(cfg.diagonal_link_sentinel if sender_id == receiver_id else rate / cfg.max_link_rate_mbps)
        return LiuStateTensorEncoding(
            tuple(values),
            state.node_count,
            state.deterministic_hash(),
            cfg.deterministic_hash(),
        )


@dataclass(frozen=True, slots=True)
class LiuActionTensorEncoding(CanonicalSerializable):
    values: tuple[float, ...]
    node_count: int
    source_action_hash: str
    version: str = ACTION_ENCODING_VERSION

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        values = _finite_tuple(self.values, "values")
        expected = node_count + 5
        if len(values) != expected:
            raise ValueError(f"action encoding must contain exactly {expected} values for N={node_count}")
        if not isinstance(self.source_action_hash, str) or not self.source_action_hash:
            raise ValueError("source_action_hash must be a non-empty string")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "values", values)

    @property
    def dimension(self) -> int:
        return len(self.values)

    def as_tensor(self, *, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu") -> Tensor:
        return torch.tensor(self.values, dtype=dtype, device=device)

    def to_dict(self) -> dict:
        return {
            "node_count": self.node_count,
            "source_action_hash": self.source_action_hash,
            "values": list(self.values),
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LiuActionTensorEncoder:
    PROTOCOL_ORDER = (
        LiuConsensusProtocol.PBFT,
        LiuConsensusProtocol.ZYZZYVA,
        LiuConsensusProtocol.LIU_QUORUM,
    )

    node_count: int
    max_block_size_mb: float
    max_block_interval_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_count", require_integer(self.node_count, "node_count", minimum=1))
        object.__setattr__(self, "max_block_size_mb", require_finite_number(self.max_block_size_mb, "max_block_size_mb", positive=True))
        object.__setattr__(self, "max_block_interval_s", require_finite_number(self.max_block_interval_s, "max_block_interval_s", positive=True))

    @property
    def dimension(self) -> int:
        return self.node_count + 5

    def encode(self, action: LiuAction) -> LiuActionTensorEncoding:
        if not isinstance(action, LiuAction) or action.node_count != self.node_count:
            raise ValueError("action must match the configured node_count")
        if action.block_size_mb > self.max_block_size_mb or action.block_interval_s > self.max_block_interval_s:
            raise ValueError("action exceeds the configured normalization bounds")
        membership = [0.0] * self.node_count
        for node_id in action.validator_ids:
            membership[node_id] = 1.0
        one_hot = [1.0 if action.consensus_protocol is protocol else 0.0 for protocol in self.PROTOCOL_ORDER]
        values = tuple(
            membership
            + one_hot
            + [action.block_size_mb / self.max_block_size_mb, action.block_interval_s / self.max_block_interval_s]
        )
        return LiuActionTensorEncoding(values, self.node_count, action.deterministic_hash())


@dataclass(frozen=True, slots=True)
class LiuCandidateMaskingResult(CanonicalSerializable):
    candidates: tuple[LiuAction, ...]
    pre_execution_mask: tuple[bool, ...]
    screening_results: tuple[LiuActionScreeningResult, ...]
    generated_count: int
    screened_out_count: int
    evaluated_count: int
    policy_version: str = CANDIDATE_MASKING_POLICY_VERSION

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        mask = tuple(self.pre_execution_mask)
        screenings = tuple(self.screening_results)
        if len(candidates) != len(mask) or len(candidates) != len(screenings):
            raise ValueError("candidate, mask, and screening lengths must match")
        if any(not isinstance(value, bool) for value in mask):
            raise ValueError("pre_execution_mask must contain booleans")
        if self.generated_count != len(candidates):
            raise ValueError("generated_count must equal candidate count")
        if self.screened_out_count != sum(not value for value in mask):
            raise ValueError("screened_out_count does not match mask")
        if self.evaluated_count != sum(mask):
            raise ValueError("evaluated_count does not match mask")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "pre_execution_mask", mask)
        object.__setattr__(self, "screening_results", screenings)

    def to_dict(self) -> dict:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "evaluated_count": self.evaluated_count,
            "generated_count": self.generated_count,
            "policy_version": self.policy_version,
            "pre_execution_mask": list(self.pre_execution_mask),
            "screened_out_count": self.screened_out_count,
            "screening_results": [result.to_dict() for result in self.screening_results],
        }


class LiuCandidateActionMasker:
    """Mask only C1/C2 failures. C3 intentionally remains runtime-unknown."""

    def screen(
        self,
        generator: LiuActionCandidateGenerator,
        state: LiuState,
        candidates: Sequence[LiuAction],
    ) -> LiuCandidateMaskingResult:
        candidate_tuple = tuple(candidates)
        screenings = tuple(generator.screen(candidate, state) for candidate in candidate_tuple)
        mask = tuple(result.pre_execution_feasible for result in screenings)
        return LiuCandidateMaskingResult(
            candidate_tuple,
            mask,
            screenings,
            len(candidate_tuple),
            sum(not value for value in mask),
            sum(mask),
        )

    @staticmethod
    def apply_to_q_values(q_values: Tensor, mask: Sequence[bool]) -> Tensor:
        if q_values.ndim != 1 or q_values.shape[0] != len(mask):
            raise ValueError("q_values and mask must have matching one-dimensional shapes")
        mask_tensor = torch.tensor(tuple(mask), dtype=torch.bool, device=q_values.device)
        return q_values.masked_fill(~mask_tensor, float("-inf"))


@dataclass(frozen=True, slots=True)
class LiuCandidateQNetworkConfig(CanonicalSerializable):
    state_dimension: int
    action_dimension: int
    hidden_sizes: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    dtype: str = "float32"
    device: str = "cpu"
    initialization_seed: int = 0
    architecture_version: str = Q_NETWORK_ARCHITECTURE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_dimension", require_integer(self.state_dimension, "state_dimension", minimum=1))
        object.__setattr__(self, "action_dimension", require_integer(self.action_dimension, "action_dimension", minimum=1))
        hidden = tuple(require_integer(value, f"hidden_sizes[{index}]", minimum=1) for index, value in enumerate(self.hidden_sizes))
        if not hidden:
            raise ValueError("hidden_sizes must contain at least one layer")
        if self.activation not in {"relu", "tanh", "gelu"}:
            raise ValueError("activation must be relu, tanh, or gelu")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty PyTorch device string")
        object.__setattr__(self, "initialization_seed", require_integer(self.initialization_seed, "initialization_seed"))
        object.__setattr__(self, "hidden_sizes", hidden)

    def to_dict(self) -> dict:
        return {
            "action_dimension": self.action_dimension,
            "activation": self.activation,
            "architecture_version": self.architecture_version,
            "device": self.device,
            "dtype": self.dtype,
            "hidden_sizes": list(self.hidden_sizes),
            "initialization_seed": self.initialization_seed,
            "state_dimension": self.state_dimension,
        }


class LiuCandidateQNetwork(nn.Module):
    """Versioned MLP mapping concatenated state/action features to scalar Q."""

    def __init__(self, config: LiuCandidateQNetworkConfig) -> None:
        super().__init__()
        if not isinstance(config, LiuCandidateQNetworkConfig):
            raise ValueError("config must be a LiuCandidateQNetworkConfig")
        self.config = config
        activation_type = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[config.activation]
        dimensions = (config.state_dimension + config.action_dimension,) + config.hidden_sizes + (1,)
        layers: list[nn.Module] = []
        # fork_rng makes initialization repeatable without consuming the caller's torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.initialization_seed)
            for index in range(len(dimensions) - 1):
                layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
                if index < len(dimensions) - 2:
                    layers.append(activation_type())
        self.model = nn.Sequential(*layers)
        self.to(device=torch.device(config.device), dtype=getattr(torch, config.dtype))

    def forward(self, state_action: Tensor) -> Tensor:
        if state_action.shape[-1] != self.config.state_dimension + self.config.action_dimension:
            raise ValueError("last input dimension must equal state_dimension + action_dimension")
        return self.model(state_action).squeeze(-1)

    def score_candidates(self, state: Tensor, candidate_actions: Tensor) -> Tensor:
        if state.ndim != 1 or state.shape[0] != self.config.state_dimension:
            raise ValueError("state must be a one-dimensional configured state encoding")
        if candidate_actions.ndim != 2 or candidate_actions.shape[1] != self.config.action_dimension:
            raise ValueError("candidate_actions must be M x action_dimension")
        state = state.to(device=self.model[0].weight.device, dtype=self.model[0].weight.dtype)
        candidate_actions = candidate_actions.to(device=state.device, dtype=state.dtype)
        repeated_state = state.unsqueeze(0).expand(candidate_actions.shape[0], -1)
        return self.forward(torch.cat((repeated_state, candidate_actions), dim=1))


@dataclass(frozen=True, slots=True)
class LiuCandidateSelection(CanonicalSerializable):
    candidate_index: int
    action_hash: str
    selection_policy: str

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "candidate_index": self.candidate_index,
            "selection_policy": self.selection_policy,
        }


def select_greedy(candidates: Sequence[LiuAction], q_values: Tensor, mask: Sequence[bool]) -> LiuCandidateSelection:
    candidates = tuple(candidates)
    masked = LiuCandidateActionMasker.apply_to_q_values(q_values, mask)
    if not candidates or not any(mask):
        raise ValueError("greedy selection requires at least one feasible candidate")
    index = int(torch.argmax(masked).item())
    return LiuCandidateSelection(index, candidates[index].deterministic_hash(), "liu_greedy_candidate_selection_v1")


def select_uniform_candidate(candidates: Sequence[LiuAction], mask: Sequence[bool], rng) -> LiuCandidateSelection:
    candidates = tuple(candidates)
    feasible = tuple(index for index, allowed in enumerate(mask) if allowed)
    if not feasible:
        raise ValueError("uniform selection requires at least one feasible candidate")
    if rng is None or not callable(getattr(rng, "randrange", None)):
        raise ValueError("an externally supplied RNG with randrange is required")
    index = feasible[rng.randrange(len(feasible))]
    return LiuCandidateSelection(index, candidates[index].deterministic_hash(), "liu_uniform_candidate_selection_v1")


@dataclass(frozen=True, slots=True)
class LiuCandidateGenerationContext(CanonicalSerializable):
    state_hash: str
    candidate_seed: int
    candidates_per_step: int
    candidate_domain_hash: str
    generator_policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_seed", require_integer(self.candidate_seed, "candidate_seed"))
        object.__setattr__(self, "candidates_per_step", require_integer(self.candidates_per_step, "candidates_per_step", minimum=1))
        for field_name in ("state_hash", "candidate_domain_hash", "generator_policy_version"):
            if not isinstance(getattr(self, field_name), str) or not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")

    def to_dict(self) -> dict:
        return {
            "candidate_domain_hash": self.candidate_domain_hash,
            "candidate_seed": self.candidate_seed,
            "candidates_per_step": self.candidates_per_step,
            "generator_policy_version": self.generator_policy_version,
            "state_hash": self.state_hash,
        }


@dataclass(frozen=True, slots=True)
class LiuCandidateSetConfig(CanonicalSerializable):
    """Explicit candidate budget; there is intentionally no replication default."""

    candidates_per_step: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidates_per_step",
            require_integer(self.candidates_per_step, "candidates_per_step", minimum=1),
        )

    def to_dict(self) -> dict:
        return {"candidates_per_step": self.candidates_per_step}


def generate_candidate_actions(
    generator: LiuActionCandidateGenerator,
    state: LiuState,
    config: LiuCandidateSetConfig,
    *,
    candidate_seed: int,
) -> tuple[tuple[LiuAction, ...], LiuCandidateGenerationContext]:
    """Create a step-local candidate set using a dedicated reconstructible RNG."""
    if not isinstance(generator, LiuActionCandidateGenerator):
        raise ValueError("generator must be a LiuActionCandidateGenerator")
    if not isinstance(config, LiuCandidateSetConfig):
        raise ValueError("config must be a LiuCandidateSetConfig")
    seed = require_integer(candidate_seed, "candidate_seed")
    candidates = tuple(generator.iter_candidates(state, random.Random(seed), config.candidates_per_step))
    context = LiuCandidateGenerationContext(
        state.deterministic_hash(),
        seed,
        len(candidates),
        generator.domain.deterministic_hash(),
        generator.domain.validator_sampling_policy,
    )
    return candidates, context


def reconstruct_candidate_actions(
    context: LiuCandidateGenerationContext,
    generator: LiuActionCandidateGenerator,
    state: LiuState,
) -> tuple[LiuAction, ...]:
    if context.state_hash != state.deterministic_hash():
        raise ValueError("candidate context does not bind this state")
    if context.candidate_domain_hash != generator.domain.deterministic_hash():
        raise ValueError("candidate context does not bind this action domain")
    if context.generator_policy_version != generator.domain.validator_sampling_policy:
        raise ValueError("candidate context policy does not match the generator")
    return tuple(
        generator.iter_candidates(
            state,
            random.Random(context.candidate_seed),
            context.candidates_per_step,
        )
    )


@dataclass(frozen=True, slots=True)
class LiuReplayTransition(CanonicalSerializable):
    state_encoding: LiuStateTensorEncoding
    action_encoding: LiuActionTensorEncoding
    reward: float
    next_state_encoding: LiuStateTensorEncoding | None
    terminated: bool
    truncated: bool
    next_candidate_actions: tuple[LiuAction, ...]
    next_candidate_mask: tuple[bool, ...]
    next_candidate_context: LiuCandidateGenerationContext | None

    def __post_init__(self) -> None:
        if not isinstance(self.state_encoding, LiuStateTensorEncoding) or not isinstance(self.action_encoding, LiuActionTensorEncoding):
            raise ValueError("state_encoding and action_encoding must be immutable Liu encodings")
        object.__setattr__(self, "reward", require_finite_number(self.reward, "reward"))
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminated and truncated must be booleans")
        candidates = tuple(self.next_candidate_actions)
        mask = tuple(self.next_candidate_mask)
        if len(candidates) != len(mask) or any(not isinstance(value, bool) for value in mask):
            raise ValueError("next candidates and boolean mask must have matching lengths")
        if self.next_state_encoding is None and candidates:
            raise ValueError("next candidates require a next-state encoding")
        if self.next_state_encoding is not None and not self.terminated and self.next_candidate_context is None:
            raise ValueError("bootstrappable next state requires deterministic candidate context")
        if self.next_candidate_context is not None:
            if self.next_state_encoding is None:
                raise ValueError("candidate context requires a next-state encoding")
            if self.next_candidate_context.state_hash != self.next_state_encoding.source_state_hash:
                raise ValueError("candidate context must bind the next state")
            if self.next_candidate_context.candidates_per_step != len(candidates):
                raise ValueError("candidate context count must match stored next candidates")
        object.__setattr__(self, "next_candidate_actions", candidates)
        object.__setattr__(self, "next_candidate_mask", mask)

    def to_dict(self) -> dict:
        return {
            "action_encoding": self.action_encoding.to_dict(),
            "next_candidate_actions": [action.to_dict() for action in self.next_candidate_actions],
            "next_candidate_context": None if self.next_candidate_context is None else self.next_candidate_context.to_dict(),
            "next_candidate_mask": list(self.next_candidate_mask),
            "next_state_encoding": None if self.next_state_encoding is None else self.next_state_encoding.to_dict(),
            "reward": self.reward,
            "state_encoding": self.state_encoding.to_dict(),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


class LiuReplayBuffer:
    """Bounded insertion-ordered replay storage with externally controlled sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = require_integer(capacity, "capacity", minimum=1)
        self._items: list[LiuReplayTransition] = []

    def __len__(self) -> int:
        return len(self._items)

    def append(self, transition: LiuReplayTransition) -> None:
        if not isinstance(transition, LiuReplayTransition):
            raise ValueError("transition must be a LiuReplayTransition")
        if len(self._items) == self.capacity:
            self._items.pop(0)
        self._items.append(transition)

    def snapshot(self) -> tuple[LiuReplayTransition, ...]:
        return tuple(self._items)

    def sample(self, batch_size: int, rng) -> tuple[LiuReplayTransition, ...]:
        count = require_integer(batch_size, "batch_size", minimum=1)
        if count > len(self._items):
            raise ValueError("batch_size cannot exceed replay buffer length")
        if rng is None or not callable(getattr(rng, "sample", None)):
            raise ValueError("an externally supplied RNG with sample is required")
        return tuple(rng.sample(self._items, count))


@dataclass(frozen=True, slots=True)
class LiuBellmanTargetConfig(CanonicalSerializable):
    gamma: float
    bootstrap_on_truncation: bool = True
    policy_version: str = BELLMAN_TARGET_POLICY_VERSION

    def __post_init__(self) -> None:
        gamma = require_finite_number(self.gamma, "gamma", non_negative=True)
        if gamma > 1:
            raise ValueError("gamma must be in [0,1]")
        if not isinstance(self.bootstrap_on_truncation, bool):
            raise ValueError("bootstrap_on_truncation must be boolean")
        object.__setattr__(self, "gamma", gamma)

    def to_dict(self) -> dict:
        return {
            "bootstrap_on_truncation": self.bootstrap_on_truncation,
            "gamma": self.gamma,
            "policy_version": self.policy_version,
        }


def compute_bellman_targets(
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    next_candidate_q_values: Tensor,
    next_candidate_mask: Tensor,
    config: LiuBellmanTargetConfig,
) -> Tensor:
    """Pure candidate-set Bellman target; no gradients or optimizer mutation."""
    if rewards.ndim != 1 or terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("rewards, terminated, and truncated must be equal one-dimensional batches")
    if next_candidate_q_values.ndim != 2 or next_candidate_q_values.shape[0] != rewards.shape[0]:
        raise ValueError("next_candidate_q_values must have shape batch x candidates")
    if next_candidate_mask.shape != next_candidate_q_values.shape:
        raise ValueError("next_candidate_mask must match next candidate Q shape")
    if next_candidate_mask.dtype is not torch.bool:
        raise ValueError("next_candidate_mask must be boolean")
    masked = next_candidate_q_values.masked_fill(~next_candidate_mask, float("-inf"))
    has_candidate = next_candidate_mask.any(dim=1)
    max_next = masked.max(dim=1).values
    max_next = torch.where(has_candidate, max_next, torch.zeros_like(max_next))
    bootstrap = ~terminated.to(dtype=torch.bool)
    if not config.bootstrap_on_truncation:
        bootstrap &= ~truncated.to(dtype=torch.bool)
    return rewards + config.gamma * bootstrap.to(dtype=rewards.dtype) * max_next
