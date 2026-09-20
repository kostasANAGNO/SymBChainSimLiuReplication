"""Lazy deterministic Liu action candidates and DES-independent screening."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil, comb
from time import perf_counter
from typing import Iterator

from Liu.Action import LiuAction, require_block_interval_s, require_block_size_mb
from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.SpatialIntensity import GridSpatialIntensityModel
from Liu.State import LiuState
from Liu.Threat import ThreatScenario
from Liu.Validation import require_finite_number, require_integer
from Utils.DecentralizationMetrics import canonical_pairwise_gini


class LiuValidatorSubsetCandidatePolicy(ABC):
    """Pluggable, non-materializing validator-subset reconstruction policy."""

    @property
    @abstractmethod
    def version(self) -> str:
        pass

    @abstractmethod
    def sample(self, node_count: int, validator_count: int, rng) -> tuple[int, ...]:
        pass


class LiuSeededValidatorSubsetSamplingPolicy(LiuValidatorSubsetCandidatePolicy):
    @property
    def version(self) -> str:
        return "liu_seeded_validator_subset_sampling_v1"

    def sample(self, node_count: int, validator_count: int, rng) -> tuple[int, ...]:
        return tuple(sorted(rng.sample(range(node_count), validator_count)))


def _require_policy_population(node_count: int, validator_count: int, expected_node_count: int) -> None:
    node_count = require_integer(node_count, "node_count", minimum=1)
    validator_count = require_integer(validator_count, "validator_count", minimum=1)
    if node_count != expected_node_count:
        raise ValueError("candidate population does not match the policy profile population")
    if validator_count > node_count:
        raise ValueError("validator_count K cannot exceed node_count N")


def _grid_cell_populations(model: GridSpatialIntensityModel) -> tuple[tuple[int, ...], ...]:
    populations: list[list[int]] = [[] for _ in range(model.grid_rows * model.grid_columns)]
    cell_width = model.spatial_profiles.region_width_km / model.grid_columns
    cell_height = model.spatial_profiles.region_height_km / model.grid_rows
    for profile in model.spatial_profiles.profiles:
        column = min(int(profile.x_km / cell_width), model.grid_columns - 1)
        row = min(int(profile.y_km / cell_height), model.grid_rows - 1)
        populations[row * model.grid_columns + column].append(profile.node_id)
    return tuple(tuple(population) for population in populations)


def _balanced_cell_quotas(
    populations: tuple[tuple[int, ...], ...], validator_count: int, rng
) -> tuple[int, ...]:
    """Randomized round-robin quotas; it proposes balance but proves no constraint."""
    quotas = [0] * len(populations)
    remaining = validator_count
    while remaining:
        eligible = tuple(
            index for index, population in enumerate(populations) if quotas[index] < len(population)
        )
        if not eligible:
            raise ValueError("spatial populations cannot supply K validators")
        minimum = min(quotas[index] for index in eligible)
        tier = tuple(index for index in eligible if quotas[index] == minimum)
        for index in rng.sample(tier, len(tier)):
            if remaining == 0:
                break
            quotas[index] += 1
            remaining -= 1
    return tuple(quotas)


@dataclass(frozen=True, slots=True)
class LiuStakeBalancedValidatorSamplingPolicy(LiuValidatorSubsetCandidatePolicy):
    """Choose the lowest-Gini proposal from bounded randomized stake-rank windows."""

    stakes_tokens: tuple[float, ...]
    proposal_trials: int = 32
    rank_window_multiplier: float = 2.0

    def __post_init__(self) -> None:
        stakes = tuple(
            require_finite_number(value, f"stakes_tokens[{index}]", non_negative=True)
            for index, value in enumerate(self.stakes_tokens)
        )
        if not stakes or sum(stakes) == 0:
            raise ValueError("stakes_tokens must be non-empty and not all zero")
        object.__setattr__(self, "stakes_tokens", stakes)
        object.__setattr__(self, "proposal_trials", require_integer(self.proposal_trials, "proposal_trials", minimum=1))
        object.__setattr__(
            self,
            "rank_window_multiplier",
            require_finite_number(self.rank_window_multiplier, "rank_window_multiplier", positive=True),
        )

    @property
    def version(self) -> str:
        return "liu_stake_balanced_validator_sampling_v1"

    def sample(self, node_count: int, validator_count: int, rng) -> tuple[int, ...]:
        _require_policy_population(node_count, validator_count, len(self.stakes_tokens))
        ranked = tuple(sorted(range(node_count), key=lambda node_id: (self.stakes_tokens[node_id], node_id)))
        window = min(node_count, max(validator_count, ceil(validator_count * self.rank_window_multiplier)))
        proposals = []
        for _ in range(self.proposal_trials):
            start = rng.randrange(node_count - window + 1)
            selected = tuple(sorted(rng.sample(ranked[start : start + window], validator_count)))
            score = canonical_pairwise_gini(self.stakes_tokens[node_id] for node_id in selected)
            proposals.append((score, selected))
        return min(proposals)[1]


@dataclass(frozen=True, slots=True)
class LiuGeographicBalancedValidatorSamplingPolicy(LiuValidatorSubsetCandidatePolicy):
    """Allocate K across equal-area grid cells before sampling nodes in each cell."""

    geographic_model: GridSpatialIntensityModel

    def __post_init__(self) -> None:
        if not isinstance(self.geographic_model, GridSpatialIntensityModel):
            raise ValueError("geographic_model must be a GridSpatialIntensityModel")

    @property
    def version(self) -> str:
        return "liu_geographic_balanced_validator_sampling_v1"

    def sample(self, node_count: int, validator_count: int, rng) -> tuple[int, ...]:
        _require_policy_population(
            node_count, validator_count, self.geographic_model.spatial_profiles.node_count
        )
        populations = _grid_cell_populations(self.geographic_model)
        quotas = _balanced_cell_quotas(populations, validator_count, rng)
        selected = []
        for population, quota in zip(populations, quotas):
            selected.extend(rng.sample(population, quota))
        return tuple(sorted(selected))


@dataclass(frozen=True, slots=True)
class LiuJointDecentralizationValidatorSamplingPolicy(LiuValidatorSubsetCandidatePolicy):
    """Grid-balanced stake targets with a bounded diversity branch."""

    stakes_tokens: tuple[float, ...]
    geographic_model: GridSpatialIntensityModel
    proposal_trials: int = 1
    shortlist_extra: int = 0
    diversity_proposal_period: int = 5
    diversity_shortlist_extra: int = 2

    def __post_init__(self) -> None:
        stakes = tuple(
            require_finite_number(value, f"stakes_tokens[{index}]", non_negative=True)
            for index, value in enumerate(self.stakes_tokens)
        )
        if not stakes or sum(stakes) == 0:
            raise ValueError("stakes_tokens must be non-empty and not all zero")
        if not isinstance(self.geographic_model, GridSpatialIntensityModel):
            raise ValueError("geographic_model must be a GridSpatialIntensityModel")
        if len(stakes) != self.geographic_model.spatial_profiles.node_count:
            raise ValueError("stake and spatial populations must have the same N")
        object.__setattr__(self, "stakes_tokens", stakes)
        object.__setattr__(self, "proposal_trials", require_integer(self.proposal_trials, "proposal_trials", minimum=1))
        object.__setattr__(self, "shortlist_extra", require_integer(self.shortlist_extra, "shortlist_extra", minimum=0))
        object.__setattr__(
            self,
            "diversity_proposal_period",
            require_integer(self.diversity_proposal_period, "diversity_proposal_period", minimum=1),
        )
        object.__setattr__(
            self,
            "diversity_shortlist_extra",
            require_integer(self.diversity_shortlist_extra, "diversity_shortlist_extra", minimum=0),
        )

    @property
    def version(self) -> str:
        return "liu_joint_decentralization_validator_sampling_v1"

    def sample(self, node_count: int, validator_count: int, rng) -> tuple[int, ...]:
        _require_policy_population(node_count, validator_count, len(self.stakes_tokens))
        populations = _grid_cell_populations(self.geographic_model)
        ranked_stakes = tuple(sorted(self.stakes_tokens))
        lower = node_count // 3
        upper = max(lower + 1, node_count - lower)
        proposals = []
        for _ in range(self.proposal_trials):
            diversity_branch = rng.randrange(self.diversity_proposal_period) == 0
            if diversity_branch:
                target_stake = ranked_stakes[rng.randrange(node_count)]
                shortlist_extra = self.diversity_shortlist_extra
            else:
                target_stake = ranked_stakes[lower + rng.randrange(upper - lower)]
                shortlist_extra = self.shortlist_extra
            quotas = _balanced_cell_quotas(populations, validator_count, rng)
            selected = []
            for population, quota in zip(populations, quotas):
                ranked_cell = tuple(
                    sorted(
                        population,
                        key=lambda node_id: (abs(self.stakes_tokens[node_id] - target_stake), node_id),
                    )
                )
                shortlist_size = min(len(ranked_cell), quota + shortlist_extra)
                selected.extend(rng.sample(ranked_cell[:shortlist_size], quota))
            candidate = tuple(sorted(selected))
            stake_score = canonical_pairwise_gini(self.stakes_tokens[node_id] for node_id in candidate)
            geographic_score = self.geographic_model.evaluate(candidate).geographic_gini
            proposals.append(((stake_score, geographic_score), candidate))
        return min(proposals)[1]


class LiuJointDecentralizationValidatorSamplingPolicyV2(
    LiuJointDecentralizationValidatorSamplingPolicy
):
    """The v1 stochastic proposal distribution paired with bounded batch repair."""

    @property
    def version(self) -> str:
        return "liu_joint_decentralization_validator_sampling_v2"


@dataclass(frozen=True, slots=True)
class LiuActionCandidateDomain(CanonicalSerializable):
    node_count: int
    validator_count: int
    protocols: tuple[LiuConsensusProtocol, ...]
    block_sizes_mb: tuple[float, ...]
    block_intervals_s: tuple[float, ...]
    validator_sampling_policy: str = "liu_seeded_validator_subset_sampling_v1"

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        validator_count = require_integer(self.validator_count, "validator_count", minimum=1)
        if validator_count > node_count:
            raise ValueError("validator_count K cannot exceed node_count N")
        protocols = tuple(self.protocols)
        if not protocols or not all(isinstance(item, LiuConsensusProtocol) for item in protocols):
            raise ValueError("protocols must contain LiuConsensusProtocol values")
        if len(set(protocols)) != len(protocols):
            raise ValueError("protocol domain values must be unique")
        sizes = tuple(require_block_size_mb(value) for value in self.block_sizes_mb)
        intervals = tuple(require_block_interval_s(value) for value in self.block_intervals_s)
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("block_sizes_mb must be non-empty and unique")
        if not intervals or len(set(intervals)) != len(intervals):
            raise ValueError("block_intervals_s must be non-empty and unique")
        if not isinstance(self.validator_sampling_policy, str) or not self.validator_sampling_policy:
            raise ValueError("validator_sampling_policy must be a non-empty string")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "validator_count", validator_count)
        object.__setattr__(self, "protocols", protocols)
        object.__setattr__(self, "block_sizes_mb", sizes)
        object.__setattr__(self, "block_intervals_s", intervals)

    @property
    def action_space_size(self) -> int:
        return (
            comb(self.node_count, self.validator_count)
            * len(self.protocols)
            * len(self.block_sizes_mb)
            * len(self.block_intervals_s)
        )

    def to_dict(self) -> dict:
        return {
            "action_space_size": self.action_space_size,
            "block_intervals_s": list(self.block_intervals_s),
            "block_sizes_mb": list(self.block_sizes_mb),
            "node_count": self.node_count,
            "protocols": [protocol.value for protocol in self.protocols],
            "validator_count": self.validator_count,
            "validator_sampling_policy": self.validator_sampling_policy,
        }


@dataclass(frozen=True, slots=True)
class LiuActionScreeningResult(CanonicalSerializable):
    action_hash: str
    stake_gini: float | None
    geographic_gini: float
    selected_malicious_count: int
    tolerated_fault_count: int
    stake_constraint_passed: bool
    geographic_constraint_passed: bool
    c1_passed: bool
    c2_passed: bool
    pre_execution_feasible: bool
    c3_status: str
    policy_versions: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "c1_passed": self.c1_passed,
            "c2_passed": self.c2_passed,
            "c3_status": self.c3_status,
            "geographic_constraint_passed": self.geographic_constraint_passed,
            "geographic_gini": self.geographic_gini,
            "policy_versions": [{"name": name, "version": version} for name, version in self.policy_versions],
            "pre_execution_feasible": self.pre_execution_feasible,
            "selected_malicious_count": self.selected_malicious_count,
            "stake_constraint_passed": self.stake_constraint_passed,
            "stake_gini": self.stake_gini,
            "tolerated_fault_count": self.tolerated_fault_count,
        }


class LiuActionCandidateGenerator:
    """Sample a bounded number of actions without materializing C(N,K)."""

    SCREENING_POLICY = "liu_c1_c2_pre_execution_screening_v1"
    C3_STATUS = "runtime_required"

    def __init__(
        self,
        domain: LiuActionCandidateDomain,
        threat_scenario: ThreatScenario,
        geographic_model: GridSpatialIntensityModel,
        *,
        stake_gini_threshold: float,
        geographic_gini_threshold: float,
        validator_subset_policy: LiuValidatorSubsetCandidatePolicy | None = None,
    ) -> None:
        if not isinstance(domain, LiuActionCandidateDomain):
            raise ValueError("domain must be a LiuActionCandidateDomain")
        if not isinstance(threat_scenario, ThreatScenario) or threat_scenario.node_count != domain.node_count:
            raise ValueError("threat_scenario must match the candidate node population")
        if not isinstance(geographic_model, GridSpatialIntensityModel):
            raise ValueError("geographic_model must be a GridSpatialIntensityModel")
        if geographic_model.spatial_profiles.node_count != domain.node_count:
            raise ValueError("geographic model must match the candidate node population")
        stake_threshold = require_finite_number(
            stake_gini_threshold, "stake_gini_threshold", non_negative=True
        )
        geographic_threshold = require_finite_number(
            geographic_gini_threshold, "geographic_gini_threshold", non_negative=True
        )
        if stake_threshold > 1 or geographic_threshold > 1:
            raise ValueError("Gini thresholds must be in [0,1]")
        self.domain = domain
        self.threat_scenario = threat_scenario
        self.geographic_model = geographic_model
        self.stake_gini_threshold = stake_threshold
        self.geographic_gini_threshold = geographic_threshold
        self.validator_subset_policy = validator_subset_policy or LiuSeededValidatorSubsetSamplingPolicy()
        if not isinstance(self.validator_subset_policy, LiuValidatorSubsetCandidatePolicy):
            raise ValueError("validator_subset_policy must implement LiuValidatorSubsetCandidatePolicy")
        if self.validator_subset_policy.version != domain.validator_sampling_policy:
            raise ValueError("candidate domain and validator-subset policy versions must match")

    def _validate_state(self, state: LiuState) -> None:
        if not isinstance(state, LiuState) or state.node_count != self.domain.node_count:
            raise ValueError("state must be a LiuState matching the candidate domain N")
        if state.spatial_profiles != self.geographic_model.spatial_profiles:
            raise ValueError("candidate geography must use the state's spatial profiles")

    @staticmethod
    def _require_rng(rng) -> None:
        if rng is None or not callable(getattr(rng, "sample", None)) or not callable(getattr(rng, "randrange", None)):
            raise ValueError("an externally supplied RNG with sample/randrange is required")

    def iter_candidates(self, state: LiuState, rng, requested_candidate_count: int) -> Iterator[LiuAction]:
        """Lazily yield at most the requested unique canonical actions."""
        self._validate_state(state)
        self._require_rng(rng)
        requested = require_integer(requested_candidate_count, "requested_candidate_count", minimum=1)
        target = min(requested, self.domain.action_space_size)

        def generate() -> Iterator[LiuAction]:
            seen: set[str] = set()
            attempts = 0
            max_attempts = max(1_000, target * 100)
            while len(seen) < target and attempts < max_attempts:
                attempts += 1
                validator_ids = self.validator_subset_policy.sample(
                    self.domain.node_count,
                    self.domain.validator_count,
                    rng,
                )
                protocol = self.domain.protocols[rng.randrange(len(self.domain.protocols))]
                size = self.domain.block_sizes_mb[rng.randrange(len(self.domain.block_sizes_mb))]
                interval = self.domain.block_intervals_s[rng.randrange(len(self.domain.block_intervals_s))]
                action = LiuAction(
                    self.domain.node_count,
                    self.domain.validator_count,
                    validator_ids,
                    protocol,
                    size,
                    interval,
                )
                action_hash = action.deterministic_hash()
                if action_hash in seen:
                    continue
                seen.add(action_hash)
                yield action
            if len(seen) != target:
                raise RuntimeError("candidate sampler could not produce the requested unique action count")

        return generate()

    def sample_candidate(self, state: LiuState, rng) -> LiuAction:
        return next(self.iter_candidates(state, rng, 1))

    @staticmethod
    def tolerated_faults(action: LiuAction) -> int:
        if action.consensus_protocol is LiuConsensusProtocol.LIU_QUORUM:
            return 0
        return (action.validator_count - 1) // 3

    def screen(self, action: LiuAction, state: LiuState) -> LiuActionScreeningResult:
        self._validate_state(state)
        if not isinstance(action, LiuAction):
            raise ValueError("action must be a LiuAction")
        if action.node_count != self.domain.node_count or action.validator_count != self.domain.validator_count:
            raise ValueError("action N/K must match the candidate domain")
        if (
            action.consensus_protocol not in self.domain.protocols
            or action.block_size_mb not in self.domain.block_sizes_mb
            or action.block_interval_s not in self.domain.block_intervals_s
        ):
            raise ValueError("action is outside the configured candidate domain")
        stakes = tuple(state.stakes_tokens[node_id] for node_id in action.validator_ids)
        try:
            stake_gini = canonical_pairwise_gini(stakes)
            stake_passed = stake_gini <= self.stake_gini_threshold
        except ValueError:
            stake_gini = None
            stake_passed = False
        geographic = self.geographic_model.evaluate(action.validator_ids).geographic_gini
        geographic_passed = geographic <= self.geographic_gini_threshold
        malicious = self.threat_scenario.malicious_validator_count(action.validator_ids)
        tolerated = self.tolerated_faults(action)
        c1_passed = stake_passed and geographic_passed
        c2_passed = malicious <= tolerated
        return LiuActionScreeningResult(
            action.deterministic_hash(),
            stake_gini,
            geographic,
            malicious,
            tolerated,
            stake_passed,
            geographic_passed,
            c1_passed,
            c2_passed,
            c1_passed and c2_passed,
            self.C3_STATUS,
            (
                ("candidate_generation", self.domain.validator_sampling_policy),
                ("screening", self.SCREENING_POLICY),
                ("geography", self.geographic_model.ESTIMATOR_VERSION),
            ),
        )


class LiuCandidateViabilityError(RuntimeError):
    """A bounded v2 repair could not establish an authoritative C1/C2 candidate."""


@dataclass(frozen=True, slots=True)
class LiuCandidateRepairDiagnostic(CanonicalSerializable):
    invoked: bool
    succeeded: bool
    replaced_candidate_index: int | None
    before_stake_gini: float | None
    after_stake_gini: float | None
    before_geographic_gini: float | None
    after_geographic_gini: float | None
    validator_swaps: tuple[tuple[int, int], ...]
    repair_iterations: int
    start_candidates_examined: int
    repaired_validator_ids: tuple[int, ...] = ()
    repaired_action_hash: str | None = None
    policy_version: str = "liu_joint_decentralization_validator_sampling_v2"

    def to_dict(self) -> dict:
        return {
            "after_geographic_gini": self.after_geographic_gini,
            "after_stake_gini": self.after_stake_gini,
            "before_geographic_gini": self.before_geographic_gini,
            "before_stake_gini": self.before_stake_gini,
            "invoked": self.invoked,
            "policy_version": self.policy_version,
            "repair_iterations": self.repair_iterations,
            "repaired_validator_ids": list(self.repaired_validator_ids),
            "repaired_action_hash": self.repaired_action_hash,
            "replaced_candidate_index": self.replaced_candidate_index,
            "start_candidates_examined": self.start_candidates_examined,
            "succeeded": self.succeeded,
            "validator_swaps": [list(value) for value in self.validator_swaps],
        }


class LiuViabilityGuaranteedActionCandidateGenerator(LiuActionCandidateGenerator):
    """V1-equivalent proposals plus bounded, learning-independent C1/C2 repair.

    The name describes a tested viability policy rather than a proof over the
    combinatorial action space. Failure after the bounded search is explicit.
    """

    POLICY_VERSION = "liu_joint_decentralization_validator_sampling_v2"

    def __init__(self, *args, max_repair_iterations: int = 64, max_start_candidates: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        if self.domain.validator_sampling_policy != self.POLICY_VERSION:
            raise ValueError("v2 viability generator requires the v2 validator policy")
        self.max_repair_iterations = require_integer(
            max_repair_iterations, "max_repair_iterations", minimum=1
        )
        self.max_start_candidates = require_integer(
            max_start_candidates, "max_start_candidates", minimum=1
        )
        self.last_repair_diagnostic = LiuCandidateRepairDiagnostic(
            False, False, None, None, None, None, None, (), 0, 0
        )
        self.last_repair_wall_s = 0.0

    @staticmethod
    def _violation_objective(result: LiuActionScreeningResult, action: LiuAction):
        stake_violation = float("inf") if result.stake_gini is None else max(0.0, result.stake_gini)
        geographic_violation = max(0.0, result.geographic_gini)
        c2_violation = max(0, result.selected_malicious_count - result.tolerated_fault_count)
        return (
            not result.stake_constraint_passed,
            stake_violation,
            not result.geographic_constraint_passed,
            geographic_violation,
            c2_violation,
            action.deterministic_hash(),
        )

    def _rank_objective(self, result: LiuActionScreeningResult, action: LiuAction):
        stake_violation = (
            float("inf") if result.stake_gini is None
            else max(0.0, result.stake_gini - self.stake_gini_threshold)
        )
        geographic_violation = max(
            0.0, result.geographic_gini - self.geographic_gini_threshold
        )
        c2_violation = max(0, result.selected_malicious_count - result.tolerated_fault_count)
        return (
            stake_violation,
            geographic_violation,
            c2_violation,
            action.deterministic_hash(),
        )

    def _repair(self, action: LiuAction, state: LiuState):
        current = action
        current_result = self.screen(current, state)
        before = current_result
        swaps: list[tuple[int, int]] = []
        all_ids = tuple(range(self.domain.node_count))
        for _ in range(self.max_repair_iterations):
            if current_result.pre_execution_feasible:
                return current, before, current_result, tuple(swaps)
            selected = set(current.validator_ids)
            current_objective = self._rank_objective(current_result, current)
            best = None
            for removed in current.validator_ids:
                for added in all_ids:
                    if added in selected:
                        continue
                    validator_ids = tuple(sorted((selected - {removed}) | {added}))
                    proposal = LiuAction(
                        current.node_count,
                        current.validator_count,
                        validator_ids,
                        current.consensus_protocol,
                        current.block_size_mb,
                        current.block_interval_s,
                    )
                    result = self.screen(proposal, state)
                    if current_result.geographic_constraint_passed and not result.geographic_constraint_passed:
                        continue
                    objective = self._rank_objective(result, proposal)
                    candidate = (objective, removed, added, proposal, result)
                    if objective < current_objective and (best is None or candidate[:3] < best[:3]):
                        best = candidate
            if best is None:
                break
            _, removed, added, current, current_result = best
            swaps.append((removed, added))
        if current_result.pre_execution_feasible:
            return current, before, current_result, tuple(swaps)
        return None, before, current_result, tuple(swaps)

    def iter_candidates(self, state: LiuState, rng, requested_candidate_count: int) -> Iterator[LiuAction]:
        candidates = tuple(super().iter_candidates(state, rng, requested_candidate_count))
        screenings = tuple(self.screen(candidate, state) for candidate in candidates)
        if any(result.pre_execution_feasible for result in screenings):
            self.last_repair_wall_s = 0.0
            self.last_repair_diagnostic = LiuCandidateRepairDiagnostic(
                False, True, None, None, None, None, None, (), 0, 0
            )
            return iter(candidates)

        ranked = sorted(
            range(len(candidates)),
            key=lambda index: self._rank_objective(screenings[index], candidates[index]),
        )
        examined = 0
        repair_started = perf_counter()
        last_before = last_after = None
        last_swaps: tuple[tuple[int, int], ...] = ()
        for index in ranked[: min(self.max_start_candidates, len(ranked))]:
            examined += 1
            repaired, before, after, swaps = self._repair(candidates[index], state)
            last_before, last_after, last_swaps = before, after, swaps
            if repaired is None:
                continue
            authoritative = self.screen(repaired, state)
            if not authoritative.pre_execution_feasible:
                raise AssertionError("v2 repair bypassed authoritative screening")
            result = list(candidates)
            result[index] = repaired
            self.last_repair_diagnostic = LiuCandidateRepairDiagnostic(
                True, True, index, before.stake_gini, authoritative.stake_gini,
                before.geographic_gini, authoritative.geographic_gini,
                swaps, len(swaps), examined, repaired.validator_ids, repaired.deterministic_hash(),
            )
            self.last_repair_wall_s = perf_counter() - repair_started
            return iter(tuple(result))

        self.last_repair_diagnostic = LiuCandidateRepairDiagnostic(
            True, False, None,
            None if last_before is None else last_before.stake_gini,
            None if last_after is None else last_after.stake_gini,
            None if last_before is None else last_before.geographic_gini,
            None if last_after is None else last_after.geographic_gini,
            last_swaps, len(last_swaps), examined,
        )
        self.last_repair_wall_s = perf_counter() - repair_started
        raise LiuCandidateViabilityError(
            "bounded v2 repair found no authoritative C1/C2-feasible validator subset"
        )
