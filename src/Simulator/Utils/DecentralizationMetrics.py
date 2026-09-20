"""Profile-aware research metrics kept separate from legacy simulator metrics."""

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable, TYPE_CHECKING

from Chain.NodeProfile import NodeProfileSet
from Chain.ValidatorSet import ValidatorSet

if TYPE_CHECKING:
    from Utils.InstrumentationMetrics import InstrumentationMetricsResult


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    population_node_ids: tuple[int, ...]
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class CapabilitySummary:
    population_node_ids: tuple[int, ...]
    configured_count: int
    mean_ghz: float | None
    median_ghz: float | None
    minimum_ghz: float | None
    maximum_ghz: float | None
    standard_deviation_ghz: float | None
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DecentralizationMetricsResult:
    producer_gini: MetricValue
    stake_gini: MetricValue
    geographic_gini: MetricValue
    validator_capability_summary: CapabilitySummary
    all_node_capability_summary: CapabilitySummary


def canonical_pairwise_gini(values: Iterable[float]) -> float:
    """Return G = sum_i sum_j |x_i-x_j| / (2*n*sum_i x_i)."""
    population = tuple(float(value) for value in values)
    if not population:
        raise ValueError("Gini coefficient requires a non-empty population")
    if any(value < 0 for value in population):
        raise ValueError("Gini coefficient requires non-negative values")
    total = sum(population)
    if total == 0:
        raise ValueError("Gini coefficient is undefined for an all-zero population")
    pairwise_difference = sum(abs(left - right) for left in population for right in population)
    return pairwise_difference / (2 * len(population) * total)


class DecentralizationMetrics:
    """Aggregate producer/stake Gini and static capability summaries."""

    NO_SPATIAL_MODEL_REASON = "No explicit spatial intensity model is configured"

    @classmethod
    def from_instrumentation(
        cls,
        profile_set: NodeProfileSet,
        validator_set: ValidatorSet,
        metrics: "InstrumentationMetricsResult",
    ) -> DecentralizationMetricsResult:
        return cls.calculate(
            profile_set,
            validator_set,
            (block.proposer for block in metrics.finalized_blocks),
        )

    @classmethod
    def calculate(
        cls,
        profile_set: NodeProfileSet,
        validator_set: ValidatorSet,
        finalized_block_proposers: Iterable[int],
    ) -> DecentralizationMetricsResult:
        validator_ids = validator_set.ids
        producer_counts = {node_id: 0.0 for node_id in validator_ids}
        for proposer in finalized_block_proposers:
            if proposer not in producer_counts:
                raise ValueError(f"Finalized block proposer {proposer} is not a validator")
            producer_counts[proposer] += 1

        if sum(producer_counts.values()) == 0:
            producer_gini = MetricValue(None, validator_ids, "No quorum-finalized blocks are available")
        else:
            producer_gini = MetricValue(canonical_pairwise_gini(producer_counts.values()), validator_ids)

        stakes = tuple(profile_set.profile_for(node_id).stake_tokens for node_id in validator_ids)
        if any(stake is None for stake in stakes):
            stake_gini = MetricValue(None, validator_ids, "Stake configuration is unavailable for one or more validators")
        elif sum(stakes) == 0:
            stake_gini = MetricValue(None, validator_ids, "Validator stake population is all zero")
        else:
            stake_gini = MetricValue(canonical_pairwise_gini(stakes), validator_ids)

        all_node_ids = tuple(range(profile_set.count))
        return DecentralizationMetricsResult(
            producer_gini=producer_gini,
            stake_gini=stake_gini,
            geographic_gini=MetricValue(None, validator_ids, cls.NO_SPATIAL_MODEL_REASON),
            validator_capability_summary=cls._capability_summary(profile_set, validator_ids),
            all_node_capability_summary=cls._capability_summary(profile_set, all_node_ids),
        )

    @staticmethod
    def _capability_summary(profile_set: NodeProfileSet, node_ids: tuple[int, ...]) -> CapabilitySummary:
        capabilities = tuple(
            profile_set.profile_for(node_id).computational_capability_ghz
            for node_id in node_ids
            if profile_set.profile_for(node_id).computational_capability_ghz is not None
        )
        if not capabilities:
            return CapabilitySummary(node_ids, 0, None, None, None, None, None, "Computational capability configuration is unavailable")
        return CapabilitySummary(
            node_ids,
            len(capabilities),
            mean(capabilities),
            median(capabilities),
            min(capabilities),
            max(capabilities),
            pstdev(capabilities),
        )
