"""Gini coefficient used for the Liu decentralization constraint (Eq. 2)."""

from typing import Iterable


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
