"""Generic immutable finite-state Markov channels for directed Liu links."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, ClassVar, Iterable, Protocol

from Liu.LinkState import LinkStateMatrix
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer


class RandomSource(Protocol):
    """Minimal external RNG API required by the FSMC transition."""

    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class LinkRateLevels(CanonicalSerializable):
    """Explicit, strictly increasing finite transmission-rate levels in Mbps."""

    rates_mbps: tuple[float, ...]

    def __post_init__(self) -> None:
        rates = tuple(
            require_finite_number(rate, f"rates_mbps[{index}]", positive=True)
            for index, rate in enumerate(self.rates_mbps)
        )
        if not rates:
            raise ValueError("at least one link-rate level is required")
        if len(set(rates)) != len(rates):
            raise ValueError("link-rate levels must be unique")
        if any(left >= right for left, right in zip(rates, rates[1:])):
            raise ValueError("link-rate levels must be strictly increasing")
        object.__setattr__(self, "rates_mbps", rates)

    @property
    def level_count(self) -> int:
        return len(self.rates_mbps)

    def index_for_rate(self, rate_mbps: float) -> int:
        rate = require_finite_number(rate_mbps, "rate_mbps", positive=True)
        try:
            return self.rates_mbps.index(rate)
        except ValueError as error:
            raise ValueError(f"rate {rate} Mbps is not one of the declared link-rate levels") from error

    def to_dict(self) -> dict[str, Any]:
        return {"rates_mbps": list(self.rates_mbps)}


@dataclass(frozen=True, slots=True)
class LinkTransitionMatrix(CanonicalSerializable):
    """One row-stochastic L-by-L transition matrix."""

    ROW_SUM_TOLERANCE: ClassVar[float] = 1e-12

    probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.probabilities)
        if not rows:
            raise ValueError("transition matrix cannot be empty")
        level_count = len(rows)
        if any(len(row) != level_count for row in rows):
            raise ValueError("transition matrix must be square LxL")

        normalized_rows = []
        for row_index, row in enumerate(rows):
            normalized = tuple(
                require_finite_number(probability, f"probabilities[{row_index}][{column_index}]", non_negative=True)
                for column_index, probability in enumerate(row)
            )
            if any(probability > 1.0 for probability in normalized):
                raise ValueError("transition probabilities cannot exceed 1")
            if not isclose(sum(normalized), 1.0, rel_tol=0.0, abs_tol=self.ROW_SUM_TOLERANCE):
                raise ValueError("each transition-probability row must sum to 1")
            normalized_rows.append(normalized)
        object.__setattr__(self, "probabilities", tuple(normalized_rows))

    @property
    def level_count(self) -> int:
        return len(self.probabilities)

    def sample_next_level(self, current_level: int, rng: RandomSource) -> int:
        current_level = require_integer(current_level, "current_level")
        if current_level >= self.level_count:
            raise ValueError("current_level is outside the transition matrix")
        if rng is None or not callable(getattr(rng, "random", None)):
            raise ValueError("an external RNG object with random() is required")
        draw = rng.random()
        if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not 0.0 <= draw < 1.0:
            raise ValueError("external RNG random() must return a value in [0, 1)")

        cumulative = 0.0
        for next_level, probability in enumerate(self.probabilities[current_level]):
            cumulative += probability
            if draw < cumulative:
                return next_level
        # Rows are validated to sum to one. This final return only absorbs
        # floating-point accumulation at the upper boundary.
        return self.level_count - 1

    def to_dict(self) -> dict[str, Any]:
        return {"probabilities": [list(row) for row in self.probabilities]}


@dataclass(frozen=True, slots=True)
class LinkTransitionTensor(CanonicalSerializable):
    """An explicit transition matrix for every directed off-diagonal link."""

    node_count: int
    matrices: tuple[tuple[LinkTransitionMatrix | None, ...], ...]

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        rows = tuple(tuple(row) for row in self.matrices)
        if len(rows) != node_count or any(len(row) != node_count for row in rows):
            raise ValueError(f"transition tensor must be an exact {node_count}x{node_count} link matrix")

        level_count: int | None = None
        for sender_id, row in enumerate(rows):
            for receiver_id, matrix in enumerate(row):
                if sender_id == receiver_id:
                    if matrix is not None:
                        raise ValueError("self-link transition matrices must be masked with None")
                    continue
                if not isinstance(matrix, LinkTransitionMatrix):
                    raise ValueError("every off-diagonal link requires an explicit LinkTransitionMatrix")
                if level_count is None:
                    level_count = matrix.level_count
                elif matrix.level_count != level_count:
                    raise ValueError("all link transition matrices must use the same number of levels")

        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "matrices", rows)

    @classmethod
    def shared(cls, node_count: int, matrix: LinkTransitionMatrix) -> "LinkTransitionTensor":
        """Use one documented shared model for every directed non-self link."""
        if not isinstance(matrix, LinkTransitionMatrix):
            raise ValueError("matrix must be a LinkTransitionMatrix")
        node_count = require_integer(node_count, "node_count", minimum=1)
        return cls(
            node_count,
            tuple(
                tuple(None if sender_id == receiver_id else matrix for receiver_id in range(node_count))
                for sender_id in range(node_count)
            ),
        )

    @property
    def level_count(self) -> int:
        for sender_id, row in enumerate(self.matrices):
            for receiver_id, matrix in enumerate(row):
                if sender_id != receiver_id:
                    return matrix.level_count
        raise ValueError("a one-node tensor has no directed off-diagonal transition model")

    def matrix_for(self, sender_id: int, receiver_id: int) -> LinkTransitionMatrix:
        sender_id = require_integer(sender_id, "sender_id")
        receiver_id = require_integer(receiver_id, "receiver_id")
        if sender_id >= self.node_count or receiver_id >= self.node_count:
            raise KeyError("link endpoint is outside the transition tensor")
        matrix = self.matrices[sender_id][receiver_id]
        if matrix is None:
            raise ValueError("self-links are masked and do not have a transition matrix")
        return matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrices": [
                [None if matrix is None else matrix.to_dict() for matrix in row]
                for row in self.matrices
            ],
            "node_count": self.node_count,
        }


@dataclass(frozen=True, slots=True)
class LinkFSMCState(CanonicalSerializable):
    """Current directed rates plus their explicit per-link transition model."""

    rate_levels: LinkRateLevels
    transition_tensor: LinkTransitionTensor
    current_links: LinkStateMatrix

    def __post_init__(self) -> None:
        if not isinstance(self.rate_levels, LinkRateLevels):
            raise ValueError("rate_levels must be LinkRateLevels")
        if not isinstance(self.transition_tensor, LinkTransitionTensor):
            raise ValueError("transition_tensor must be a LinkTransitionTensor")
        if not isinstance(self.current_links, LinkStateMatrix):
            raise ValueError("current_links must be a LinkStateMatrix")
        node_count = self.current_links.node_count
        if self.transition_tensor.node_count != node_count:
            raise ValueError("current link matrix and transition tensor must have the same N")
        if node_count > 1 and self.transition_tensor.level_count != self.rate_levels.level_count:
            raise ValueError("transition matrix dimensions must match the declared rate levels")
        for sender_id in range(node_count):
            for receiver_id in range(node_count):
                if sender_id != receiver_id:
                    self.rate_levels.index_for_rate(self.current_links.rate(sender_id, receiver_id))

    def transition(self, rng: RandomSource) -> LinkStateMatrix:
        """Return the next immutable rate matrix without mutating this state."""
        if rng is None or not callable(getattr(rng, "random", None)):
            raise ValueError("an external RNG object with random() is required")
        rows: list[tuple[float | None, ...]] = []
        for sender_id in range(self.current_links.node_count):
            row: list[float | None] = []
            for receiver_id in range(self.current_links.node_count):
                if sender_id == receiver_id:
                    row.append(None)
                    continue
                current_level = self.rate_levels.index_for_rate(self.current_links.rate(sender_id, receiver_id))
                next_level = self.transition_tensor.matrix_for(sender_id, receiver_id).sample_next_level(current_level, rng)
                row.append(self.rate_levels.rates_mbps[next_level])
            rows.append(tuple(row))
        return LinkStateMatrix(self.current_links.node_count, tuple(rows))

    def with_current_links(self, current_links: LinkStateMatrix) -> "LinkFSMCState":
        """Construct the next FSMC state while retaining levels and transition models."""
        return LinkFSMCState(self.rate_levels, self.transition_tensor, current_links)

    def advance(self, rng: RandomSource) -> "LinkFSMCState":
        return self.with_current_links(self.transition(rng))

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_links": self.current_links.to_dict(),
            "rate_levels": self.rate_levels.to_dict(),
            "transition_tensor": self.transition_tensor.to_dict(),
        }
