"""Directed pairwise transmission rates for the Liu state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer


@dataclass(frozen=True, slots=True)
class LinkStateMatrix(CanonicalSerializable):
    """An immutable N-by-N matrix in Mbps with ``None`` self-links."""

    node_count: int
    rates_mbps: tuple[tuple[float | None, ...], ...]

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        rows = tuple(tuple(row) for row in self.rates_mbps)
        if len(rows) != node_count or any(len(row) != node_count for row in rows):
            raise ValueError(f"rates_mbps must be an exact {node_count}x{node_count} matrix")

        normalized_rows: list[tuple[float | None, ...]] = []
        for sender_id, row in enumerate(rows):
            normalized_row: list[float | None] = []
            for receiver_id, value in enumerate(row):
                if sender_id == receiver_id:
                    if value is not None:
                        raise ValueError("self-links must be masked with None")
                    normalized_row.append(None)
                else:
                    if value is None:
                        raise ValueError("off-diagonal directed links must have an explicit Mbps rate")
                    normalized_row.append(
                        require_finite_number(value, f"rates_mbps[{sender_id}][{receiver_id}]", positive=True)
                    )
            normalized_rows.append(tuple(normalized_row))

        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "rates_mbps", tuple(normalized_rows))

    @classmethod
    def from_rows(cls, rates_mbps: Iterable[Iterable[float | None]]) -> "LinkStateMatrix":
        rows = tuple(tuple(row) for row in rates_mbps)
        return cls(len(rows), rows)

    def rate(self, sender_id: int, receiver_id: int) -> float:
        require_integer(sender_id, "sender_id")
        require_integer(receiver_id, "receiver_id")
        if sender_id >= self.node_count or receiver_id >= self.node_count:
            raise KeyError("link endpoint is outside the matrix")
        value = self.rates_mbps[sender_id][receiver_id]
        if value is None:
            raise ValueError("self-links are masked and do not have a transmission rate")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {"node_count": self.node_count, "rates_mbps": [list(row) for row in self.rates_mbps]}
