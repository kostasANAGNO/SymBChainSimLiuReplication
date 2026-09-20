"""Immutable, ordered membership for the active static validator population."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ValidatorSet:
    """Single source of truth for validator membership and proposer ordering."""

    ids: tuple[int, ...]
    _id_lookup: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.ids:
            raise ValueError("Validator set cannot be empty")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Validator IDs must be unique")
        object.__setattr__(self, "_id_lookup", frozenset(self.ids))

    @classmethod
    def first_nodes(cls, total_nodes: int, validator_count: int | None = None) -> "ValidatorSet":
        """Select the first K node IDs deterministically; omitted K means K=N."""
        count = total_nodes if validator_count is None else validator_count
        if total_nodes <= 0:
            raise ValueError("Total node count must be positive")
        if count <= 0 or count > total_nodes:
            raise ValueError(f"validator_count must be between 1 and Nn ({total_nodes}), got {count}")
        return cls(tuple(range(count)))

    @property
    def count(self) -> int:
        return len(self.ids)

    def contains(self, node_id: int) -> bool:
        return node_id in self._id_lookup

    def proposer_for(self, selection_value: int) -> int:
        """Map existing round/hash arithmetic onto the ordered validator set."""
        return self.ids[selection_value % self.count]
