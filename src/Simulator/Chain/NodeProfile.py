"""Immutable, ordered static research profiles for simulation nodes."""

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class NodeProfile:
    """Static research attributes for one node.

    Location is deliberately not duplicated here; the network model remains
    the single owner of geographical placement.
    """

    node_id: int
    stake_tokens: float | None
    computational_capability_ghz: float | None


@dataclass(frozen=True, slots=True)
class NodeProfileSet:
    """Single source of truth for all node profiles in node-ID order."""

    profiles: tuple[NodeProfile, ...]

    def __post_init__(self) -> None:
        expected_ids = tuple(range(len(self.profiles)))
        actual_ids = tuple(profile.node_id for profile in self.profiles)
        if actual_ids != expected_ids:
            raise ValueError(f"Node profiles must be ordered and complete for IDs {expected_ids}, got {actual_ids}")

    @classmethod
    def from_config(cls, total_nodes: int, config: dict[str, Any] | None = None) -> "NodeProfileSet":
        if total_nodes <= 0:
            raise ValueError("Total node count must be positive")

        config = {} if config is None else config
        if not isinstance(config, dict):
            raise ValueError("node_profiles must be a mapping")

        stakes = cls._ordered_values(config.get("stake_tokens"), total_nodes, "stake_tokens", allow_zero=True)
        capabilities = cls._ordered_values(
            config.get("computational_capability_ghz"),
            total_nodes,
            "computational_capability_ghz",
            allow_zero=False,
        )
        return cls(
            tuple(
                NodeProfile(
                    node_id=node_id,
                    stake_tokens=stakes[node_id],
                    computational_capability_ghz=capabilities[node_id],
                )
                for node_id in range(total_nodes)
            )
        )

    @staticmethod
    def _ordered_values(values: Any, total_nodes: int, field_name: str, allow_zero: bool) -> tuple[float | None, ...]:
        if values is None:
            return (None,) * total_nodes
        if not isinstance(values, list) or len(values) != total_nodes:
            raise ValueError(f"node_profiles.{field_name} must contain exactly Nn={total_nodes} ordered values")

        result: list[float | None] = []
        for node_id, value in enumerate(values):
            if value is None:
                result.append(None)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"node_profiles.{field_name}[{node_id}] must be numeric or null")
            numeric = float(value)
            valid = numeric >= 0 if allow_zero else numeric > 0
            if not isfinite(numeric) or not valid:
                comparison = "non-negative" if allow_zero else "positive"
                raise ValueError(f"node_profiles.{field_name}[{node_id}] must be finite and {comparison}")
            result.append(numeric)
        return tuple(result)

    @property
    def count(self) -> int:
        return len(self.profiles)

    def profile_for(self, node_id: int) -> NodeProfile:
        if node_id < 0 or node_id >= self.count:
            raise KeyError(f"No profile for node ID {node_id}")
        return self.profiles[node_id]

    @property
    def capability_scaling_enabled(self) -> bool:
        return any(profile.computational_capability_ghz is not None for profile in self.profiles)
