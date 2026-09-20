"""Immutable representation of the Liu state S(t) = [chi, Upsilon, x, c, R]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Liu.LinkState import LinkStateMatrix
from Liu.Serialization import CanonicalSerializable
from Liu.Spatial import SpatialProfileSet
from Liu.Validation import require_finite_number


@dataclass(frozen=True, slots=True)
class LiuState(CanonicalSerializable):
    transaction_size_bytes: float
    stakes_tokens: tuple[float, ...]
    spatial_profiles: SpatialProfileSet
    computational_capabilities_ghz: tuple[float, ...]
    link_state_matrix: LinkStateMatrix

    def __post_init__(self) -> None:
        transaction_size = require_finite_number(
            self.transaction_size_bytes,
            "transaction_size_bytes",
            positive=True,
        )
        stakes = tuple(
            require_finite_number(value, f"stakes_tokens[{node_id}]", non_negative=True)
            for node_id, value in enumerate(self.stakes_tokens)
        )
        capabilities = tuple(
            require_finite_number(value, f"computational_capabilities_ghz[{node_id}]", positive=True)
            for node_id, value in enumerate(self.computational_capabilities_ghz)
        )
        if not stakes:
            raise ValueError("LiuState requires at least one node")
        node_count = len(stakes)
        if len(capabilities) != node_count:
            raise ValueError("stakes_tokens and computational_capabilities_ghz must have the same N")
        if not isinstance(self.spatial_profiles, SpatialProfileSet):
            raise ValueError("spatial_profiles must be a SpatialProfileSet")
        if not isinstance(self.link_state_matrix, LinkStateMatrix):
            raise ValueError("link_state_matrix must be a LinkStateMatrix")
        if self.spatial_profiles.node_count != node_count or self.link_state_matrix.node_count != node_count:
            raise ValueError("all LiuState components must describe the same N nodes")

        object.__setattr__(self, "transaction_size_bytes", transaction_size)
        object.__setattr__(self, "stakes_tokens", stakes)
        object.__setattr__(self, "computational_capabilities_ghz", capabilities)

    @property
    def node_count(self) -> int:
        return len(self.stakes_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "computational_capabilities_ghz": list(self.computational_capabilities_ghz),
            "link_state_matrix": self.link_state_matrix.to_dict(),
            "spatial_profiles": self.spatial_profiles.to_dict(),
            "stakes_tokens": list(self.stakes_tokens),
            "transaction_size_bytes": self.transaction_size_bytes,
        }
