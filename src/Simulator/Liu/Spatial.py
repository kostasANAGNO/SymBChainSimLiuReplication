"""Explicit two-dimensional node placement for the Liu domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer


@dataclass(frozen=True, slots=True)
class SpatialProfile(CanonicalSerializable):
    node_id: int
    x_km: float
    y_km: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", require_integer(self.node_id, "node_id"))
        object.__setattr__(self, "x_km", require_finite_number(self.x_km, "x_km", non_negative=True))
        object.__setattr__(self, "y_km", require_finite_number(self.y_km, "y_km", non_negative=True))

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "x_km": self.x_km, "y_km": self.y_km}


@dataclass(frozen=True, slots=True)
class SpatialProfileSet(CanonicalSerializable):
    profiles: tuple[SpatialProfile, ...]
    region_width_km: float
    region_height_km: float

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles:
            raise ValueError("SpatialProfileSet requires at least one profile")
        if not all(isinstance(profile, SpatialProfile) for profile in profiles):
            raise ValueError("profiles must contain only SpatialProfile values")

        expected_ids = tuple(range(len(profiles)))
        actual_ids = tuple(profile.node_id for profile in profiles)
        if actual_ids != expected_ids:
            raise ValueError(f"spatial profiles must be complete and ordered for node IDs {expected_ids}")

        width = require_finite_number(self.region_width_km, "region_width_km", positive=True)
        height = require_finite_number(self.region_height_km, "region_height_km", positive=True)
        for profile in profiles:
            if profile.x_km > width or profile.y_km > height:
                raise ValueError(f"coordinates for node {profile.node_id} are outside the configured region bounds")

        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "region_width_km", width)
        object.__setattr__(self, "region_height_km", height)

    @classmethod
    def from_coordinates(
        cls,
        coordinates_km: Iterable[tuple[float, float]],
        *,
        region_width_km: float,
        region_height_km: float,
    ) -> "SpatialProfileSet":
        coordinates = tuple(coordinates_km)
        return cls(
            tuple(SpatialProfile(node_id, coordinate[0], coordinate[1]) for node_id, coordinate in enumerate(coordinates)),
            region_width_km,
            region_height_km,
        )

    @property
    def node_count(self) -> int:
        return len(self.profiles)

    def profile_for(self, node_id: int) -> SpatialProfile:
        require_integer(node_id, "node_id")
        if node_id >= self.node_count:
            raise KeyError(f"No spatial profile for node ID {node_id}")
        return self.profiles[node_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "region_height_km": self.region_height_km,
            "region_width_km": self.region_width_km,
        }
