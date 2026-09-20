"""Pure spatial-intensity estimators for Liu geographic decentralization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose
from typing import Any, ClassVar, Iterable

from Liu.Serialization import CanonicalSerializable
from Liu.Spatial import SpatialProfileSet
from Liu.Validation import require_finite_number, require_integer


@dataclass(frozen=True, slots=True)
class GeographicGiniResult(CanonicalSerializable):
    """Versioned result of integrating a selected validator intensity field."""

    estimator_version: str
    selected_validator_ids: tuple[int, ...]
    grid_rows: int
    grid_columns: int
    region_area_km2: float
    cell_area_km2: float
    cell_validator_counts: tuple[int, ...]
    cell_intensities_per_km2: tuple[float, ...]
    integrated_intensity: float
    expected_validator_count: int
    integration_error: float
    geographic_gini: float

    def __post_init__(self) -> None:
        if not isinstance(self.estimator_version, str) or not self.estimator_version:
            raise ValueError("estimator_version must be a non-empty string")
        selected_ids = tuple(self.selected_validator_ids)
        if not selected_ids:
            raise ValueError("selected_validator_ids cannot be empty")
        if any(isinstance(node_id, bool) or not isinstance(node_id, int) for node_id in selected_ids):
            raise ValueError("selected validator IDs must be integers")
        if selected_ids != tuple(sorted(set(selected_ids))):
            raise ValueError("selected validator IDs must be unique and ordered")

        grid_rows = require_integer(self.grid_rows, "grid_rows", minimum=1)
        grid_columns = require_integer(self.grid_columns, "grid_columns", minimum=1)
        cell_count = grid_rows * grid_columns
        counts = tuple(self.cell_validator_counts)
        if len(counts) != cell_count:
            raise ValueError("cell_validator_counts shape must match the grid")
        counts = tuple(
            require_integer(count, f"cell_validator_counts[{index}]")
            for index, count in enumerate(counts)
        )
        intensities = tuple(
            require_finite_number(
                intensity,
                f"cell_intensities_per_km2[{index}]",
                non_negative=True,
            )
            for index, intensity in enumerate(self.cell_intensities_per_km2)
        )
        if len(intensities) != cell_count:
            raise ValueError("cell_intensities_per_km2 shape must match the grid")

        region_area = require_finite_number(self.region_area_km2, "region_area_km2", positive=True)
        cell_area = require_finite_number(self.cell_area_km2, "cell_area_km2", positive=True)
        if not isclose(cell_area * cell_count, region_area, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("cell areas must sum to the configured region area")
        expected_count = require_integer(
            self.expected_validator_count,
            "expected_validator_count",
            minimum=1,
        )
        if len(selected_ids) != expected_count or sum(counts) != expected_count:
            raise ValueError("selected IDs and cell counts must both represent K validators")
        for index, (count, intensity) in enumerate(zip(counts, intensities)):
            if not isclose(intensity * cell_area, count, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"cell intensity {index} is inconsistent with its validator count")

        integrated = require_finite_number(self.integrated_intensity, "integrated_intensity", non_negative=True)
        integration_error = require_finite_number(self.integration_error, "integration_error")
        if not isclose(integrated - expected_count, integration_error, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("integration_error is inconsistent with integrated_intensity and K")
        geographic_gini = require_finite_number(self.geographic_gini, "geographic_gini", non_negative=True)
        if geographic_gini > 1.0:
            raise ValueError("geographic_gini must be in [0, 1]")

        object.__setattr__(self, "selected_validator_ids", selected_ids)
        object.__setattr__(self, "grid_rows", grid_rows)
        object.__setattr__(self, "grid_columns", grid_columns)
        object.__setattr__(self, "region_area_km2", region_area)
        object.__setattr__(self, "cell_area_km2", cell_area)
        object.__setattr__(self, "cell_validator_counts", counts)
        object.__setattr__(self, "cell_intensities_per_km2", intensities)
        object.__setattr__(self, "integrated_intensity", integrated)
        object.__setattr__(self, "expected_validator_count", expected_count)
        object.__setattr__(self, "integration_error", integration_error)
        object.__setattr__(self, "geographic_gini", geographic_gini)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_area_km2": self.cell_area_km2,
            "cell_intensities_per_km2": list(self.cell_intensities_per_km2),
            "cell_validator_counts": list(self.cell_validator_counts),
            "estimator_version": self.estimator_version,
            "expected_validator_count": self.expected_validator_count,
            "geographic_gini": self.geographic_gini,
            "grid_columns": self.grid_columns,
            "grid_rows": self.grid_rows,
            "integrated_intensity": self.integrated_intensity,
            "integration_error": self.integration_error,
            "region_area_km2": self.region_area_km2,
            "selected_validator_ids": list(self.selected_validator_ids),
        }


class SpatialIntensityModel(ABC):
    """Interface for estimating Liu's spatial intensity from selected nodes."""

    @abstractmethod
    def evaluate(self, selected_validator_ids: Iterable[int]) -> GeographicGiniResult:
        """Estimate the selected-validator intensity and its geographic Gini."""


@dataclass(frozen=True, slots=True)
class GridSpatialIntensityModel(SpatialIntensityModel, CanonicalSerializable):
    """Equal-area histogram estimator over a rectangular spatial profile region."""

    ESTIMATOR_VERSION: ClassVar[str] = "grid_v1"
    INTEGRATION_TOLERANCE: ClassVar[float] = 1e-12

    spatial_profiles: SpatialProfileSet
    grid_rows: int
    grid_columns: int

    def __post_init__(self) -> None:
        if not isinstance(self.spatial_profiles, SpatialProfileSet):
            raise ValueError("spatial_profiles must be a SpatialProfileSet")
        object.__setattr__(self, "grid_rows", require_integer(self.grid_rows, "grid_rows", minimum=1))
        object.__setattr__(self, "grid_columns", require_integer(self.grid_columns, "grid_columns", minimum=1))

    @property
    def region_area_km2(self) -> float:
        return self.spatial_profiles.region_width_km * self.spatial_profiles.region_height_km

    @property
    def cell_area_km2(self) -> float:
        return self.region_area_km2 / (self.grid_rows * self.grid_columns)

    def evaluate(self, selected_validator_ids: Iterable[int]) -> GeographicGiniResult:
        supplied_ids = tuple(selected_validator_ids)
        if not supplied_ids:
            raise ValueError("selected_validator_ids cannot be empty")
        if any(isinstance(node_id, bool) or not isinstance(node_id, int) for node_id in supplied_ids):
            raise ValueError("selected validator IDs must be integers")
        if len(set(supplied_ids)) != len(supplied_ids):
            raise ValueError("selected validator IDs must be unique")
        if any(node_id < 0 or node_id >= self.spatial_profiles.node_count for node_id in supplied_ids):
            raise ValueError("selected validator ID is outside the SpatialProfileSet")
        selected_ids = tuple(sorted(supplied_ids))

        cell_count = self.grid_rows * self.grid_columns
        counts = [0] * cell_count
        cell_width = self.spatial_profiles.region_width_km / self.grid_columns
        cell_height = self.spatial_profiles.region_height_km / self.grid_rows
        for node_id in selected_ids:
            profile = self.spatial_profiles.profile_for(node_id)
            # Coordinates on the inclusive upper boundary belong to the final
            # row/column rather than creating an out-of-range cell.
            column = min(int(profile.x_km / cell_width), self.grid_columns - 1)
            row = min(int(profile.y_km / cell_height), self.grid_rows - 1)
            counts[row * self.grid_columns + column] += 1

        cell_area = self.cell_area_km2
        intensities = tuple(count / cell_area for count in counts)
        integrated_intensity = sum(intensity * cell_area for intensity in intensities)
        expected_count = len(selected_ids)
        integration_error = integrated_intensity - expected_count
        if not isclose(
            integrated_intensity,
            expected_count,
            rel_tol=0.0,
            abs_tol=self.INTEGRATION_TOLERANCE,
        ):
            raise ValueError("grid intensity does not integrate to the selected validator count K")

        pairwise_count_difference = sum(abs(left - right) for left in counts for right in counts)
        geographic_gini = pairwise_count_difference / (2 * cell_count * expected_count)
        return GeographicGiniResult(
            estimator_version=self.ESTIMATOR_VERSION,
            selected_validator_ids=selected_ids,
            grid_rows=self.grid_rows,
            grid_columns=self.grid_columns,
            region_area_km2=self.region_area_km2,
            cell_area_km2=cell_area,
            cell_validator_counts=tuple(counts),
            cell_intensities_per_km2=intensities,
            integrated_intensity=integrated_intensity,
            expected_validator_count=expected_count,
            integration_error=integration_error,
            geographic_gini=geographic_gini,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator_version": self.ESTIMATOR_VERSION,
            "grid_columns": self.grid_columns,
            "grid_rows": self.grid_rows,
            "spatial_profiles": self.spatial_profiles.to_dict(),
        }
