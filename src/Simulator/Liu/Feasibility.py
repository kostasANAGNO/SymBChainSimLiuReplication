"""Immutable results for later Liu constraint evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number


@dataclass(frozen=True, slots=True)
class ConstraintResult(CanonicalSerializable):
    constraint_name: str
    satisfied: bool
    measured_value: float | None = None
    threshold_value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.constraint_name, str) or not self.constraint_name.strip():
            raise ValueError("constraint_name must be a non-empty string")
        if not isinstance(self.satisfied, bool):
            raise ValueError("satisfied must be a boolean")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("reason must be None or a non-empty string")
        if self.measured_value is not None:
            object.__setattr__(
                self,
                "measured_value",
                require_finite_number(self.measured_value, "measured_value"),
            )
        if self.threshold_value is not None:
            object.__setattr__(
                self,
                "threshold_value",
                require_finite_number(self.threshold_value, "threshold_value"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "measured_value": self.measured_value,
            "reason": self.reason,
            "satisfied": self.satisfied,
            "threshold_value": self.threshold_value,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityResult(CanonicalSerializable):
    constraint_results: tuple[ConstraintResult, ...]

    def __post_init__(self) -> None:
        results = tuple(self.constraint_results)
        if not all(isinstance(result, ConstraintResult) for result in results):
            raise ValueError("constraint_results must contain only ConstraintResult values")
        names = tuple(result.constraint_name for result in results)
        if len(set(names)) != len(names):
            raise ValueError("constraint names must be unique")
        object.__setattr__(self, "constraint_results", results)

    @property
    def feasible(self) -> bool:
        return all(result.satisfied for result in self.constraint_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_results": [result.to_dict() for result in self.constraint_results],
            "feasible": self.feasible,
        }
