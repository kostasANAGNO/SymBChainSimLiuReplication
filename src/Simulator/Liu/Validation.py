"""Shared validation primitives for the Liu domain layer."""

from __future__ import annotations

from math import isfinite
from typing import Any


def require_integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def require_finite_number(
    value: Any,
    field_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{field_name} must be positive")
    if non_negative and numeric < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return numeric
