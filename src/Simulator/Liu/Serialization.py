"""Canonical serialization helpers for immutable Liu domain values."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from typing import Any


class CanonicalSerializable(ABC):
    """Expose stable JSON serialization and a process-independent SHA-256."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation using explicit domain units."""

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def deterministic_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
