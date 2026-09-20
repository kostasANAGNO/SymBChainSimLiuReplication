"""Explicit Byzantine identities kept outside the Liu state vector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_integer


@dataclass(frozen=True, slots=True)
class ThreatScenario(CanonicalSerializable):
    node_count: int
    malicious_node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        node_count = require_integer(self.node_count, "node_count", minimum=1)
        malicious_ids = tuple(self.malicious_node_ids)
        if any(isinstance(node_id, bool) or not isinstance(node_id, int) for node_id in malicious_ids):
            raise ValueError("malicious node IDs must be integers")
        if len(set(malicious_ids)) != len(malicious_ids):
            raise ValueError("malicious node IDs must be unique")
        if any(node_id < 0 or node_id >= node_count for node_id in malicious_ids):
            raise ValueError("malicious node ID is outside the configured N nodes")
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "malicious_node_ids", tuple(sorted(malicious_ids)))

    def malicious_validator_count(self, validator_ids: tuple[int, ...]) -> int:
        malicious = frozenset(self.malicious_node_ids)
        return sum(node_id in malicious for node_id in validator_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"malicious_node_ids": list(self.malicious_node_ids), "node_count": self.node_count}
