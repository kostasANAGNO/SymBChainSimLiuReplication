"""Instance-scoped instrumentation for Liu environment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True, slots=True)
class LiuEnvironmentEventRecord:
    event_type: str
    episode_id: int
    decision_step: int
    event_time: float
    state_hash: str | None = None
    action_hash: str | None = None
    evaluation_hash: str | None = None
    status: str | None = None
    detail: str | None = None


class LiuEnvironmentInstrumentationCollector:
    """A per-environment append-only stream; never shared through class state."""

    def __init__(self) -> None:
        self.records: list[LiuEnvironmentEventRecord] = []

    def reset(self) -> None:
        self.records = []

    def append(self, record: LiuEnvironmentEventRecord) -> None:
        if not isinstance(record, LiuEnvironmentEventRecord):
            raise ValueError("record must be a LiuEnvironmentEventRecord")
        self.records.append(record)

    def snapshot(self) -> tuple[LiuEnvironmentEventRecord, ...]:
        return tuple(self.records)

    def deterministic_hash(self) -> str:
        encoded = json.dumps(
            [asdict(record) for record in self.records],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
