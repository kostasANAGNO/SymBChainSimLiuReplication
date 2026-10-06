"""Transaction-creation log.

The only record the Liu runtime consumes functionally: the previous epoch's creation records give
the empirical mean transaction size chi (StateBuilder / StateEvolution). Everything else that used
to be recorded here (block proposals, local decisions, block observations, run profiles) was
write-only and has been removed.
"""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class TransactionCreationRecord:
    transaction_id: int
    creator_node_id: int
    original_creation_time: float
    transaction_size: float


class InstrumentationCollector:
    """Run-scoped log of transaction creations."""

    transaction_creations: ClassVar[list[TransactionCreationRecord]] = []

    @classmethod
    def reset(cls) -> None:
        cls.transaction_creations = []

    @classmethod
    def snapshot(cls) -> dict:
        return {"transaction_creations": tuple(cls.transaction_creations)}

    @classmethod
    def restore(cls, snapshot: dict) -> None:
        cls.transaction_creations = list(snapshot["transaction_creations"])

    @classmethod
    def record_transaction_creation(cls, record: TransactionCreationRecord) -> None:
        cls.transaction_creations.append(record)
