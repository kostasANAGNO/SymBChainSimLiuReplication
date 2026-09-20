"""Immutable cryptographic work expressed in Liu alpha/beta operations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiuProcessingWork:
    signature_operations: int
    mac_operations: int

    def __post_init__(self) -> None:
        if self.signature_operations < 0 or self.mac_operations < 0:
            raise ValueError("processing operation counts cannot be negative")

    @classmethod
    def quorum_replica_work(cls) -> "LiuProcessingWork":
        return cls(signature_operations=1, mac_operations=2)

    @classmethod
    def pbft_primary_request_work(cls) -> "LiuProcessingWork":
        return cls(signature_operations=1, mac_operations=2)

    @classmethod
    def pbft_backup_preprepare_work(cls) -> "LiuProcessingWork":
        return cls(signature_operations=1, mac_operations=1)

    @classmethod
    def pbft_prepare_quorum_work(cls, replica_count: int) -> "LiuProcessingWork":
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        return cls(signature_operations=0, mac_operations=2 * replica_count)

    @classmethod
    def pbft_commit_quorum_work(cls, replica_count: int) -> "LiuProcessingWork":
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        return cls(signature_operations=0, mac_operations=2 * replica_count)

    @classmethod
    def zyzzyva_primary_order_work(cls, replica_count: int) -> "LiuProcessingWork":
        """Equation-16 primary-only overhead for a single-request batch."""
        if replica_count <= 0:
            raise ValueError("replica_count must be positive")
        return cls(signature_operations=0, mac_operations=replica_count)

    @classmethod
    def zyzzyva_speculative_replica_work(cls) -> "LiuProcessingWork":
        """Equation-16 per-replica speculative validation/execution work for M=1."""
        return cls(signature_operations=1, mac_operations=2)

    @classmethod
    def zyzzyva_recovery_work(cls, is_primary: bool, tolerated_faults: int) -> "LiuProcessingWork":
        """Equation-17 additional work for the M=1 recovery path."""
        if tolerated_faults < 0:
            raise ValueError("tolerated_faults cannot be negative")
        return cls(signature_operations=0, mac_operations=tolerated_faults + 2 if is_primary else 2)
