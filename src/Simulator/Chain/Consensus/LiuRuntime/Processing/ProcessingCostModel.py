"""Liu cryptographic cycles converted to seconds using c_i GHz."""

from dataclasses import dataclass

from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork


@dataclass(frozen=True, slots=True)
class LiuProcessingCostModel:
    signature_cycles_alpha: float
    mac_cycles_beta: float

    def __post_init__(self) -> None:
        if self.signature_cycles_alpha < 0 or self.mac_cycles_beta < 0:
            raise ValueError("alpha and beta cycle costs cannot be negative")

    def cycles(self, work: LiuProcessingWork) -> float:
        return work.signature_operations * self.signature_cycles_alpha + work.mac_operations * self.mac_cycles_beta

    def duration_s(self, work: LiuProcessingWork, capability_ghz: float) -> float:
        if capability_ghz <= 0:
            raise ValueError("computational capability must be positive GHz")
        return self.cycles(work) / (capability_ghz * 1_000_000_000.0)

