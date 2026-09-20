"""A serial, deterministic CPU reservation queue for one simulated node."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProcessingReservation:
    arrival_time: float
    processing_start: float
    processing_end: float
    processing_duration: float


class NodeComputeQueue:
    def __init__(self) -> None:
        self.available_at = 0.0

    def reserve(self, arrival_time: float, processing_duration: float) -> ProcessingReservation:
        if processing_duration < 0:
            raise ValueError("processing duration cannot be negative")
        start = max(arrival_time, self.available_at)
        end = start + processing_duration
        self.available_at = end
        return ProcessingReservation(arrival_time, start, end, processing_duration)

