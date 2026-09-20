"""Explicit message-size interpretation used by the Liu paper equations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LiuPaperMessageSizePolicy:
    """Liu Appendix-B policy: Request and Reply each carry S^B MB."""

    version: str = "liu_paper_block_size_per_phase_v1"

    def request_size_mb(self, block_size_mb: float) -> float:
        return block_size_mb

    def reply_size_mb(self, block_size_mb: float) -> float:
        return block_size_mb

    def finalized_announcement_size_mb(self, block_size_mb: float) -> float:
        return block_size_mb

