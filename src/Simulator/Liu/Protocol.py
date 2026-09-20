"""Consensus choices exposed by the Liu action domain."""

from enum import Enum


class LiuConsensusProtocol(str, Enum):
    PBFT = "PBFT"
    ZYZZYVA = "ZYZZYVA"
    LIU_QUORUM = "LIU_QUORUM"


def require_liu_protocol(value: object) -> LiuConsensusProtocol:
    if not isinstance(value, LiuConsensusProtocol):
        allowed = ", ".join(protocol.value for protocol in LiuConsensusProtocol)
        raise ValueError(f"consensus_protocol must be a LiuConsensusProtocol ({allowed})")
    return value
