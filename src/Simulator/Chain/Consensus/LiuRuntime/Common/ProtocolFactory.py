"""Protocol-neutral construction of Liu runtime instances for a new epoch."""

from __future__ import annotations

from dataclasses import dataclass

from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.LiuPBFT.Configuration import LiuPBFTRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuQuorum.Configuration import (
    LiuQuorumRuntimeConfiguration,
    ReplicaFault,
    ReplicaFaultPolicy,
)
from Chain.Consensus.LiuRuntime.LiuZyzzyva.Configuration import LiuZyzzyvaRuntimeConfiguration
from Liu.Protocol import LiuConsensusProtocol
from Parameters import Parameters


@dataclass(frozen=True, slots=True)
class LiuRuntimeProtocolSettings:
    signature_cycles_alpha: float
    mac_cycles_beta: float
    request_timeout_s: float
    propagation_delay_s: float = 0.0
    quorum_replica_faults: tuple[ReplicaFault, ...] = ()
    zyzzyva_faulty_replica_count: int = 0
    quorum_timeout_safety_factor: float = 2.0
    quorum_timeout_maximum_s: float = 320.0


class LiuRuntimeProtocolFactory:
    """Bind common runtime settings to an immutable epoch context."""

    def __init__(self, settings: LiuRuntimeProtocolSettings) -> None:
        self.settings = settings

    @classmethod
    def from_parameters(cls, protocol: LiuConsensusProtocol) -> "LiuRuntimeProtocolFactory":
        if protocol is LiuConsensusProtocol.LIU_QUORUM:
            source = Parameters.LiuQuorum
        elif protocol is LiuConsensusProtocol.PBFT:
            source = Parameters.LiuPBFT
        else:
            source = Parameters.LiuZyzzyva
        quorum_faults = ()
        if protocol is LiuConsensusProtocol.LIU_QUORUM:
            quorum_faults = tuple(
                ReplicaFault(
                    int(item["node_id"]),
                    ReplicaFaultPolicy(item["policy"]),
                    float(item.get("delay_s", 0.0)),
                )
                for item in source.get("replica_faults", ())
            )
        return cls(
            LiuRuntimeProtocolSettings(
                float(source["signature_cycles_alpha"]),
                float(source["mac_cycles_beta"]),
                float(source["request_timeout_s"]),
                float(source.get("propagation_delay_s", 0.0)),
                quorum_faults,
                int(source.get("faulty_replica_count", 0)),
                float(source.get("timeout_safety_factor", 2.0)),
                float(source.get("timeout_maximum_s", 320.0)),
            )
        )

    def configuration_for(self, context: LiuRuntimeEpochContext):
        settings = self.settings
        protocol = context.epoch_configuration.consensus_protocol
        common = (
            context,
            settings.signature_cycles_alpha,
            settings.mac_cycles_beta,
            settings.request_timeout_s,
            settings.propagation_delay_s,
        )
        if protocol is LiuConsensusProtocol.LIU_QUORUM:
            return LiuQuorumRuntimeConfiguration(
                *common,
                settings.quorum_replica_faults,
                settings.quorum_timeout_safety_factor,
                settings.quorum_timeout_maximum_s,
            )
        if protocol is LiuConsensusProtocol.PBFT:
            return LiuPBFTRuntimeConfiguration(*common)
        if protocol is LiuConsensusProtocol.ZYZZYVA:
            return LiuZyzzyvaRuntimeConfiguration(*common, settings.zyzzyva_faulty_replica_count)
        raise ValueError(f"unsupported Liu runtime protocol: {protocol}")

    def create(self, node, context: LiuRuntimeEpochContext):
        # Lazy imports avoid making the common package depend on protocol module
        # initialization order.
        from Chain.Consensus.LiuRuntime.LiuPBFT.Protocol import LiuPBFT
        from Chain.Consensus.LiuRuntime.LiuQuorum.Protocol import LiuQuorum
        from Chain.Consensus.LiuRuntime.LiuZyzzyva.Protocol import LiuZyzzyva

        protocol_type = {
            LiuConsensusProtocol.LIU_QUORUM: LiuQuorum,
            LiuConsensusProtocol.PBFT: LiuPBFT,
            LiuConsensusProtocol.ZYZZYVA: LiuZyzzyva,
        }[context.epoch_configuration.consensus_protocol]
        return protocol_type(node, self.configuration_for(context))
