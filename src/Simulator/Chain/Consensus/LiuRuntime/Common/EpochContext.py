"""Immutable runtime snapshot used by every Liu consensus event."""

from dataclasses import dataclass

from Chain.NodeProfile import NodeProfileSet
from Liu.EpochConfiguration import EpochConfiguration
from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.Threat import ThreatScenario


@dataclass(frozen=True, slots=True)
class LiuRuntimeEpochContext(CanonicalSerializable):
    epoch_configuration: EpochConfiguration
    node_profiles: NodeProfileSet
    link_state_matrix: LinkStateMatrix
    threat_scenario: ThreatScenario

    def __post_init__(self) -> None:
        epoch = self.epoch_configuration
        if epoch.consensus_protocol not in (
            LiuConsensusProtocol.LIU_QUORUM,
            LiuConsensusProtocol.PBFT,
            LiuConsensusProtocol.ZYZZYVA,
        ):
            raise ValueError("unsupported Liu runtime consensus protocol")
        node_count = len(self.node_profiles.profiles)
        if self.link_state_matrix.node_count != node_count or self.threat_scenario.node_count != node_count:
            raise ValueError("profiles, links, and threat scenario must describe the same N nodes")
        if any(node_id >= node_count for node_id in epoch.validator_set.ids):
            raise ValueError("epoch validator is outside the runtime node population")
        if any(profile.computational_capability_ghz is None for profile in self.node_profiles.profiles):
            raise ValueError("Liu runtime requires an explicit computational capability for every node")

    @property
    def epoch_id(self) -> int:
        return self.epoch_configuration.epoch_id

    @property
    def epoch_hash(self) -> str:
        return self.epoch_configuration.deterministic_hash()

    @property
    def validator_ids(self) -> tuple[int, ...]:
        return self.epoch_configuration.validator_set.ids

    def capability_ghz(self, node_id: int) -> float:
        capability = self.node_profiles.profile_for(node_id).computational_capability_ghz
        assert capability is not None
        return capability

    def to_dict(self) -> dict:
        return {
            "epoch_configuration": self.epoch_configuration.to_dict(),
            "link_state_matrix": self.link_state_matrix.to_dict(),
            "node_profiles": [
                {
                    "computational_capability_ghz": profile.computational_capability_ghz,
                    "node_id": profile.node_id,
                    "stake_tokens": profile.stake_tokens,
                }
                for profile in self.node_profiles.profiles
            ],
            "threat_scenario": self.threat_scenario.to_dict(),
        }
