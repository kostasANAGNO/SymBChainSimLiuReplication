"""Immutable configuration for Liu Zyzzyva fast, recovery, and view-change paths."""

from dataclasses import dataclass

from Chain.Consensus.LiuRuntime.Common.Configuration import runtime_epoch_from_parameters
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Liu.Protocol import LiuConsensusProtocol
from Parameters import Parameters


@dataclass(frozen=True, slots=True)
class LiuZyzzyvaRuntimeConfiguration:
    TIMEOUT_POLICY_VERSION = "liu_runtime_phase_relative_timeout_v2"
    PROCESSING_DECOMPOSITION_VERSION = "liu_zyzzyva_fast_processing_decomposition_v1"
    FAST_REPLY_POLICY_VERSION = "liu_zyzzyva_all_replica_replies_v1"
    ROLE_SELECTION_VERSION = "liu_zyzzyva_height_indexed_client_v1"
    RECOVERY_QUORUM_POLICY_VERSION = "liu_zyzzyva_recovery_quorum_v1"
    RECOVERY_SAFE_DIGEST_POLICY_VERSION = "liu_zyzzyva_recovery_safe_digest_v1"
    RECOVERY_PROCESSING_DECOMPOSITION_VERSION = "liu_zyzzyva_recovery_processing_decomposition_v1"
    VIEW_CHANGE_QUORUM_POLICY_VERSION = "liu_zyzzyva_view_change_quorum_v1"
    VIEW_CHANGE_SAFE_VALUE_POLICY_VERSION = "liu_zyzzyva_safe_value_selection_v1"
    VIEW_CHANGE_RESUME_POLICY_VERSION = "liu_zyzzyva_view_change_resume_v1"
    VIEW_TIMEOUT_POLICY_VERSION = "liu_zyzzyva_view_timeout_v1"

    epoch_context: LiuRuntimeEpochContext
    signature_cycles_alpha: float
    mac_cycles_beta: float
    request_timeout_s: float
    propagation_delay_s: float = 0.0
    faulty_replica_count: int = 0

    def __post_init__(self) -> None:
        if self.epoch_context.epoch_configuration.consensus_protocol is not LiuConsensusProtocol.ZYZZYVA:
            raise ValueError("LiuZyzzyvaRuntimeConfiguration requires a ZYZZYVA epoch")
        if self.validator_count_k < 3:
            raise ValueError("LiuZyzzyva requires a client, primary, and at least one backup")
        if self.signature_cycles_alpha < 0 or self.mac_cycles_beta < 0:
            raise ValueError("alpha/beta cycle costs cannot be negative")
        if self.request_timeout_s <= 0 or self.propagation_delay_s < 0:
            raise ValueError("timeout must be positive and propagation delay non-negative")
        if isinstance(self.faulty_replica_count, bool) or not isinstance(self.faulty_replica_count, int):
            raise ValueError("faulty_replica_count must be an integer")
        if self.faulty_replica_count < 0:
            raise ValueError("faulty_replica_count cannot be negative")
        # Liu Eq.(9)/(17): C3 requires f <= F^1 = floor((K-1)/3). Selected-set malicious
        # nodes exceeding F^1 make finality impossible regardless of protocol path.
        malicious_selected = frozenset(self.epoch_context.threat_scenario.malicious_node_ids) & frozenset(
            self.epoch_context.validator_ids
        )
        if len(malicious_selected) > self.tolerated_faults:
            raise ValueError(
                "LiuZyzzyva C3 violation: selected-set faulty > F^1 = floor((K-1)/3)"
            )
        if self.faulty_replica_count != len(malicious_selected):
            raise ValueError(
                "faulty_replica_count must equal |malicious_node_ids ∩ selected validators|"
            )

    @property
    def validator_count_k(self) -> int:
        return len(self.epoch_context.validator_ids)

    @property
    def faulty_replica_ids(self) -> frozenset:
        """Selected validators flagged Byzantine by the epoch threat scenario.

        These are the nodes whose speculative reply digest is deliberately mismatched to
        trigger Liu's Eq.(17) recovery path (`DES_RECONSTRUCTION` for "faulty replica may
        send incorrect messages"; Liu does not specify the exact incorrect content).
        """
        return frozenset(self.epoch_context.threat_scenario.malicious_node_ids) & frozenset(
            self.epoch_context.validator_ids
        )

    @property
    def required_fast_replies(self) -> int:
        return self.validator_count_k - 1

    @property
    def tolerated_faults(self) -> int:
        return (self.validator_count_k - 1) // 3

    @property
    def speculative_recovery_quorum(self) -> int:
        return 2 * self.tolerated_faults + 1

    @property
    def local_commit_quorum(self) -> int:
        return 2 * self.tolerated_faults + 1

    @property
    def replica_count(self) -> int:
        return self.validator_count_k - 1

    @property
    def view_change_quorum(self) -> int:
        """Smallest q whose two-quorum intersection is strictly larger than f."""
        return (self.replica_count + self.tolerated_faults) // 2 + 1

    @property
    def speculative_safety_threshold(self) -> int:
        return self.tolerated_faults + 1

    @classmethod
    def from_parameters(cls, node) -> "LiuZyzzyvaRuntimeConfiguration":
        config = Parameters.LiuZyzzyva
        return cls(
            runtime_epoch_from_parameters(node, LiuConsensusProtocol.ZYZZYVA, config),
            float(config["signature_cycles_alpha"]),
            float(config["mac_cycles_beta"]),
            float(config["request_timeout_s"]),
            float(config.get("propagation_delay_s", 0.0)),
            int(config.get("faulty_replica_count", 0)),
        )
