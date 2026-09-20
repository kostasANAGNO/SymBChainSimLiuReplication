"""Immutable LiuPBFT normal-path configuration and quorum reconstruction."""

from dataclasses import dataclass

from Chain.Consensus.LiuRuntime.Common.Configuration import runtime_epoch_from_parameters
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Liu.Protocol import LiuConsensusProtocol
from Parameters import Parameters


@dataclass(frozen=True, slots=True)
class LiuPBFTRuntimeConfiguration:
    TIMEOUT_POLICY_VERSION = "liu_runtime_phase_relative_timeout_v2"
    VIEW_CATCHUP_POLICY_VERSION = "liu_pbft_view_catchup_v1"
    VIEW_CATCHUP_CLASSIFICATION = "RECONSTRUCTION_REQUIRED"
    TIMEOUT_BACKOFF_POLICY_VERSION = "liu_pbft_deterministic_timeout_backoff_v1"
    TIMEOUT_BACKOFF_CLASSIFICATION = "RECONSTRUCTION_REQUIRED"
    NEW_VIEW_PROPOSAL_DELIVERY_VERSION = "liu_pbft_new_view_proposal_delivery_v1"
    NEW_VIEW_PROPOSAL_DELIVERY_CLASSIFICATION = "RECONSTRUCTION_REQUIRED"
    QUORUM_POLICY_VERSION = "liu_pbft_k_minus_one_replicas_v1"
    PROCESSING_DECOMPOSITION_VERSION = "liu_pbft_appendix_b_aggregate_split_v1"
    VIEW_CHANGE_QUORUM_POLICY_VERSION = "liu_pbft_view_change_quorum_v1"
    SAFE_VALUE_SELECTION_VERSION = "liu_pbft_safe_value_selection_v1"

    epoch_context: LiuRuntimeEpochContext
    signature_cycles_alpha: float
    mac_cycles_beta: float
    request_timeout_s: float
    propagation_delay_s: float = 0.0
    timeout_backoff_factor: float = 2.0
    timeout_max_s: float | None = None
    maximum_timeout_view_changes: int = 6

    def __post_init__(self) -> None:
        if self.epoch_context.epoch_configuration.consensus_protocol is not LiuConsensusProtocol.PBFT:
            raise ValueError("LiuPBFTRuntimeConfiguration requires a PBFT epoch")
        if self.validator_count_k < 4:
            raise ValueError("LiuPBFT requires at least K=4 validators")
        if self.signature_cycles_alpha < 0 or self.mac_cycles_beta < 0:
            raise ValueError("alpha/beta cycle costs cannot be negative")
        if self.request_timeout_s <= 0 or self.propagation_delay_s < 0:
            raise ValueError("timeout must be positive and propagation delay non-negative")
        if self.timeout_backoff_factor <= 1.0:
            raise ValueError("PBFT timeout backoff factor must be greater than one")
        if self.timeout_max_s is None:
            object.__setattr__(self, "timeout_max_s", self.request_timeout_s * 64.0)
        if self.timeout_max_s < self.request_timeout_s:
            raise ValueError("PBFT maximum timeout cannot be below the initial timeout")
        if self.maximum_timeout_view_changes < 1:
            raise ValueError("PBFT maximum timeout-driven view changes must be positive")

    def effective_timeout_s(self, consecutive_timeout_view_changes: int) -> float:
        if consecutive_timeout_view_changes < 0:
            raise ValueError("consecutive timeout count cannot be negative")
        return min(
            self.timeout_max_s,
            self.request_timeout_s * self.timeout_backoff_factor ** consecutive_timeout_view_changes,
        )

    @property
    def validator_count_k(self) -> int:
        return len(self.epoch_context.validator_ids)

    @property
    def replica_count(self) -> int:
        return self.validator_count_k - 1

    @property
    def tolerated_faults(self) -> int:
        return (self.validator_count_k - 1) // 3

    @property
    def prepare_quorum(self) -> int:
        return self.replica_count - self.tolerated_faults

    @property
    def commit_quorum(self) -> int:
        return self.replica_count - self.tolerated_faults

    @property
    def reply_quorum(self) -> int:
        return self.tolerated_faults + 1

    @property
    def view_change_quorum(self) -> int:
        return self.replica_count - self.tolerated_faults

    @classmethod
    def from_parameters(cls, node) -> "LiuPBFTRuntimeConfiguration":
        config = Parameters.LiuPBFT
        return cls(
            runtime_epoch_from_parameters(node, LiuConsensusProtocol.PBFT, config),
            float(config["signature_cycles_alpha"]),
            float(config["mac_cycles_beta"]),
            float(config["request_timeout_s"]),
            float(config.get("propagation_delay_s", 0.0)),
            float(config.get("timeout_backoff_factor", 2.0)),
            float(config.get("timeout_max_s", 320.0)),
            int(config.get("maximum_timeout_view_changes", 6)),
        )
