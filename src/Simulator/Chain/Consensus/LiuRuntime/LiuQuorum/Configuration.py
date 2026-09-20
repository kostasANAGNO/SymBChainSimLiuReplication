"""Validated immutable configuration for a LiuQuorum runtime epoch."""

from dataclasses import dataclass
from enum import Enum

from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.Configuration import runtime_epoch_from_parameters
from Liu.Protocol import LiuConsensusProtocol
from Parameters import Parameters


class ReplicaFaultPolicy(str, Enum):
    OMIT = "omit"
    CORRUPT = "corrupt"
    DELAY = "delay"


@dataclass(frozen=True, slots=True)
class ReplicaFault:
    node_id: int
    policy: ReplicaFaultPolicy
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int) or self.node_id < 0:
            raise ValueError("fault node_id must be a non-negative integer")
        if not isinstance(self.policy, ReplicaFaultPolicy):
            raise ValueError("fault policy must be a ReplicaFaultPolicy")
        if self.delay_s < 0:
            raise ValueError("fault delay cannot be negative")
        if self.policy is ReplicaFaultPolicy.DELAY and self.delay_s <= 0:
            raise ValueError("delay policy requires a positive delay_s")


@dataclass(frozen=True, slots=True)
class LiuQuorumRuntimeConfiguration:
    TIMEOUT_POLICY_VERSION = "liu_runtime_phase_relative_timeout_v2"
    REQUEST_TIMEOUT_POLICY_VERSION = "liu_quorum_state_action_timeout_v1"
    epoch_context: LiuRuntimeEpochContext
    signature_cycles_alpha: float
    mac_cycles_beta: float
    request_timeout_s: float
    propagation_delay_s: float = 0.0
    replica_faults: tuple[ReplicaFault, ...] = ()
    timeout_safety_factor: float = 2.0
    timeout_maximum_s: float = 320.0

    def __post_init__(self) -> None:
        if self.signature_cycles_alpha < 0 or self.mac_cycles_beta < 0:
            raise ValueError("alpha/beta cycle costs cannot be negative")
        if self.request_timeout_s <= 0 or self.propagation_delay_s < 0:
            raise ValueError("timeout must be positive and propagation delay non-negative")
        if self.timeout_safety_factor < 1.0:
            raise ValueError("timeout safety factor must be >= 1 so the estimate is never undersized")
        if self.timeout_maximum_s < self.request_timeout_s:
            raise ValueError("timeout ceiling cannot be below the minimum request timeout")
        fault_ids = tuple(fault.node_id for fault in self.replica_faults)
        if len(fault_ids) != len(set(fault_ids)):
            raise ValueError("only one deterministic fault policy may be assigned per replica")
        if any(node_id not in self.epoch_context.validator_ids for node_id in fault_ids):
            raise ValueError("LiuQuorum fault-injection policies apply only to selected validators")
        if not self.faithful_configuration_feasible:
            raise ValueError("LiuQuorum F^2=0: a selected malicious validator makes the runtime configuration infeasible")

    @property
    def faithful_configuration_feasible(self) -> bool:
        malicious = set(self.epoch_context.threat_scenario.malicious_node_ids)
        return not malicious.intersection(self.epoch_context.validator_ids)

    def fault_for(self, node_id: int) -> ReplicaFault | None:
        return next((fault for fault in self.replica_faults if fault.node_id == node_id), None)

    def request_timeout_for(self, estimated_slowest_round_trip_s: float) -> float:
        """Clamp the state/action-aware estimate into [request_timeout_s, ceiling].

        ``request_timeout_s`` stays the floor, so small-block actions keep the
        frozen pilot behavior exactly. ``timeout_safety_factor`` absorbs serial
        CPU-queue contention and any small additive overhead not in the estimate,
        and ``timeout_maximum_s`` is a finite liveness ceiling shared with the
        PBFT reconstruction family. See ``liu_quorum_state_action_timeout_v1``.
        """
        if estimated_slowest_round_trip_s < 0:
            raise ValueError("estimated round-trip cannot be negative")
        scaled = self.timeout_safety_factor * estimated_slowest_round_trip_s
        return min(self.timeout_maximum_s, max(self.request_timeout_s, scaled))

    @classmethod
    def from_parameters(cls, node) -> "LiuQuorumRuntimeConfiguration":
        config = Parameters.LiuQuorum
        epoch_context = runtime_epoch_from_parameters(node, LiuConsensusProtocol.LIU_QUORUM, config)
        faults = tuple(
            ReplicaFault(
                int(item["node_id"]),
                ReplicaFaultPolicy(item["policy"]),
                float(item.get("delay_s", 0.0)),
            )
            for item in config.get("replica_faults", ())
        )
        return cls(
            epoch_context,
            float(config["signature_cycles_alpha"]),
            float(config["mac_cycles_beta"]),
            float(config["request_timeout_s"]),
            float(config.get("propagation_delay_s", 0.0)),
            faults,
            float(config.get("timeout_safety_factor", 2.0)),
            float(config.get("timeout_maximum_s", 320.0)),
        )
