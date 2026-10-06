"""Single-action DES digital-twin closed loop (PBFT, Zyzzyva and LiuQuorum).

Assembles a real SymBChainSim runtime (Engine.EventQueue.Queue + real Node objects +
Network + TransactionFactory + the Liu runtime protocols) from an explicit S0 snapshot,
applies one Liu action A0, drives the ACTUAL event engine to one finalized height, measures
the DES outcome, and returns the DES-observed constraints/reward plus the next state S1.

The Appendix-B analytical reference is NOT computed here; compare against it explicitly with
``PaperReference.Comparison`` when needed.

Epoch 0 is bootstrapped with PBFT regardless of the action's protocol.
Network evolution is disabled for the controlled single-action test: R1 = R0
(NETWORK_EVOLUTION_DISABLED_FOR_CONTROLLED_SINGLE_ACTION_TEST); this is NOT Liu's FSMC run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean
from types import SimpleNamespace
from typing import Sequence

from Chain.Block import Block
from Chain.Consensus.LiuRuntime.Common.ActionApplicator import LiuRuntimeActionApplicator
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.ProtocolFactory import (
    LiuRuntimeProtocolFactory,
    LiuRuntimeProtocolSettings,
)
from Chain.Consensus.LiuRuntime.Common.StateBuilder import LiuRuntimeStateBuilder
from Chain.Network import Network
from Chain.Node import Node
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.ValidatorSet import ValidatorSet
from Engine.EventQueue import Queue
from Engine.Handler import handle_event
from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel
from Liu.DigitalTwin import LiuDigitalTwinEvaluation, evaluate_des_observed
from Liu.EpochConfiguration import EpochConfiguration
from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters, nominal_throughput_tps
from Liu.Spatial import SpatialProfileSet
from Liu.State import LiuState
from Liu.Threat import ThreatScenario
from Parameters import Parameters
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector

PBFT_DES_RUNTIME_VERIFIED = "PBFT_DES_RUNTIME_VERIFIED"
ZYZZYVA_DES_RUNTIME_PENDING = "ZYZZYVA_DES_RUNTIME_PENDING"
QUORUM_DES_RUNTIME_PENDING = "QUORUM_DES_RUNTIME_PENDING"
NETWORK_EVOLUTION_LABEL = "NETWORK_EVOLUTION_DISABLED_FOR_CONTROLLED_SINGLE_ACTION_TEST"


class C2DeadlineExceeded(Exception):
    """DES simulated time exceeded the Liu C2 finality deadline (omega * T_I).

    Classification: AUTHORITATIVE Liu C2 failure.
    Liu defines: T_F = T_I + T_C <= omega * T_I  (Eq. 6, 8).
    When this is raised, the DES observed T_F > omega * T_I → C2=False, reward=0.
    This is the primary C2 failure mode for DRL training data.
    """


class DESQueueExhausted(Exception):
    """DES event queue drained before height finalized without hitting the C2 deadline.

    Classification: IMPLEMENTATION_GUARD / OUR_RECONSTRUCTION.
    Root cause: PBFT maximum_timeout_view_changes cap (OUR_RECONSTRUCTION — not Liu-specified).
    This is NOT an authoritative Liu C2 failure. For DRL: label as RUNTIME_GUARD_FAILURE;
    do not treat as training data unless independently verified to imply C2 deadline exceeded.
    """


@dataclass(frozen=True)
class LiuSingleActionSnapshot:
    """Explicit deterministic S0 for the digital twin (no hidden RNG)."""

    node_count: int
    transaction_size_bytes: float
    stakes_tokens: tuple[float, ...]
    capabilities_ghz: tuple[float, ...]
    positions_km: tuple[tuple[float, float], ...]
    link_rows_mbps: tuple[tuple[float | None, ...], ...]
    faulty_node_ids: tuple[int, ...]
    region_width_km: float = 1.0
    region_height_km: float = 1.0
    epoch0_validator_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = self.node_count
        for name, seq in (
            ("stakes_tokens", self.stakes_tokens),
            ("capabilities_ghz", self.capabilities_ghz),
            ("positions_km", self.positions_km),
            ("link_rows_mbps", self.link_rows_mbps),
        ):
            if len(seq) != n:
                raise ValueError(f"{name} must have exactly node_count={n} entries")
        if any(len(row) != n for row in self.link_rows_mbps):
            raise ValueError("link_rows_mbps must be an N x N matrix")
        for i in range(n):
            if self.link_rows_mbps[i][i] is not None:
                raise ValueError("link_rows_mbps diagonal must be None (no self-link)")

    def link_matrix(self) -> LinkStateMatrix:
        return LinkStateMatrix.from_rows(self.link_rows_mbps)

    def spatial_profiles(self) -> SpatialProfileSet:
        return SpatialProfileSet.from_coordinates(
            self.positions_km,
            region_width_km=self.region_width_km,
            region_height_km=self.region_height_km,
        )

    def node_profiles(self) -> NodeProfileSet:
        return NodeProfileSet(
            tuple(
                NodeProfile(i, float(self.stakes_tokens[i]), float(self.capabilities_ghz[i]))
                for i in range(self.node_count)
            )
        )


@dataclass
class _AssembledRuntime:
    nodes: list
    queue: Queue
    applicator: LiuRuntimeActionApplicator
    state_builder: LiuRuntimeStateBuilder
    epoch0_context: LiuRuntimeEpochContext
    threat: ThreatScenario


def _assemble(
    snapshot: LiuSingleActionSnapshot,
    action: LiuAction,
    *,
    signature_cycles_alpha: float,
    mac_cycles_beta: float,
    request_timeout_s: float,
    propagation_delay_s: float,
) -> _AssembledRuntime:
    n = snapshot.node_count
    Parameters.simulation = {"event_id": 0, "events": {}, "debugging_mode": False}
    Parameters.application = {"Nn": n, "transaction_model": "global"}
    Parameters.data = {"base_block_size": 0.0}
    Parameters.network = {"gossip": True}
    InstrumentationCollector.reset()
    LiuRuntimeInstrumentationCollector.reset()
    TransactionFactory.global_mempool = deque()
    TransactionFactory.depth_removed = -1
    TransactionFactory.produced_tx = 0

    epoch0_validators = snapshot.epoch0_validator_ids or action.validator_ids
    validator_set = ValidatorSet(tuple(epoch0_validators))
    profiles = snapshot.node_profiles()
    links = snapshot.link_matrix()
    threat = ThreatScenario(n, snapshot.faulty_node_ids)
    epoch0 = EpochConfiguration(
        0,
        validator_set,
        LiuConsensusProtocol.PBFT,
        action.block_size_mb,
        action.block_interval_s,
    )
    epoch0_context = LiuRuntimeEpochContext(epoch0, profiles, links, threat)

    # Zyzzyva Configuration cross-checks that faulty_replica_count equals the count of
    # threat-scenario malicious nodes actually inside the action's selected validator set.
    zyzzyva_faulty_selected = len(set(snapshot.faulty_node_ids) & set(action.validator_ids))
    settings = LiuRuntimeProtocolSettings(
        signature_cycles_alpha=signature_cycles_alpha,
        mac_cycles_beta=mac_cycles_beta,
        request_timeout_s=request_timeout_s,
        propagation_delay_s=propagation_delay_s,
        zyzzyva_faulty_replica_count=zyzzyva_faulty_selected,
    )
    factories = {protocol: LiuRuntimeProtocolFactory(settings) for protocol in LiuConsensusProtocol}

    queue = Queue()
    nodes = [Node(i, queue, validator_set, profiles.profile_for(i)) for i in range(n)]
    Network.nodes = nodes
    Network.received = {node: set() for node in nodes}
    TransactionFactory.nodes = nodes

    genesis = Block(depth=0, id=7, previous=-1, size=0.0)
    genesis.extra_data = {"configuration_depth": 0, "round": -1}
    pbft_factory = factories[LiuConsensusProtocol.PBFT]
    for node in nodes:
        node.blockchain = [genesis.copy()]
        node.reconfiguration_state.configuration = SimpleNamespace(
            block_size=action.block_size_mb, block_time=action.block_interval_s
        )
        node.cp = pbft_factory.create(node, epoch0_context)
    for node in nodes:
        node.cp.init(0.0, 0)

    applicator = LiuRuntimeActionApplicator(nodes, epoch0_context, factories)
    spatial = snapshot.spatial_profiles()
    state_builder = LiuRuntimeStateBuilder(
        spatial,
        snapshot.transaction_size_bytes / 1_000_000.0,
        geographic_model=None,
    )
    return _AssembledRuntime(nodes, queue, applicator, state_builder, epoch0_context, threat)


def _run_to_height(
    assembled: _AssembledRuntime,
    target_height: int,
    max_events: int,
    simulated_time_limit: float | None = None,
    consensus_epoch_id: int | None = None,
    consensus_protocol: str | None = None,
) -> float:
    """Drive the event engine until height is finalized, with an optional Liu C2 deadline.

    simulated_time_limit (optional): if an event's timestamp exceeds this value before
    consensus finality is recorded, raises C2DeadlineExceeded.  This is the AUTHORITATIVE Liu
    C2 check (T_F = T_I + T_C <= omega * T_I, i.e. finality_time <= boundary + omega*T_I).

    consensus_epoch_id + consensus_protocol (optional, both required): when provided, the
    termination condition is an instrumentation-based ProtocolFinalityRecord for that epoch
    and height, rather than all N nodes reaching the target depth.  This is REQUIRED for the
    measurement epoch because:
      - ProtocolFinalityRecord is created when the first validator commits (quorum guaranteed).
      - Waiting for all N nodes is wrong: non-validator gossip delivery can cross the C2
        deadline AFTER validator consensus has already completed within the deadline.
      - The simulated_time_limit must still fire when TRUE consensus failure occurs (no
        finality record before the deadline), preserving the Liu C2 operational deadline.
    """
    use_finality_record = (
        consensus_epoch_id is not None and consensus_protocol is not None
    )
    last_time = 0.0
    for _ in range(max_events):
        if use_finality_record:
            # Authoritative termination: first ProtocolFinalityRecord for this epoch/height.
            # PBFT safety: any single local commit implies quorum (2f+1 commits received).
            if any(
                r.epoch_id == consensus_epoch_id
                and r.height == target_height
                and r.protocol == consensus_protocol
                for r in LiuRuntimeInstrumentationCollector.protocol_finalities
            ):
                return last_time
        else:
            if all(node.last_block.depth == target_height for node in assembled.nodes):
                return last_time
        if assembled.queue.size() == 0:
            raise DESQueueExhausted(
                f"event queue exhausted before height {target_height} finalized "
                "(IMPLEMENTATION_GUARD: PBFT view-change cap is OUR_RECONSTRUCTION, not Liu-specified)"
            )
        event = assembled.queue.pop_next_event()
        if simulated_time_limit is not None and event.time > simulated_time_limit:
            raise C2DeadlineExceeded(
                f"simulated time {event.time:.3f}s > Liu C2 deadline {simulated_time_limit:.3f}s "
                f"(T_F = T_I + T_C <= omega * T_I; PAPER_EXACT Eq.8)"
            )
        handle_event(event)
        last_time = max(last_time, event.time)
    raise RuntimeError(
        f"runtime did not finalize height {target_height} within {max_events} events "
        "(IMPLEMENTATION_GUARD: max_events is OUR_RECONSTRUCTION, not Liu-specified)"
    )


_RUNTIME_PROTOCOL_NAMES: dict = {
    LiuConsensusProtocol.PBFT: "LiuPBFT",
    LiuConsensusProtocol.ZYZZYVA: "LiuZyzzyva",
    LiuConsensusProtocol.LIU_QUORUM: "LiuQuorum",
}


def _height_finality_record(epoch_id: int, height: int, protocol_name: str = "LiuPBFT"):
    matches = [
        record
        for record in LiuRuntimeInstrumentationCollector.protocol_finalities
        if record.epoch_id == epoch_id and record.height == height and record.protocol == protocol_name
    ]
    digests = {record.block_digest for record in matches}
    if len(digests) != 1:
        raise RuntimeError(
            f"expected exactly one protocol-finalized digest at epoch {epoch_id} height {height}"
            f" for {protocol_name}, got {len(digests)}"
        )
    return matches[0]


def _finalized_block(nodes, height: int) -> Block:
    for node in nodes:
        if node.last_block.depth == height:
            return node.last_block
    raise RuntimeError(f"no node holds a finalized block at height {height}")


def run_single_action(
    snapshot: LiuSingleActionSnapshot,
    action: LiuAction,
    geographic_model: ContinuousSpatialIntensityModel,
    *,
    reference_params: LiuReferenceParameters,
    stake_gini_threshold: float,
    geographic_gini_threshold: float,
    finality_multiplier_omega: float,
    signature_cycles_alpha: float,
    mac_cycles_beta: float,
    request_timeout_s: float = 100.0,
    propagation_delay_s: float = 0.0,
    offered_workload_tx: int = 2000,
    workload_tx_size_mb: float | None = None,
    max_events: int = 200_000,
) -> LiuDigitalTwinEvaluation:
    """Execute S0 -> A0 -> real DES -> measured outcome -> DES rewards -> S1."""
    if action.consensus_protocol not in (
        LiuConsensusProtocol.PBFT,
        LiuConsensusProtocol.ZYZZYVA,
        LiuConsensusProtocol.LIU_QUORUM,
    ):
        raise ValueError("single-action closed loop authoritative protocols: PBFT, Zyzzyva, Quorum")
    if action.node_count != snapshot.node_count:
        raise ValueError("action N must equal snapshot node_count")

    tx_size_mb = (
        workload_tx_size_mb
        if workload_tx_size_mb is not None
        else snapshot.transaction_size_bytes / 1_000_000.0
    )
    assembled = _assemble(
        snapshot,
        action,
        signature_cycles_alpha=signature_cycles_alpha,
        mac_cycles_beta=mac_cycles_beta,
        request_timeout_s=request_timeout_s,
        propagation_delay_s=propagation_delay_s,
    )

    # Saturated controlled workload: enough transactions to fill blocks for both heights.
    for tx_id in range(offered_workload_tx):
        TransactionFactory.global_mempool.append(Transaction(0, tx_id, 0.0, tx_size_mb))

    # Bootstrap epoch 0 to a finalized height 1 (establishes S0's current chain).
    _run_to_height(assembled, 1, max_events)
    state0 = assembled.state_builder.build(
        assembled.epoch0_context,
        observation_time=0.0,
        is_initial_state=True,
    ).state

    # Apply A0 at the finalized height-1 boundary; the epoch-1 action IS applied to the DES.
    boundary_time = _height_finality_record(0, 1).finality_time
    offered_before = len([tx for tx in TransactionFactory.global_mempool if not tx.processed])
    assembled.applicator.apply(action, current_finalized_height=1, time=boundary_time)

    # Liu C2 deadline: T_F = T_I + T_C <= omega * T_I  →  finality_time <= boundary + omega*T_I.
    # This is the AUTHORITATIVE operational deadline (PAPER_EXACT Eq. 8, Table I: omega=6).
    c2_deadline_s = boundary_time + finality_multiplier_omega * action.block_interval_s

    # Execute the measurement epoch.  Termination uses the ProtocolFinalityRecord
    # (instrumentation-based consensus finality), NOT all-N-nodes reaching height 2.
    # Rationale: non-validator gossip delivery can cross the deadline after validator
    # consensus has completed within it; only validator commit triggers finality.
    # The simulated_time_limit still enforces C2: if no finality record exists before
    # c2_deadline_s, the next event crossing that boundary raises C2DeadlineExceeded.
    # DESQueueExhausted propagates to the caller (implementation guard, labeled separately).
    _run_to_height(
        assembled, 2, max_events,
        simulated_time_limit=c2_deadline_s,
        consensus_epoch_id=1,
        consensus_protocol=_RUNTIME_PROTOCOL_NAMES[action.consensus_protocol],
    )

    finality = _height_finality_record(1, 2, _RUNTIME_PROTOCOL_NAMES[action.consensus_protocol])
    t_c_des = finality.finality_time - finality.request_sent_at
    end_time = finality.finality_time
    duration = end_time - boundary_time
    block = _finalized_block(assembled.nodes, 2)
    finalized_tx = len(block.transactions)
    direct_latency = (
        fmean(finality.finality_time - tx.original_creation_time for tx in block.transactions)
        if block.transactions
        else None
    )

    # Liu's Omega = floor(S_B/chi)/T_I (Eq. 1); gated below by the DES-observed C1/C2/C3.
    nominal_omega = nominal_throughput_tps(action.block_size_mb, action.block_interval_s, reference_params)

    des = evaluate_des_observed(
        selected_validator_stakes=[snapshot.stakes_tokens[i] for i in action.validator_ids],
        geographic_gini_lambda=geographic_model.geographic_gini(),
        stake_gini_threshold=stake_gini_threshold,
        geographic_gini_threshold=geographic_gini_threshold,
        protocol=action.consensus_protocol,
        validator_count=action.validator_count,
        malicious_validator_count=assembled.threat.malicious_validator_count(action.validator_ids),
        block_interval_s=action.block_interval_s,
        finality_multiplier_omega=finality_multiplier_omega,
        t_c_des_s=t_c_des,
        offered_transactions=offered_before,
        included_transactions=finalized_tx,
        finalized_transactions=finalized_tx,
        measurement_duration_s=duration,
        direct_tx_finalization_latency_s=direct_latency,
        throughput_paper_nominal=nominal_omega,
    )

    # S1: static Upsilon/x/c; R1 = R0 (network evolution disabled); chi' empirical from the epoch.
    empirical_chi_bytes = (
        fmean(tx.size for tx in block.transactions) * 1_000_000.0
        if block.transactions
        else snapshot.transaction_size_bytes
    )
    state1 = LiuState(
        empirical_chi_bytes,
        state0.stakes_tokens,
        state0.spatial_profiles,
        state0.computational_capabilities_ghz,
        snapshot.link_matrix(),
    )
    next_state = {
        **state1.to_dict(),
        "chi_source": "epoch_empirical_mean" if block.transactions else "configured_workload_mean",
        "configured_chi_bytes": snapshot.transaction_size_bytes,
        "network_evolution": NETWORK_EVOLUTION_LABEL,
        "state_hash": state1.deterministic_hash(),
    }

    return LiuDigitalTwinEvaluation(
        state={**state0.to_dict(), "state_hash": state0.deterministic_hash()},
        action=action.to_dict(),
        des_observed=des,
        next_state=next_state,
    )
