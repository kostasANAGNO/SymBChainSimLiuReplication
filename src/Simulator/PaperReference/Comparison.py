"""Paper-reference vs DES comparison for one action (pure, opt-in).

The DES (``Liu.DigitalTwin`` / ``run_single_action``) never calls this module. Analysis scripts
compute the paper reference for the same (S, A) explicitly and compare it with the DES result:

    des = run_single_action(snapshot, action, geo, ...)
    paper = paper_reference_for(snapshot, action, geo, params)
    paired = pair_with_paper_reference(des, paper)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from Liu.DigitalTwin import LiuDesObservedRecord, LiuDigitalTwinEvaluation
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.Serialization import CanonicalSerializable
from Liu.Threat import ThreatScenario
from PaperReference.AnalyticalConsensus import LiuTransmissionUnitPolicy
from PaperReference.ReferenceEvaluation import (
    DEFAULT_TRANSMISSION_UNIT_POLICY,
    LiuReferenceEvaluation,
    evaluate_reference,
)

if TYPE_CHECKING:
    from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import LiuSingleActionSnapshot
    from Liu.Action import LiuAction
    from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel

PAPER_REWARD_POLICY = "liu_reference_core_eq13_v1"


@dataclass(frozen=True, slots=True)
class LiuPaperReferenceRecord(CanonicalSerializable):
    stake_gini: float
    geographic_gini: float
    delivery_delay_s: float
    validation_delay_s: float
    consensus_latency_s: float
    finality_latency_s: float
    finality_limit_s: float
    transaction_capacity: int
    throughput_paper_nominal: float
    c1_decentralization_passed: bool
    c2_finality_passed: bool
    c3_security_passed: bool
    feasible: bool
    reward_paper_reference: float

    @classmethod
    def from_reference(cls, ev: LiuReferenceEvaluation) -> "LiuPaperReferenceRecord":
        return cls(
            ev.stake_gini,
            ev.geographic_gini,
            ev.delivery_delay_s,
            ev.validation_delay_s,
            ev.consensus_latency_s,
            ev.finality_latency_s,
            ev.finality_limit_s,
            ev.transaction_capacity,
            ev.throughput_tps,
            ev.c1_decentralization_passed,
            ev.c2_finality_passed,
            ev.c3_security_passed,
            ev.feasible,
            ev.reward,
        )

    def to_dict(self) -> dict:
        return {
            "c1_decentralization_passed": self.c1_decentralization_passed,
            "c2_finality_passed": self.c2_finality_passed,
            "c3_security_passed": self.c3_security_passed,
            "consensus_latency_s": self.consensus_latency_s,
            "delivery_delay_s": self.delivery_delay_s,
            "feasible": self.feasible,
            "finality_latency_s": self.finality_latency_s,
            "finality_limit_s": self.finality_limit_s,
            "geographic_gini": self.geographic_gini,
            "reward_paper_reference": self.reward_paper_reference,
            "stake_gini": self.stake_gini,
            "throughput_paper_nominal": self.throughput_paper_nominal,
            "transaction_capacity": self.transaction_capacity,
            "validation_delay_s": self.validation_delay_s,
        }


@dataclass(frozen=True, slots=True)
class LiuDigitalTwinDelta(CanonicalSerializable):
    consensus_latency_delta_s: float
    finality_delta_s: float
    throughput_delta_tps: float
    c1_agrees: bool
    c2_agrees: bool
    c3_agrees: bool
    reward_delta: float
    constraint_disagreement_reason: str | None
    # Delta on the Liu-compatible objective. Zero when paper and DES agree on feasibility;
    # nonzero exactly when the constraint verdicts disagree (same Omega both sides).
    reward_liu_objective_delta: float = 0.0

    @classmethod
    def between(cls, paper: LiuPaperReferenceRecord, des: LiuDesObservedRecord) -> "LiuDigitalTwinDelta":
        c1_agrees = paper.c1_decentralization_passed == des.c1_decentralization_passed
        c2_agrees = paper.c2_finality_passed == des.c2_finality_passed
        c3_agrees = paper.c3_security_passed == des.c3_security_passed
        disagreements = []
        if not c1_agrees:
            disagreements.append(f"C1 paper={paper.c1_decentralization_passed} des={des.c1_decentralization_passed}")
        if not c2_agrees:
            disagreements.append(f"C2 paper={paper.c2_finality_passed} des={des.c2_finality_passed}")
        if not c3_agrees:
            disagreements.append(f"C3 paper={paper.c3_security_passed} des={des.c3_security_passed}")
        return cls(
            consensus_latency_delta_s=des.t_c_des_s - paper.consensus_latency_s,
            finality_delta_s=des.t_f_des_s - paper.finality_latency_s,
            throughput_delta_tps=des.throughput_des_finalized - paper.throughput_paper_nominal,
            c1_agrees=c1_agrees,
            c2_agrees=c2_agrees,
            c3_agrees=c3_agrees,
            reward_delta=des.reward_des_realized - paper.reward_paper_reference,
            constraint_disagreement_reason="; ".join(disagreements) if disagreements else None,
            reward_liu_objective_delta=des.reward_des_liu_objective - paper.reward_paper_reference,
        )

    def to_dict(self) -> dict:
        return {
            "c1_agrees": self.c1_agrees,
            "c2_agrees": self.c2_agrees,
            "c3_agrees": self.c3_agrees,
            "consensus_latency_delta_s": self.consensus_latency_delta_s,
            "constraint_disagreement_reason": self.constraint_disagreement_reason,
            "finality_delta_s": self.finality_delta_s,
            "reward_delta": self.reward_delta,
            "reward_liu_objective_delta": self.reward_liu_objective_delta,
            "throughput_delta_tps": self.throughput_delta_tps,
        }


@dataclass(frozen=True, slots=True)
class LiuDigitalTwinPairedEvaluation(CanonicalSerializable):
    state: dict
    action: dict
    paper_reference: LiuPaperReferenceRecord
    des_observed: LiuDesObservedRecord
    delta: LiuDigitalTwinDelta
    next_state: dict

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "delta": self.delta.to_dict(),
            "des_observed": self.des_observed.to_dict(),
            "next_state": self.next_state,
            "paper_reference": self.paper_reference.to_dict(),
            "state": self.state,
        }


def paper_reference_for(
    snapshot: "LiuSingleActionSnapshot",
    action: "LiuAction",
    geographic_model: "ContinuousSpatialIntensityModel",
    params: LiuReferenceParameters,
    *,
    transmission_unit_policy: LiuTransmissionUnitPolicy = DEFAULT_TRANSMISSION_UNIT_POLICY,
) -> LiuPaperReferenceRecord:
    """Appendix-B reference for the same S0 and A0 that the DES executed (read-only, no DES)."""
    selected = set(action.validator_ids)
    mask = tuple(1 if i in selected else 0 for i in range(snapshot.node_count))
    threat = ThreatScenario(snapshot.node_count, snapshot.faulty_node_ids)
    oracle = evaluate_reference(
        node_stakes=snapshot.stakes_tokens,
        node_capabilities_ghz=snapshot.capabilities_ghz,
        link_rates=snapshot.link_matrix(),
        geographic_gini=geographic_model.geographic_gini(),
        producer_mask=mask,
        protocol=action.consensus_protocol,
        block_size_mb=action.block_size_mb,
        block_interval_s=action.block_interval_s,
        malicious_count=threat.malicious_validator_count(action.validator_ids),
        params=params,
        transmission_unit_policy=transmission_unit_policy,
    )
    return LiuPaperReferenceRecord.from_reference(oracle)


def pair_with_paper_reference(
    des: LiuDigitalTwinEvaluation,
    paper: LiuPaperReferenceRecord,
) -> LiuDigitalTwinPairedEvaluation:
    return LiuDigitalTwinPairedEvaluation(
        state=des.state,
        action=des.action,
        paper_reference=paper,
        des_observed=des.des_observed,
        delta=LiuDigitalTwinDelta.between(paper, des.des_observed),
        next_state=des.next_state,
    )
