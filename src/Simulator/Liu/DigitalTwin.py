"""DES digital-twin single-action evaluation (pure).

Result types for the digital-twin single-action closed loop: the Liu constraints and reward
observed from a real DES execution. The analytical paper reference lives in ``PaperReference``
and is never consulted here.

Two rewards are produced and never share an ambiguous `reward` field:

  reward_des_liu_objective : Liu's Omega = floor(S_B/chi)/T_I gated by DES-observed C1/C2/C3.
  reward_des_realized      : DES finalized TPS gated by DES-observed C1/C2/C3 (diagnostic).

The DES-observed C1 uses Liu Eq.(2) stake Gini over the selected K and the explicit Eq.(3)
G(lambda); C2 uses the DES-observed consensus latency; C3 uses the actual selected faulty
count. This module is pure: it consumes already-measured DES quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from Liu.Protocol import LiuConsensusProtocol
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer
from Utils.DecentralizationMetrics import canonical_pairwise_gini

DES_REWARD_POLICY = "LIU_EQ13_DES_REALIZATION_finalized_tps_v1"
# Liu-compatible objective: Omega = floor(S_B/chi)/T_I, gated by DES-observed C1/C2/C3.
# Preserves the paper's objective function (see reward_semantics.md); T_C_DES enters only
# through C2, not through the reward numerator (that avoids the double-counting present in
# the historical reward_des_realized diagnostic).
DES_LIU_OBJECTIVE_POLICY = "liu_omega_gated_by_des_constraints_v1"


def des_tolerated_faults(protocol: LiuConsensusProtocol, validator_count: int) -> int:
    """Eq.(9): F^0 = F^1 = floor((K-1)/3), F^2 = 0 (Quorum)."""
    if protocol is LiuConsensusProtocol.LIU_QUORUM:
        return 0
    return (validator_count - 1) // 3


@dataclass(frozen=True, slots=True)
class LiuDesObservedRecord(CanonicalSerializable):
    stake_gini: float
    geographic_gini: float
    t_c_des_s: float
    t_f_des_s: float
    finality_limit_s: float
    offered_transactions: int
    included_transactions: int
    finalized_transactions: int
    measurement_duration_s: float
    offered_load_tps: float
    throughput_des_finalized: float
    malicious_validator_count: int
    tolerated_fault_count: int
    c1_decentralization_passed: bool
    c2_finality_passed: bool
    c3_security_passed: bool
    feasible: bool
    reward_des_realized: float
    reward_policy_version: str
    measurement_window_policy: str
    direct_tx_finalization_latency_s: float | None = None
    # Liu-compatible objective: Omega gated by DES-observed C1/C2/C3 (see reward_semantics.md).
    # Additive fields so historical golden JSONs remain valid; default 0.0 for records built
    # before this milestone landed.
    reward_des_liu_objective: float = 0.0
    throughput_paper_nominal: float = 0.0
    reward_des_liu_objective_policy_version: str = DES_LIU_OBJECTIVE_POLICY

    def to_dict(self) -> dict:
        return {
            "c1_decentralization_passed": self.c1_decentralization_passed,
            "c2_finality_passed": self.c2_finality_passed,
            "c3_security_passed": self.c3_security_passed,
            "direct_tx_finalization_latency_s": self.direct_tx_finalization_latency_s,
            "feasible": self.feasible,
            "finalized_transactions": self.finalized_transactions,
            "finality_limit_s": self.finality_limit_s,
            "geographic_gini": self.geographic_gini,
            "included_transactions": self.included_transactions,
            "malicious_validator_count": self.malicious_validator_count,
            "measurement_duration_s": self.measurement_duration_s,
            "measurement_window_policy": self.measurement_window_policy,
            "offered_load_tps": self.offered_load_tps,
            "offered_transactions": self.offered_transactions,
            "reward_des_liu_objective": self.reward_des_liu_objective,
            "reward_des_liu_objective_policy_version": self.reward_des_liu_objective_policy_version,
            "reward_des_realized": self.reward_des_realized,
            "reward_policy_version": self.reward_policy_version,
            "stake_gini": self.stake_gini,
            "t_c_des_s": self.t_c_des_s,
            "t_f_des_s": self.t_f_des_s,
            "throughput_des_finalized": self.throughput_des_finalized,
            "throughput_paper_nominal": self.throughput_paper_nominal,
            "tolerated_fault_count": self.tolerated_fault_count,
        }


MEASUREMENT_WINDOW_POLICY = "liu_single_epoch_target_height_finality_window_v1"


def evaluate_des_observed(
    *,
    selected_validator_stakes: Sequence[float],
    geographic_gini_lambda: float,
    stake_gini_threshold: float,
    geographic_gini_threshold: float,
    protocol: LiuConsensusProtocol,
    validator_count: int,
    malicious_validator_count: int,
    block_interval_s: float,
    finality_multiplier_omega: float,
    t_c_des_s: float,
    offered_transactions: int,
    included_transactions: int,
    finalized_transactions: int,
    measurement_duration_s: float,
    direct_tx_finalization_latency_s: float | None = None,
    throughput_paper_nominal: float = 0.0,
) -> LiuDesObservedRecord:
    """Operational C1/C2/C3 + Liu-Omega-gated reward from measured DES quantities (pure).

    `throughput_paper_nominal` is Liu's Omega = floor(S_B/chi)/T_I. The Liu-compatible DES
    reward (`reward_des_liu_objective`) reuses this Omega and gates it by DES C1/C2/C3, so
    T_C_DES enters the reward only through C2 (not the numerator). Historical
    `reward_des_realized` (single-epoch finalized TPS gated by DES C1/C2/C3) is retained
    unchanged as a DES_OPERATIONAL_METRIC diagnostic.
    """
    stake_gini = canonical_pairwise_gini(tuple(float(s) for s in selected_validator_stakes))
    geo = require_finite_number(geographic_gini_lambda, "geographic_gini_lambda", non_negative=True)
    t_c = require_finite_number(t_c_des_s, "t_c_des_s", non_negative=True)
    block_interval_s = require_finite_number(block_interval_s, "block_interval_s", positive=True)
    duration = require_finite_number(measurement_duration_s, "measurement_duration_s", positive=True)
    malicious = require_integer(malicious_validator_count, "malicious_validator_count", minimum=0)
    finalized = require_integer(finalized_transactions, "finalized_transactions", minimum=0)

    c1 = (stake_gini <= stake_gini_threshold) and (geo <= geographic_gini_threshold)
    tolerated = des_tolerated_faults(protocol, validator_count)
    c3 = malicious <= tolerated
    t_f_des = block_interval_s + t_c
    finality_limit = finality_multiplier_omega * block_interval_s
    c2 = t_f_des <= finality_limit

    throughput_des = finalized / duration
    feasible = c1 and c2 and c3
    reward_des = throughput_des if feasible else 0.0
    omega = require_finite_number(throughput_paper_nominal, "throughput_paper_nominal", non_negative=True)
    reward_liu = omega if feasible else 0.0

    return LiuDesObservedRecord(
        stake_gini=stake_gini,
        geographic_gini=geo,
        t_c_des_s=t_c,
        t_f_des_s=t_f_des,
        finality_limit_s=finality_limit,
        offered_transactions=require_integer(offered_transactions, "offered_transactions", minimum=0),
        included_transactions=require_integer(included_transactions, "included_transactions", minimum=0),
        finalized_transactions=finalized,
        measurement_duration_s=duration,
        offered_load_tps=offered_transactions / duration,
        throughput_des_finalized=throughput_des,
        malicious_validator_count=malicious,
        tolerated_fault_count=tolerated,
        c1_decentralization_passed=c1,
        c2_finality_passed=c2,
        c3_security_passed=c3,
        feasible=feasible,
        reward_des_realized=reward_des,
        reward_policy_version=DES_REWARD_POLICY,
        measurement_window_policy=MEASUREMENT_WINDOW_POLICY,
        direct_tx_finalization_latency_s=direct_tx_finalization_latency_s,
        reward_des_liu_objective=reward_liu,
        throughput_paper_nominal=omega,
        reward_des_liu_objective_policy_version=DES_LIU_OBJECTIVE_POLICY,
    )


@dataclass(frozen=True, slots=True)
class LiuDigitalTwinEvaluation(CanonicalSerializable):
    state: dict
    action: dict
    des_observed: LiuDesObservedRecord
    next_state: dict

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "des_observed": self.des_observed.to_dict(),
            "next_state": self.next_state,
            "state": self.state,
        }
