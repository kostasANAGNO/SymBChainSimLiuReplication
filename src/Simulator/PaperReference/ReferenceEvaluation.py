"""LIU_PAPER_REFERENCE_MODE: paper-equation-only constraint/reward evaluation.

Given a paper state S = [chi, Upsilon, x, c, R] (Eq. 10) and an arbitrary paper action
A = [a, delta, S_B, T_I] (Eq. 11), compute the Liu constraints C1/C2/C3 (Eq. 12) and the
immediate reward (Eq. 13) using ONLY the paper equations and the Appendix-B analytical
consensus models. This module reads no discrete-event outcome: it is a pure deterministic
function of its inputs and explicitly-configured PAPER_UNDERSPECIFIED parameters.

Paper: Liu et al., IEEE TII 2019, DOI 10.1109/TII.2019.2897805. Equation citations inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters, paper_stake_gini
from PaperReference.AnalyticalConsensus import (
    AnalyticalConsensusInput,
    AnalyticalConsensusResult,
    LiuTransmissionUnitPolicy,
    ZyzzyvaAnalyticalPath,
)
from PaperReference.AnalyticalLiuQuorum import LiuQuorumAnalyticalModel
from PaperReference.AnalyticalPBFT import PBFTAnalyticalModel
from PaperReference.AnalyticalZyzzyva import ZyzzyvaAnalyticalModel

REFERENCE_MODE = "LIU_PAPER_REFERENCE_MODE"
REFERENCE_CORE_VERSION = "liu_reference_core_equations_v1"
DEFAULT_TRANSMISSION_UNIT_POLICY = LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1

_MODELS = {
    LiuConsensusProtocol.PBFT: PBFTAnalyticalModel(),
    LiuConsensusProtocol.ZYZZYVA: ZyzzyvaAnalyticalModel(),
    LiuConsensusProtocol.LIU_QUORUM: LiuQuorumAnalyticalModel(),
}


def _reference_tolerated_faults(protocol: LiuConsensusProtocol, k: int) -> int:
    """Eq. (9): F^0 = F^1 = floor((K-1)/3), F^2 = 0."""
    if protocol is LiuConsensusProtocol.LIU_QUORUM:
        return 0
    return (k - 1) // 3


def reference_consensus(
    *,
    producer_ids: Sequence[int],
    protocol: LiuConsensusProtocol,
    block_size_mb: float,
    block_interval_s: float,
    node_capabilities_ghz: Sequence[float],
    link_rates: LinkStateMatrix,
    malicious_count: int,
    params: LiuReferenceParameters,
    client_id: int | None = None,
    primary_id: int | None = None,
    transmission_unit_policy: LiuTransmissionUnitPolicy = DEFAULT_TRANSMISSION_UNIT_POLICY,
) -> AnalyticalConsensusResult:
    """Appendix-B analytical T_C,delta = T_D,delta + T_V,delta and T_F,delta = T_I + T_C,delta.

    Builds the K-aligned validator submatrix and capability vector from the paper state, then
    dispatches to the verified Appendix-B model. Roles default to a canonical assignment
    (client = lowest producer id, primary = next); the round-robin schedule is a runtime concern.
    """
    ids = tuple(sorted(int(i) for i in producer_ids))
    k = len(ids)
    if client_id is None:
        client_id = ids[0]
    if primary_id is None and protocol is not LiuConsensusProtocol.LIU_QUORUM:
        primary_id = next(i for i in ids if i != client_id)
    capabilities = tuple(float(node_capabilities_ghz[i]) for i in ids)
    submatrix = LinkStateMatrix.from_rows(
        tuple(
            tuple(None if a == b else link_rates.rate(a, b) for b in ids)
            for a in ids
        )
    )
    batch = 1 if protocol is LiuConsensusProtocol.LIU_QUORUM else params.pbft_zyzzyva_batch_m
    if protocol is LiuConsensusProtocol.ZYZZYVA:
        path = ZyzzyvaAnalyticalPath.FAST if malicious_count == 0 else ZyzzyvaAnalyticalPath.RECOVERY
        recovery = 0.0 if malicious_count == 0 else params.recovery_delay_s
    else:
        path = None
        recovery = 0.0
    analytical_input = AnalyticalConsensusInput(
        validator_ids=ids,
        validator_count_k=k,
        protocol=protocol,
        client_validator_id=client_id,
        primary_validator_id=primary_id,
        block_size_mb=block_size_mb,
        transaction_size_bytes=params.transaction_size_bytes,
        batch_size_m=batch,
        block_interval_s=block_interval_s,
        computational_capabilities_ghz=capabilities,
        directed_link_rates_mbps=submatrix,
        signature_verification_cycles_alpha=params.signature_verification_cycles_alpha,
        mac_operation_cycles_beta=params.mac_operation_cycles_beta,
        network_timeout_s=params.network_timeout_s,
        faulty_replica_count=malicious_count,
        zyzzyva_path=path,
        recovery_delay_s=recovery,
        transmission_unit_policy=transmission_unit_policy,
    )
    return _MODELS[protocol].evaluate(analytical_input)


@dataclass(frozen=True, slots=True)
class LiuReferenceEvaluation:
    reference_mode: str
    stake_gini: float
    geographic_gini: float
    stake_constraint_passed: bool          # Eq. (4)
    geographic_constraint_passed: bool     # Eq. (5)
    c1_decentralization_passed: bool       # Eq. (12) C1
    delivery_delay_s: float                # T_D,delta (Appendix B)
    validation_delay_s: float              # T_V,delta (Appendix B)
    consensus_latency_s: float             # T_C,delta = T_D + T_V (Eq. 7)
    finality_latency_s: float              # T_F,delta = T_I + T_C (Eq. 6)
    finality_limit_s: float                # omega * T_I (Eq. 8)
    c2_finality_passed: bool               # Eq. (12) C2
    malicious_count: int                   # f
    tolerated_faults: int                  # F^delta
    c3_security_passed: bool               # Eq. (12) C3
    feasible: bool                         # C1 AND C2 AND C3
    transaction_capacity: int              # floor(S_B / chi) (Eq. 1 numerator)
    throughput_tps: float                  # Omega (Eq. 1)
    reward: float                          # Eq. (13)


def evaluate_reference(
    *,
    # paper state S = [chi, Upsilon, x, c, R]
    node_stakes: Sequence[float],
    node_capabilities_ghz: Sequence[float],
    link_rates: LinkStateMatrix,
    geographic_gini: float,               # G(lambda), Eq. (3), from the spatial abstraction
    # paper action A = [a, delta, S_B, T_I]
    producer_mask: Sequence[int],
    protocol: LiuConsensusProtocol,
    block_size_mb: float,
    block_interval_s: float,
    # scenario
    malicious_count: int,                  # f (from threat scenario over the selected K)
    # parameters (Table I / PAPER_UNDERSPECIFIED, explicitly configured)
    params: LiuReferenceParameters,
    client_id: int | None = None,
    primary_id: int | None = None,
    transmission_unit_policy: LiuTransmissionUnitPolicy = DEFAULT_TRANSMISSION_UNIT_POLICY,
) -> LiuReferenceEvaluation:
    """Evaluate C1/C2/C3 (Eq. 12) and the Eq. (13) reward for a given S and A. Pure; no DES."""
    mask = tuple(int(v) for v in producer_mask)
    if any(v not in (0, 1) for v in mask):
        raise ValueError("producer_mask must be a binary vector (Eq. 11: a_n in {0,1})")
    producer_ids = tuple(n for n, v in enumerate(mask) if v == 1)
    k = len(producer_ids)
    if k < 1:
        raise ValueError("action must select at least one block producer (sum(a)=K)")

    # --- C1 decentralization (Eq. 2/3/4/5) ---
    stake_gini = paper_stake_gini([node_stakes[n] for n in producer_ids])
    if not (0.0 <= geographic_gini <= 1.0):
        raise ValueError("geographic_gini (Eq. 3) must be in [0,1]")
    stake_ok = stake_gini <= params.stake_gini_threshold_eta_s
    geo_ok = geographic_gini <= params.geographic_gini_threshold_eta_l
    c1 = stake_ok and geo_ok

    # --- C3 security (Eq. 9) ---
    tolerated = _reference_tolerated_faults(protocol, k)
    c3 = malicious_count <= tolerated

    # --- C2 finality (Eq. 6/7/8) via Appendix-B analytical model ---
    consensus = reference_consensus(
        producer_ids=producer_ids,
        protocol=protocol,
        block_size_mb=block_size_mb,
        block_interval_s=block_interval_s,
        node_capabilities_ghz=node_capabilities_ghz,
        link_rates=link_rates,
        malicious_count=min(malicious_count, tolerated),  # latency worst-case within F^delta
        params=params,
        client_id=client_id,
        primary_id=primary_id,
        transmission_unit_policy=transmission_unit_policy,
    )
    delivery_delay = consensus.delivery_delay_s        # T_D,delta (Appendix B)
    validation_delay = consensus.validation_delay_s    # T_V,delta (Appendix B)
    finality_latency = consensus.finality_delay_s      # T_F,delta = T_I + T_C,delta
    consensus_latency = consensus.consensus_delay_s
    finality_limit = params.finality_multiplier_omega * block_interval_s
    c2 = finality_latency <= finality_limit

    feasible = c1 and c2 and c3

    # --- reward Eq. (1)/(13) ---
    capacity = floor(block_size_mb * params.megabytes_to_bytes / params.transaction_size_bytes)
    throughput = capacity / block_interval_s
    reward = throughput if feasible else 0.0

    return LiuReferenceEvaluation(
        reference_mode=REFERENCE_MODE,
        stake_gini=stake_gini,
        geographic_gini=geographic_gini,
        stake_constraint_passed=stake_ok,
        geographic_constraint_passed=geo_ok,
        c1_decentralization_passed=c1,
        delivery_delay_s=delivery_delay,
        validation_delay_s=validation_delay,
        consensus_latency_s=consensus_latency,
        finality_latency_s=finality_latency,
        finality_limit_s=finality_limit,
        c2_finality_passed=c2,
        malicious_count=malicious_count,
        tolerated_faults=tolerated,
        c3_security_passed=c3,
        feasible=feasible,
        transaction_capacity=capacity,
        throughput_tps=throughput,
        reward=reward,
    )
