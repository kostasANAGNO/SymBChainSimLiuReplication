"""Shared setup for the Liu et al. (2019) Section VI figure replications.

Everything is evaluated by the SymBChainSim DES (no analytical shortcut):
reward R = Omega = floor(S_B/chi)/T_I if the DES reports C1 & C2 & C3, else 0.

Scale: N=30 / K=28 (same reduced population as drl/n30, OUR_RECONSTRUCTION).
Action grid: full Liu grid, S_B in {0.2, 0.4, ..., S_max} MB, T_I in {0.5, ..., 10} s,
delta in {PBFT, Zyzzyva, Quorum}, a = any of C(N, K) validator sets.
"""
from __future__ import annotations

import itertools
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIU_ROOT = ROOT / "experiments" / "liu_replication"
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
sys.path.insert(0, str(LIU_ROOT / "dynamic_epoch"))
sys.path.insert(0, str(LIU_ROOT / "drl"))
sys.path.insert(0, str(LIU_ROOT / "drl" / "n30"))

import train_n30 as base  # population + REF_PARAMS + env factory (reused, not modified)
from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from liu_dynamic_env import LiuDynamicEpochEnv

N, K = base.N, base.K
PROTOCOLS = (LiuConsensusProtocol.PBFT, LiuConsensusProtocol.ZYZZYVA, LiuConsensusProtocol.LIU_QUORUM)
PROTOCOL_NAMES = ("PBFT", "ZYZZYVA", "LIU_QUORUM")
T_I_MAX = 10.0
T_I_GRID = tuple(float(Decimal("0.5") * i) for i in range(1, 21))


def block_size_grid(s_max_mb: float) -> tuple[float, ...]:
    steps = int((Decimal(str(s_max_mb)) / Decimal("0.2")).to_integral_value())
    return tuple(float(Decimal("0.2") * i) for i in range(1, steps + 1))


VALIDATOR_SETS: tuple[tuple[int, ...], ...] = tuple(
    tuple(i for i in range(N) if i not in excluded)
    for excluded in itertools.combinations(range(N), N - K)
)  # 435 sets, index = action head `a`


def make_action(vset: int, proto: int, sb: float, ti: float) -> LiuAction:
    return LiuAction(N, K, VALIDATOR_SETS[vset], PROTOCOLS[proto], sb, ti)


def make_env(seed: int, *, chi_bytes: float = 200.0, omega: float = 6.0, capability_scale: float = 1.0, offered_tx: int = 100_000) -> LiuDynamicEpochEnv:
    """Dynamic DES environment (FSMC links evolve each epoch, never reset within a run)."""
    ref = base.REF_PARAMS
    if chi_bytes != ref.transaction_size_bytes or omega != ref.finality_multiplier_omega:
        ref = LiuReferenceParameters(
            signature_verification_cycles_alpha=ref.signature_verification_cycles_alpha,
            mac_operation_cycles_beta=ref.mac_operation_cycles_beta,
            network_timeout_s=ref.network_timeout_s,
            finality_multiplier_omega=omega,
            stake_gini_threshold_eta_s=ref.stake_gini_threshold_eta_s,
            geographic_gini_threshold_eta_l=ref.geographic_gini_threshold_eta_l,
            transaction_size_bytes=int(chi_bytes),
            recovery_delay_s=ref.recovery_delay_s,
        )
    caps = tuple(c * capability_scale for c in base.CAPABILITIES)
    env = LiuDynamicEpochEnv(
        node_count=N,
        transaction_size_bytes=float(chi_bytes),
        stakes_tokens=base.STAKES,
        capabilities_ghz=caps,
        positions_km=base.POSITIONS,
        initial_link_rows_mbps=base.INITIAL_LINK_ROWS,
        faulty_node_ids=(),
        reference_params=ref,
        geo_model=base.GEO,
        offered_workload_tx=offered_tx,
        max_events=2_000_000,
        seed=seed,
    )
    env.reset(seed=seed)
    return env
