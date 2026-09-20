"""Micro DRL ground-truth environment.

Label: MICRO_DRL_GROUND_TRUTH_ENVIRONMENT
This is an ALGORITHMIC TEST environment, NOT a Liu scientific result.
It is NOT the N=30/K=28 authoritative environment.

N=6, K=4 → C(6,4) = 15 validator sets.
3 protocols × 2 S_B × 2 T_I = 12 (protocol, S_B, T_I) configurations.
Total: 15 × 12 = 180 legal actions.

Population fixture (MICRO_VALIDATION_FIXTURE / OUR_RECONSTRUCTION):
  Stakes:      [4, 5, 6, 7, 8, 9]  (range 4-9, all subsets have Gini << 0.2)
  Caps (GHz):  [10, 11, 12, 13, 14, 15]
  Positions:   2x3 grid at [(0.2,0.0),(0.4,0.0),(0.6,0.0),(0.2,0.5),(0.4,0.5),(0.6,0.5)] km
  chi = 200 bytes (PAPER_EXACT Table I)

S_B choices: {1.0, 4.0} MB — chosen deterministically to span C2 sensitivity.
T_I choices: {0.5, 2.0} s  — chosen deterministically to span C2 boundary.
  Selection rule: smallest valid T_I from {0.5, 2.0} × {S_B 1.0, 4.0} that creates
                  at least one C2=False action at POOR_NETWORK (10 Mbps) and
                  one C2=True action at GOOD_NETWORK (100 Mbps).
  Label: MICRO_VALIDATION_FIXTURE

Network states (for DES):
  GOOD_NETWORK:   all links = 100 Mbps
  MEDIUM_NETWORK: all links = 55 Mbps
  POOR_NETWORK:   all links = 10 Mbps

offered_workload_tx = 100_000:
  Must exceed max block capacity to avoid empty-mempool at height-2.
  S_B=4MB at chi=200B: capacity = 4*1e6/200 = 20,000 tx/block.
  Height-1 bootstrap consumes at most one block capacity.
  100,000 >> 20,000 ensures height-2 always has pending transactions.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
# Also add dynamic_epoch dir so we can import liu_dynamic_env
_DYN = ROOT / "experiments" / "liu_replication" / "dynamic_epoch"
sys.path.insert(0, str(_DYN))

from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from liu_dynamic_env import DESFailureReason, LiuDynamicEpochEnv

# ── Micro fixture (OUR_RECONSTRUCTION / MICRO_VALIDATION_FIXTURE) ──────────────
N_MICRO = 6
K_MICRO = 4
CHI_BYTES = 200  # PAPER_EXACT

MICRO_STAKES = tuple(float(4 + i) for i in range(N_MICRO))        # [4,5,6,7,8,9]
MICRO_CAPS = tuple(float(10 + i) for i in range(N_MICRO))         # [10..15] GHz
MICRO_POSITIONS = (
    (0.2, 0.0), (0.4, 0.0), (0.6, 0.0),
    (0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
)  # 2×3 grid in 1km×1km region

# Liu paper reference parameters (PAPER_EXACT where marked)
MICRO_REF_PARAMS = LiuReferenceParameters(
    signature_verification_cycles_alpha=2_000_000.0,  # PAPER_AMBIGUOUS
    mac_operation_cycles_beta=1_000_000.0,             # PAPER_AMBIGUOUS
    network_timeout_s=100.0,
    finality_multiplier_omega=6.0,                     # PAPER_EXACT
    stake_gini_threshold_eta_s=0.2,                    # PAPER_EXACT
    geographic_gini_threshold_eta_l=0.3,               # PAPER_EXACT
    transaction_size_bytes=CHI_BYTES,
    recovery_delay_s=0.05,
)

# Geographic model (OUR_RECONSTRUCTION — paper specifies PPP, not parameters)
MICRO_GEO = ContinuousSpatialIntensityModel(
    planar_gradient_intensity(K_MICRO, 0.6), K_MICRO, lambda_form="planar_gradient s=0.6"
)

# ── S_B and T_I choices (MICRO_VALIDATION_FIXTURE) ────────────────────────────
# Selected to create C2 sensitivity at different network speeds.
# S_B=4MB at 10Mbps: T_C ~ 4s → C2 fails with T_I=0.5s (limit = 3.0s)
# S_B=1MB at 100Mbps: T_C ~ 0.4s → C2 passes with T_I=0.5s (limit = 3.0s)
MICRO_BLOCK_SIZES = (1.0, 4.0)      # MB
MICRO_BLOCK_INTERVALS = (0.5, 2.0)  # seconds (both multiples of 0.5s per LiuAction constraint)

# ── Network states ────────────────────────────────────────────────────────────
GOOD_LINKS = tuple(
    tuple(None if i == j else 100.0 for j in range(N_MICRO))
    for i in range(N_MICRO)
)
MEDIUM_LINKS = tuple(
    tuple(None if i == j else 55.0 for j in range(N_MICRO))
    for i in range(N_MICRO)
)
POOR_LINKS = tuple(
    tuple(None if i == j else 10.0 for j in range(N_MICRO))
    for i in range(N_MICRO)
)
NETWORK_STATES = {
    "GOOD_NETWORK": GOOD_LINKS,
    "MEDIUM_NETWORK": MEDIUM_LINKS,
    "POOR_NETWORK": POOR_LINKS,
}

# ── Action domain ─────────────────────────────────────────────────────────────

def all_micro_actions() -> list[dict]:
    """Enumerate all 180 micro actions deterministically.

    Order: validator_sets outer (C(6,4)=15), configs inner (12 = 3×2×2).
    """
    vsets = list(combinations(range(N_MICRO), K_MICRO))
    assert len(vsets) == 15

    protocols = [LiuConsensusProtocol.PBFT, LiuConsensusProtocol.ZYZZYVA, LiuConsensusProtocol.LIU_QUORUM]
    configs = [(p, sb, ti) for p in protocols for sb in MICRO_BLOCK_SIZES for ti in MICRO_BLOCK_INTERVALS]
    assert len(configs) == 12  # 3 protocols × 2 S_B × 2 T_I

    actions = []
    aid = 0
    for vset in vsets:
        mask = [1 if i in set(vset) else 0 for i in range(N_MICRO)]
        for proto, sb, ti in configs:
            actions.append({
                "action_id": aid,
                "protocol": proto.value,
                "block_size_mb": sb,
                "block_interval_s": ti,
                "validator_ids": list(vset),
                "validator_mask": mask,
                "_liu_action": LiuAction(N_MICRO, K_MICRO, tuple(vset), proto, sb, ti),
            })
            aid += 1
    assert len(actions) == 180, f"Expected 180, got {len(actions)}"
    return actions


_OFFERED_WORKLOAD = 100_000  # >> max block capacity (4MB / 200B = 20,000 tx)


def make_micro_env(link_rows: tuple, seed: int = 42) -> LiuDynamicEpochEnv:
    """Build a micro DES environment with given fixed link rates."""
    return LiuDynamicEpochEnv(
        node_count=N_MICRO,
        transaction_size_bytes=float(CHI_BYTES),
        stakes_tokens=MICRO_STAKES,
        capabilities_ghz=MICRO_CAPS,
        positions_km=MICRO_POSITIONS,
        initial_link_rows_mbps=link_rows,
        faulty_node_ids=(),
        reference_params=MICRO_REF_PARAMS,
        geo_model=MICRO_GEO,
        offered_workload_tx=_OFFERED_WORKLOAD,
        max_events=500_000,
        seed=seed,
    )


def state_dict_from_links(link_rows: tuple) -> dict:
    """Build a state dict for DRL encoders from a link rate matrix."""
    return {
        "chi_bytes": float(CHI_BYTES),
        "stakes": MICRO_STAKES,
        "positions_km": MICRO_POSITIONS,
        "capabilities_ghz": MICRO_CAPS,
        "link_rows_mbps": link_rows,
    }
