"""N30 Validator Sensitivity Check.

Gate: N30_VALIDATOR_SELECTION_LEARNING_SIGNAL_VERIFIED

Evaluates all 435 validator sets (C(30,28) complementary sets) across three
representative FSMC link-rate states for each of the 9 action-domain configs.
Uses the paper reference formula (no DES) for tractability.

Purpose: confirm that validator-set choice creates a genuine reward learning
signal in the N=30/K=28 environment before committing to full DRL training.

Classification: OUR_RECONSTRUCTION / ALGORITHMIC_VALIDATION / PRE_TRAINING_GATE

Population: identical to Phase 3 (run_validator_selection.py) and Phase 4
(verify_phase4.py): cycling stakes 4..13, 5-col × 6-row grid positions,
capabilities 10..30 GHz, initial link rates 10+((i+j)%91) quantized to FSMC
levels {10, 55, 100} Mbps.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))

from Liu.ReferenceCore import LiuReferenceParameters
from PaperReference.ReferenceEvaluation import evaluate_reference
from Liu.LinkState import LinkStateMatrix
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.Protocol import LiuConsensusProtocol

# ── Population (Phase 3/4 fixture — OUR_RECONSTRUCTION) ──────────────────────
N = 30
K = 28
CHI_BYTES = 200

STAKES = tuple(float(4 + (i % 10)) for i in range(N))
CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(N))
POSITIONS = tuple((0.2 * (i % 5), 0.2 * (i // 5)) for i in range(N))

REF_PARAMS = LiuReferenceParameters(
    signature_verification_cycles_alpha=2_000_000.0,
    mac_operation_cycles_beta=1_000_000.0,
    network_timeout_s=100.0,
    finality_multiplier_omega=6.0,
    stake_gini_threshold_eta_s=0.2,
    geographic_gini_threshold_eta_l=0.3,
    transaction_size_bytes=CHI_BYTES,
    recovery_delay_s=0.05,
)

GEO = ContinuousSpatialIntensityModel(
    planar_gradient_intensity(K, 0.6), K,
    lambda_form="planar_gradient s=0.6",
)

# ── FSMC representative states (3 FSMC quantized levels) ─────────────────────
def _all_link_matrix(rate_mbps: float) -> LinkStateMatrix:
    rows = tuple(
        tuple(None if i == j else rate_mbps for j in range(N))
        for i in range(N)
    )
    return LinkStateMatrix.from_rows(rows)


FSMC_STATES = {
    "GOOD_100Mbps": _all_link_matrix(100.0),
    "MEDIUM_55Mbps": _all_link_matrix(55.0),
    "POOR_10Mbps": _all_link_matrix(10.0),
}

# ── 9 action-domain configs (top-3 per protocol from controlled sweep) ────────
ACTION_CONFIGS = [
    # LIU_QUORUM
    {"protocol": LiuConsensusProtocol.LIU_QUORUM, "block_size_mb": 3.4, "block_interval_s": 1.0},
    {"protocol": LiuConsensusProtocol.LIU_QUORUM, "block_size_mb": 6.8, "block_interval_s": 2.0},
    {"protocol": LiuConsensusProtocol.LIU_QUORUM, "block_size_mb": 5.0, "block_interval_s": 1.5},
    # PBFT
    {"protocol": LiuConsensusProtocol.PBFT, "block_size_mb": 7.0, "block_interval_s": 3.0},
    {"protocol": LiuConsensusProtocol.PBFT, "block_size_mb": 5.8, "block_interval_s": 2.5},
    {"protocol": LiuConsensusProtocol.PBFT, "block_size_mb": 4.6, "block_interval_s": 2.0},
    # ZYZZYVA
    {"protocol": LiuConsensusProtocol.ZYZZYVA, "block_size_mb": 6.2, "block_interval_s": 2.5},
    {"protocol": LiuConsensusProtocol.ZYZZYVA, "block_size_mb": 7.4, "block_interval_s": 3.0},
    {"protocol": LiuConsensusProtocol.ZYZZYVA, "block_size_mb": 1.2, "block_interval_s": 0.5},
]


def enumerate_validator_sets() -> list[tuple[int, ...]]:
    """All C(30,2) = 435 complementary validator sets."""
    all_nodes = set(range(N))
    sets = []
    for i, j in itertools.combinations(range(N), 2):
        excluded = {i, j}
        validators = tuple(sorted(all_nodes - excluded))
        sets.append(validators)
    assert len(sets) == 435, f"expected 435, got {len(sets)}"
    return sets


def evaluate_one(
    validator_ids: tuple[int, ...],
    link_rates: LinkStateMatrix,
    protocol: LiuConsensusProtocol,
    block_size_mb: float,
    block_interval_s: float,
) -> dict:
    mask = tuple(1 if i in set(validator_ids) else 0 for i in range(N))
    geo_gini = GEO.geographic_gini()
    ref = evaluate_reference(
        node_stakes=STAKES,
        node_capabilities_ghz=CAPABILITIES,
        link_rates=link_rates,
        geographic_gini=geo_gini,
        producer_mask=mask,
        protocol=protocol,
        block_size_mb=block_size_mb,
        block_interval_s=block_interval_s,
        malicious_count=0,
        params=REF_PARAMS,
    )
    return {
        "c1": bool(ref.c1_decentralization_passed),
        "c2": bool(ref.c2_finality_passed),
        "c3": bool(ref.c3_security_passed),
        "stake_gini": float(ref.stake_gini),
        "t_c_s": float(ref.consensus_latency_s),
        "reward": float(ref.reward),
    }


def main() -> dict:
    validator_sets = enumerate_validator_sets()
    geo_gini = GEO.geographic_gini()

    print(f"N={N}, K={K}, validator sets: {len(validator_sets)}, configs: {len(ACTION_CONFIGS)}")
    print(f"Geographic Gini: {geo_gini:.4f} (threshold 0.3: {'PASS' if geo_gini <= 0.3 else 'FAIL'})")
    print()

    gate_pass = False
    all_results = {}

    for state_name, link_matrix in FSMC_STATES.items():
        print(f"=== {state_name} ===")
        all_results[state_name] = {}

        for cfg in ACTION_CONFIGS:
            proto = cfg["protocol"]
            sb = cfg["block_size_mb"]
            ti = cfg["block_interval_s"]
            cfg_key = f"{proto.value}_SB{sb}_TI{ti}"

            rewards = []
            c1_count = 0
            c2_count = 0
            c3_count = 0
            feasible_count = 0
            stake_ginis = []
            t_c_values = []

            for vs in validator_sets:
                r = evaluate_one(vs, link_matrix, proto, sb, ti)
                rewards.append(r["reward"])
                stake_ginis.append(r["stake_gini"])
                t_c_values.append(r["t_c_s"])
                if r["c1"]:
                    c1_count += 1
                if r["c2"]:
                    c2_count += 1
                if r["c3"]:
                    c3_count += 1
                if r["c1"] and r["c2"] and r["c3"]:
                    feasible_count += 1

            reward_min = min(rewards)
            reward_max = max(rewards)
            reward_range = reward_max - reward_min
            has_signal = reward_range > 0

            if has_signal:
                gate_pass = True

            # Distinct reward values (how many unique reward levels)
            distinct_rewards = sorted(set(rewards))

            all_results[state_name][cfg_key] = {
                "protocol": proto.value,
                "block_size_mb": sb,
                "block_interval_s": ti,
                "n_validator_sets": len(validator_sets),
                "c1_pass_count": c1_count,
                "c1_pass_rate": round(c1_count / len(validator_sets), 4),
                "c2_pass_count": c2_count,
                "c2_pass_rate": round(c2_count / len(validator_sets), 4),
                "c3_pass_count": c3_count,
                "feasible_count": feasible_count,
                "feasible_rate": round(feasible_count / len(validator_sets), 4),
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_range": reward_range,
                "distinct_reward_values": distinct_rewards,
                "stake_gini_min": round(min(stake_ginis), 5),
                "stake_gini_max": round(max(stake_ginis), 5),
                "stake_gini_range": round(max(stake_ginis) - min(stake_ginis), 5),
                "t_c_min_s": round(min(t_c_values), 4),
                "t_c_max_s": round(max(t_c_values), 4),
                "has_learning_signal": has_signal,
            }

            signal_tag = "YES" if has_signal else "NO "
            print(
                f"  {cfg_key:40s}  "
                f"C1={c1_count}/435  C2={c2_count}/435  feasible={feasible_count}/435  "
                f"R=[{reward_min:.0f},{reward_max:.0f}]  "
                f"signal={signal_tag}"
            )
        print()

    print(f"=== GATE: N30_VALIDATOR_SELECTION_LEARNING_SIGNAL_VERIFIED ===")
    if gate_pass:
        print("RESULT: AWARDED")
        print("Evidence: at least one (FSMC state, config) pair has reward variance > 0,")
        print("confirming that validator-set choice is a genuine learnable dimension.")
    else:
        print("RESULT: NOT AWARDED")
        print("FAIL: No reward variance found across validator sets for any (state, config) pair.")

    out = {
        "schema": "n30_validator_sensitivity_v1",
        "gate": "N30_VALIDATOR_SELECTION_LEARNING_SIGNAL_VERIFIED",
        "gate_awarded": gate_pass,
        "label": "OUR_RECONSTRUCTION / ALGORITHMIC_VALIDATION / PRE_TRAINING_GATE",
        "method": "paper_reference_formula (no DES)",
        "n": N,
        "k": K,
        "chi_bytes": CHI_BYTES,
        "n_validator_sets": len(validator_sets),
        "n_configs": len(ACTION_CONFIGS),
        "geographic_gini": round(geo_gini, 4),
        "fsmc_states_tested": list(FSMC_STATES.keys()),
        "results": all_results,
    }

    out_path = HERE / "validator_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten: {out_path}")
    return out


if __name__ == "__main__":
    main()
