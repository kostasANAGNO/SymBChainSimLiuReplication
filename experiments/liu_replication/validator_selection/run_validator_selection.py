"""REDUCED_EXHAUSTIVE_VALIDATOR_VALIDATION_ENVIRONMENT
N=30, K=28: enumerate all C(30,28)=C(30,2)=435 validator sets.

Purpose: demonstrate that the validator-selection action dimension `a` causally changes
C1 (stake Gini, geographic Gini), consensus timing (T_C_DES), C2, C3, and reward.

Population: deterministic N=30 with stakes cycling 4,5,...,13,4,5,...,13,4,5,...,13
(OUR_RECONSTRUCTION: base=4, cycle=10 chosen so that population stake Gini ≈ 0.19,
which places ~51/435 validator subsets above η_s=0.2 and ~384/435 below — achieving
meaningful C1 stake-Gini discrimination across the 435 sets).

Fixed configuration:  S_B=1.0 MB, T_I=2.0 s (always feasible at K=28, f=0, c≥10 GHz).
                      Three protocols are run for each of the 435 validator sets.
Main experiment:      435 × 3 protocols = 1305 DES runs (f=0).
Security tests:       6 additional runs with f=1 malicious node (one included, one excluded).

Checkpoint: results are written atomically after every completed action.
Resume: re-launch after interruption — already-SUCCESS action_ids are skipped.

Classification: REDUCED_EXHAUSTIVE_VALIDATOR_VALIDATION_ENVIRONMENT (not a Liu N=100,K=21 run)

Stake Gini constraint note: with K=28/N=30, the validator set covers 93% of the population,
so stake Gini varies little across the 435 sets (~±0.013 around population Gini). The cycling
stake assignment was chosen to centre this range around η_s=0.2, producing ~11.7% failing sets.
Without this calibration, all sets either all-pass (uniform stakes) or all-fail (1..30 stakes).
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
sys.path.insert(0, str(Path(__file__).parent.parent / "controlled_sweep"))  # for run_controlled_smoke

from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot,
    run_single_action,
)
from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from PaperReference.Comparison import paper_reference_for
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

# ── Population parameters ──────────────────────────────────────────────────────
N = 30
K = 28
CHI_BYTES = 200
FIXED_SEED = 0
LAMBDA_SLOPE = 0.6
ZYZZYVA_RECOVERY_DELAY_S = 0.05

# Fixed (S_B, T_I) configuration — in Liu grid, comfortably feasible.
FIXED_SB = 1.0   # MB — 1.0 ∈ {0.2*i; i=1..40}
FIXED_TI = 2.0   # s  — 2.0 ∈ {0.5*i; i=1..20}

# ── Deterministic N=30 population ────────────────────────────────────────────
# Stakes: cycling 4..13 (OUR_RECONSTRUCTION: population Gini ≈ 0.19, which centres the
# K=28 validator-subset Ginis around η_s=0.2, giving ~51/435 failing and ~384/435 passing)
STAKES = tuple(float(4 + (i % 10)) for i in range(N))

# Capabilities: 10+(i%21) → range 10..30 GHz (same formula as smoke/full sweep)
CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(N))

# Positions: 5-column × 6-row grid in 1×1 km² (0.2 km spacing)
POSITIONS = tuple((0.2 * (i % 5), 0.2 * (i // 5)) for i in range(N))

# Link rates: 10+((i+j)%91) → range 10..100 Mbps (same formula as smoke/full sweep)
LINK_ROWS = tuple(
    tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N))
    for i in range(N)
)

TX_SIZE_MB = float(CHI_BYTES) / 1_000_000.0
OFFERED_WORKLOAD_TX = 30_000   # >> 2 blocks × K blocks × 1.0 MB / 200 B = 10 000 tx

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent
CSV_PATH = OUT_DIR / "validator_selection_results.csv"

CSV_COLUMNS = (
    "action_id",
    "experiment_type",     # "main" or "security"
    "protocol",
    "excluded_nodes",      # comma-separated pair of excluded node IDs (for C(30,2) enumeration)
    "validator_ids",       # comma-separated sorted validator IDs
    "faulty_node_ids",     # comma-separated; empty if f=0
    "block_size_mb",
    "block_interval_s",
    # C1 measures
    "stake_gini_paper",
    "geo_gini",
    "paper_c1",
    "des_c1",
    # C2 measures
    "paper_t_c",
    "paper_t_f",
    "paper_c2",
    "des_t_c",
    "des_t_f",
    "des_c2",
    # C3 measures
    "malicious_selected",
    "paper_c3",
    "des_c3",
    # Reward
    "throughput_paper_nominal",
    "reward_paper_reference",
    "reward_des_liu_objective",
    # Run health
    "run_status",
    "failure_reason",
    "wall_clock_seconds",
)


def build_snapshot(faulty_node_ids: tuple[int, ...] = ()) -> LiuSingleActionSnapshot:
    return LiuSingleActionSnapshot(
        node_count=N,
        transaction_size_bytes=float(CHI_BYTES),
        stakes_tokens=STAKES,
        capabilities_ghz=CAPABILITIES,
        positions_km=POSITIONS,
        link_rows_mbps=LINK_ROWS,
        faulty_node_ids=faulty_node_ids,
        epoch0_validator_ids=tuple(range(K)),  # bootstrap with first K nodes
    )


def build_geo_model() -> ContinuousSpatialIntensityModel:
    return ContinuousSpatialIntensityModel(
        planar_gradient_intensity(K, LAMBDA_SLOPE), K,
        lambda_form=f"planar_gradient s={LAMBDA_SLOPE}",
    )


def build_params() -> LiuReferenceParameters:
    return LiuReferenceParameters(
        signature_verification_cycles_alpha=2_000_000.0,
        mac_operation_cycles_beta=1_000_000.0,
        network_timeout_s=100.0,
        finality_multiplier_omega=6.0,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=CHI_BYTES,
        recovery_delay_s=ZYZZYVA_RECOVERY_DELAY_S,
    )


def enumerate_validator_sets() -> list[tuple[int, ...]]:
    """All C(30,2) = 435 complementary validator sets: exclude each pair (i,j), i<j."""
    all_nodes = set(range(N))
    sets = []
    for i, j in itertools.combinations(range(N), 2):
        excluded = {i, j}
        validators = tuple(sorted(all_nodes - excluded))
        sets.append(validators)
    assert len(sets) == 435, f"expected 435, got {len(sets)}"
    return sets


def _run_one(
    action_id: int,
    experiment_type: str,
    protocol: LiuConsensusProtocol,
    validator_ids: tuple[int, ...],
    excluded: str,
    faulty_ids: tuple[int, ...],
) -> dict:
    row = {name: "" for name in CSV_COLUMNS}
    row.update(
        action_id=action_id,
        experiment_type=experiment_type,
        protocol=protocol.value,
        excluded_nodes=excluded,
        validator_ids=",".join(str(v) for v in validator_ids),
        faulty_node_ids=",".join(str(f) for f in faulty_ids),
        block_size_mb=FIXED_SB,
        block_interval_s=FIXED_TI,
    )
    started = time.time()
    malicious_selected = sum(1 for f in faulty_ids if f in set(validator_ids))
    try:
        action = LiuAction(N, K, validator_ids, protocol, FIXED_SB, FIXED_TI)
        snapshot = build_snapshot(faulty_ids)
        geo = build_geo_model()
        params = build_params()

        result = run_single_action(
            snapshot,
            action,
            geo,
            reference_params=params,
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            finality_multiplier_omega=6.0,
            signature_cycles_alpha=2_000_000.0,
            mac_cycles_beta=1_000_000.0,
            offered_workload_tx=OFFERED_WORKLOAD_TX,
            workload_tx_size_mb=TX_SIZE_MB,
            max_events=2_000_000,
        )
        p = paper_reference_for(snapshot, action, geo, params)
        d = result.des_observed

        row.update(
            stake_gini_paper=p.stake_gini,
            geo_gini=p.geographic_gini,
            paper_c1=p.c1_decentralization_passed,
            des_c1=d.c1_decentralization_passed,
            paper_t_c=p.consensus_latency_s,
            paper_t_f=p.finality_latency_s,
            paper_c2=p.c2_finality_passed,
            des_t_c=d.t_c_des_s,
            des_t_f=d.t_f_des_s,
            des_c2=d.c2_finality_passed,
            malicious_selected=malicious_selected,
            paper_c3=p.c3_security_passed,
            des_c3=d.c3_security_passed,
            throughput_paper_nominal=p.throughput_paper_nominal,
            reward_paper_reference=p.reward_paper_reference,
            reward_des_liu_objective=d.reward_des_liu_objective,
            run_status="SUCCESS",
        )
    except ValueError as err:
        msg = str(err)
        # LIU_QUORUM raises ValueError when f>0 (F^2=0 means zero fault tolerance).
        # This is semantically correct: C3 fails, reward=0. Record as SUCCESS with C3=False.
        if "F^2=0" in msg or "infeasible" in msg.lower():
            from PaperReference.ReferenceEvaluation import evaluate_reference
            from Liu.LinkState import LinkStateMatrix
            links = LinkStateMatrix.from_rows(LINK_ROWS)
            geo = build_geo_model()
            params = build_params()
            mask = tuple(1 if i in set(validator_ids) else 0 for i in range(N))
            try:
                # evaluate_reference returns LiuReferenceEvaluation with .reward and .throughput_tps
                ref = evaluate_reference(
                    node_stakes=STAKES,
                    node_capabilities_ghz=CAPABILITIES,
                    link_rates=links,
                    geographic_gini=geo.geographic_gini(),
                    producer_mask=mask,
                    protocol=protocol,
                    block_size_mb=FIXED_SB,
                    block_interval_s=FIXED_TI,
                    malicious_count=malicious_selected,
                    params=params,
                )
                # ref.c3_security_passed=False (f=1 > F^2=0); ref.reward=0 (infeasible)
                row.update(
                    stake_gini_paper=ref.stake_gini,
                    geo_gini=ref.geographic_gini,
                    paper_c1=ref.c1_decentralization_passed,
                    des_c1=ref.c1_decentralization_passed,
                    paper_t_c=ref.consensus_latency_s,
                    paper_t_f=ref.finality_latency_s,
                    paper_c2=ref.c2_finality_passed,
                    des_t_c="",
                    des_t_f="",
                    des_c2="",
                    malicious_selected=malicious_selected,
                    paper_c3=ref.c3_security_passed,
                    des_c3=False,
                    throughput_paper_nominal=ref.throughput_tps,
                    reward_paper_reference=ref.reward,
                    reward_des_liu_objective=0.0,
                    run_status="SUCCESS",
                    failure_reason=f"LIU_QUORUM_C3_INFEASIBLE(expected): {msg}",
                )
            except Exception as inner:
                row["run_status"] = "ERROR"
                row["failure_reason"] = f"{type(inner).__name__}: {inner}"
        else:
            row["run_status"] = "ERROR"
            row["failure_reason"] = f"ValueError: {msg}"
            traceback.print_exc()
    except Exception as err:
        row["run_status"] = "ERROR"
        row["failure_reason"] = f"{type(err).__name__}: {err}"
        traceback.print_exc()
    finally:
        row["wall_clock_seconds"] = time.time() - started
    return row


def _load_prior(csv_path: Path) -> dict[int, dict]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return {int(r["action_id"]): r for r in csv.DictReader(fh)}


def _write_all(csv_path: Path, results: dict[int, dict]) -> None:
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for aid in sorted(results):
            writer.writerow({col: results[aid].get(col, "") for col in CSV_COLUMNS})
    last_err = None
    for attempt in range(20):
        try:
            tmp.replace(csv_path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"could not replace {csv_path}: {last_err}")


def main() -> None:
    geo = build_geo_model()
    geo_gini = geo.geographic_gini()
    print(f"N={N}, K={K}, C(N,2)=435 validator sets")
    print(f"Fixed: S_B={FIXED_SB} MB, T_I={FIXED_TI} s")
    print(f"Geographic Gini (shared for all runs): {geo_gini:.4f} (threshold 0.3: {'PASS' if geo_gini <= 0.3 else 'FAIL'})")
    print()

    validator_sets = enumerate_validator_sets()
    protocols = list(LiuConsensusProtocol)

    # Build action list: 435 validator sets × 3 protocols (f=0 main runs)
    actions: list[dict] = []
    for vs_idx, vs in enumerate(validator_sets):
        # excluded pair
        all_nodes = set(range(N))
        excluded_nodes = tuple(sorted(all_nodes - set(vs)))
        exc_str = ",".join(str(x) for x in excluded_nodes)
        for proto in protocols:
            actions.append({
                "type": "main",
                "protocol": proto,
                "validator_ids": vs,
                "excluded": exc_str,
                "faulty_ids": (),
            })

    # Security tests: malicious node_id=0 (low stake)
    # One validator set that includes node 0, one that excludes it
    # For each of 3 protocols: 6 runs
    vs_including_0 = tuple(sorted(set(range(N)) - {N-1, N-2}))  # exclude last 2 → includes node 0
    vs_excluding_0 = tuple(sorted(set(range(N)) - {0, 1}))      # exclude 0 and 1 → excludes node 0
    malicious_node = 0
    for proto in protocols:
        actions.append({
            "type": "security_included",
            "protocol": proto,
            "validator_ids": vs_including_0,
            "excluded": f"{N-2},{N-1}",
            "faulty_ids": (malicious_node,),
        })
        actions.append({
            "type": "security_excluded",
            "protocol": proto,
            "validator_ids": vs_excluding_0,
            "excluded": f"0,1",
            "faulty_ids": (malicious_node,),
        })

    print(f"Total actions: {len(actions)} (1305 main + 6 security)")
    prior = _load_prior(CSV_PATH)
    success_ids = {aid for aid, row in prior.items() if row.get("run_status") == "SUCCESS"}
    print(f"Prior: {len(prior)} rows ({len(success_ids)} SUCCESS, resuming...)")

    results = dict(prior)
    started_global = time.time()
    attempted = 0
    for action_id, spec in enumerate(actions):
        if action_id in success_ids:
            continue
        row = _run_one(
            action_id,
            spec["type"],
            spec["protocol"],
            spec["validator_ids"],
            spec["excluded"],
            spec["faulty_ids"],
        )
        results[action_id] = row
        attempted += 1
        _write_all(CSV_PATH, results)
        elapsed = time.time() - started_global
        rate = attempted / elapsed if elapsed > 0 else 0
        remaining = len(actions) - (action_id + 1) - sum(
            1 for j in range(action_id + 1, len(actions)) if j in success_ids
        )
        eta_min = (remaining / rate / 60.0) if rate > 0 else float("inf")
        print(
            f"[{action_id + 1:04d}/{len(actions)}] {spec['type']:20s} "
            f"{spec['protocol'].value:11s}  {row.get('run_status','?'):8s}  "
            f"({row.get('wall_clock_seconds', 0):.2f}s)  ETA {eta_min:.1f}min",
            flush=True,
        )

    _write_all(CSV_PATH, results)
    total = len(results)
    success = sum(1 for r in results.values() if r.get("run_status") == "SUCCESS")
    print(f"\nDone: {success}/{total} SUCCESS")
    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()
