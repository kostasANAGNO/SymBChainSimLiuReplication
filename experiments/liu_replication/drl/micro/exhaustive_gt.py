"""Exhaustive DES ground truth for the micro environment.

Runs all 180 actions through the real DES for 3 fixed network states:
  GOOD_NETWORK (100 Mbps), MEDIUM_NETWORK (55 Mbps), POOR_NETWORK (10 Mbps).

Outputs:
  exhaustive_ground_truth.csv      — one row per (state, action) pair
  exhaustive_ground_truth_summary.json — best action per state, feasibility stats

Asserts that the three states do NOT all have exactly the same optimal action set.

Label: MICRO_VALIDATION_FIXTURE
This is a correctness test for the DRL algorithm, NOT a Liu scientific result.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
_DYN = ROOT / "experiments" / "liu_replication" / "dynamic_epoch"
sys.path.insert(0, str(_DYN))
sys.path.insert(0, str(HERE.parent))  # for config.py etc in drl root

from micro_env import (
    all_micro_actions, make_micro_env, state_dict_from_links,
    NETWORK_STATES, N_MICRO, K_MICRO, CHI_BYTES,
    MICRO_BLOCK_SIZES, MICRO_BLOCK_INTERVALS,
)
from liu_dynamic_env import DESFailureReason

OUT_CSV = HERE / "exhaustive_ground_truth.csv"
OUT_JSON = HERE / "exhaustive_ground_truth_summary.json"


def run_exhaustive() -> dict:
    actions = all_micro_actions()
    print(f"Actions: {len(actions)} (expected 180)")
    assert len(actions) == 180

    all_rows = []
    results_by_state: dict[str, list] = {}

    for state_name, link_rows in NETWORK_STATES.items():
        print(f"\n=== {state_name} ===")
        env = make_micro_env(link_rows, seed=42)
        env.reset(seed=42)
        rows = []

        for a in actions:
            action_obj = a["_liu_action"]
            t0 = time.perf_counter()
            r = env.step(action_obj)
            # Reset FSMC to the fixed state after each step
            env.reset(seed=42)
            wall = time.perf_counter() - t0

            row = {
                "state_name": state_name,
                "action_id": a["action_id"],
                "protocol": a["protocol"],
                "block_size_mb": a["block_size_mb"],
                "block_interval_s": a["block_interval_s"],
                "validator_ids": json.dumps(a["validator_ids"]),
                "reward": r.reward,
                "des_c1": r.des_c1,
                "des_c2": r.des_c2,
                "des_c3": r.des_c3,
                "des_t_c_s": r.des_t_c_s if not r.stalled else "inf",
                "failure_reason": r.failure_reason.value,
                "wall_s": round(wall, 4),
            }
            rows.append(row)
            all_rows.append(row)
            if a["action_id"] % 30 == 0:
                print(f"  action {a['action_id']:3d}: R={r.reward:.1f} C1={r.des_c1} C2={r.des_c2} C3={r.des_c3} [{r.failure_reason.value}]")

        results_by_state[state_name] = rows

    # Write CSV
    fieldnames = ["state_name","action_id","protocol","block_size_mb","block_interval_s",
                  "validator_ids","reward","des_c1","des_c2","des_c3","des_t_c_s","failure_reason","wall_s"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Build summary
    summary = {"states": {}}
    all_optima: list[set] = []
    for state_name, rows in results_by_state.items():
        feasible = [r for r in rows if r["reward"] > 0]
        max_reward = max((r["reward"] for r in rows), default=0.0)
        optimal_ids = {r["action_id"] for r in rows if r["reward"] == max_reward and max_reward > 0}
        all_optima.append(optimal_ids)
        summary["states"][state_name] = {
            "feasible_count": len(feasible),
            "infeasible_count": len(rows) - len(feasible),
            "max_reward": max_reward,
            "optimal_action_ids": sorted(optimal_ids),
            "optimal_actions": [
                {"action_id": r["action_id"], "protocol": r["protocol"],
                 "block_size_mb": r["block_size_mb"], "block_interval_s": r["block_interval_s"],
                 "validator_ids": r["validator_ids"]}
                for r in rows if r["action_id"] in optimal_ids
            ],
        }
        print(f"\n{state_name}: {len(feasible)}/180 feasible, max_reward={max_reward:.1f}, "
              f"optimal_ids={sorted(optimal_ids)[:5]}{'...' if len(optimal_ids)>5 else ''}")

    # Assert that the three states do NOT all have identical optimal action sets
    same_optima = all_optima[0] == all_optima[1] == all_optima[2]
    if same_optima:
        raise RuntimeError(
            "MICRO_VALIDATION_FIXTURE FAILED: all 3 network states have identical optimal action sets.\n"
            "Adjust network states or S_B/T_I grid so that different R produces different optima.\n"
            "Do NOT change Liu reward semantics."
        )
    summary["distinct_optima_across_states"] = True
    summary["fixture_label"] = "MICRO_VALIDATION_FIXTURE / OUR_RECONSTRUCTION"
    summary["n"] = N_MICRO
    summary["k"] = K_MICRO
    summary["chi_bytes"] = CHI_BYTES
    summary["block_sizes_mb"] = list(MICRO_BLOCK_SIZES)
    summary["block_intervals_s"] = list(MICRO_BLOCK_INTERVALS)
    summary["total_actions"] = 180
    summary["note"] = "Algorithmic test only. NOT a Liu scientific result."

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"\nDistinct optima across states: {not same_optima}")
    print(f"Written: {OUT_CSV}")
    print(f"Written: {OUT_JSON}")
    return summary


if __name__ == "__main__":
    summary = run_exhaustive()
    print("\n=== GROUND TRUTH COMPLETE ===")
    for state_name, info in summary["states"].items():
        print(f"  {state_name}: feasible={info['feasible_count']}/180  max_reward={info['max_reward']:.1f}")
