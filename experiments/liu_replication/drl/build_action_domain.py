"""Build the REDUCED_COMPLETE_LIU_ACTION_DOMAIN for N=30, K=28.

Classification: REDUCED_COMPLETE_LIU_ACTION_DOMAIN / OUR_RECONSTRUCTION_FOR_TRACTABLE_VALIDATION
This is NOT the original full Liu action cardinality (N=100/K=21 with continuous validator sampling).

Process:
1. Read the 2400-action controlled sweep.
2. For each protocol, filter feasible rows (run_status=SUCCESS, C1=C2=C3=True).
3. Sort by (reward DESC, des_t_f ASC, block_size_mb ASC, block_interval_s ASC).
4. Retain the top 3 DISTINCT (S_B, T_I) pairs per protocol.
5. Generate all C(30,28)=435 validator sets as sorted tuples.
6. Cross all 435 validator sets with all 9 (protocol, S_B, T_I) configs.
7. Persist reduced_action_domain.csv and reduced_action_domain_metadata.json.
"""
from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))

SWEEP_CSV = ROOT / "experiments/liu_replication/controlled_sweep/controlled_full_results.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_CSV = OUT_DIR / "reduced_action_domain.csv"
OUT_META = OUT_DIR / "reduced_action_domain_metadata.json"

N = 30
K = 28
PROTOCOLS = ["PBFT", "ZYZZYVA", "LIU_QUORUM"]
TOP_K_CONFIGS = 3


def _select_top_configs(sweep_path: Path) -> dict[str, list[tuple[float, float, float, float, str]]]:
    """For each protocol, return top-3 distinct (S_B, T_I) from feasible sweep rows.

    Returns {protocol: [(S_B, T_I, reward, des_t_f, source_action_id), ...]}
    """
    rows: dict[str, list] = {p: [] for p in PROTOCOLS}
    with open(sweep_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["run_status"] == "SUCCESS"
                and row["des_c1"] == "True"
                and row["des_c2"] == "True"
                and row["des_c3"] == "True"
            ):
                p = row["protocol"]
                if p in rows:
                    rows[p].append((
                        float(row["block_size_mb"]),
                        float(row["block_interval_s"]),
                        float(row["reward_des_liu_objective"]),
                        float(row["des_t_f"]),
                        row["action_id"],
                    ))

    selected: dict[str, list] = {}
    for p in PROTOCOLS:
        # Sort: reward DESC, des_t_f ASC, block_size_mb ASC, block_interval_s ASC
        prows = sorted(rows[p], key=lambda x: (-x[2], x[3], x[0], x[1]))
        seen: list[tuple[float, float, float, float, str]] = []
        for sb, ti, r, tf, aid in prows:
            if (sb, ti) not in [(s[0], s[1]) for s in seen]:
                seen.append((sb, ti, r, tf, aid))
                if len(seen) == TOP_K_CONFIGS:
                    break
        selected[p] = seen
    return selected


def _all_validator_sets(n: int, k: int) -> list[tuple[int, ...]]:
    """All C(n,k) k-subsets of range(n), each sorted ascending."""
    return [combo for combo in combinations(range(n), k)]


def build_domain() -> dict:
    """Build the complete reduced action domain and persist artifacts."""
    configs = _select_top_configs(SWEEP_CSV)

    # Flat list of (protocol, S_B, T_I) configurations
    protocol_configs: list[tuple[str, float, float, float, float, str]] = []
    for p in PROTOCOLS:
        for sb, ti, reward, tf, aid in configs[p]:
            protocol_configs.append((p, sb, ti, reward, tf, aid))

    validator_sets = _all_validator_sets(N, K)
    assert len(validator_sets) == 435, f"Expected 435 validator sets, got {len(validator_sets)}"

    actions = []
    action_id = 0
    for vset in validator_sets:
        mask = [1 if i in set(vset) else 0 for i in range(N)]
        for p, sb, ti, reward, tf, source_aid in protocol_configs:
            actions.append({
                "action_id": action_id,
                "protocol": p,
                "block_size_mb": sb,
                "block_interval_s": ti,
                "validator_ids": list(vset),
                "validator_mask": mask,
                "source_sweep_action_id": source_aid,
                "source_sweep_reward": reward,
                "source_sweep_des_t_f": tf,
            })
            action_id += 1

    # Assert uniqueness: every (vset, protocol, S_B, T_I) tuple is unique
    keys = {(a["protocol"], a["block_size_mb"], a["block_interval_s"], tuple(a["validator_ids"])) for a in actions}
    assert len(keys) == len(actions), "Action domain contains duplicates!"

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["action_id", "protocol", "block_size_mb", "block_interval_s",
                      "validator_ids", "validator_mask", "source_sweep_action_id",
                      "source_sweep_reward", "source_sweep_des_t_f"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in actions:
            writer.writerow({**a,
                             "validator_ids": json.dumps(a["validator_ids"]),
                             "validator_mask": json.dumps(a["validator_mask"])})

    # Write metadata JSON
    metadata = {
        "schema_version": "reduced_complete_liu_action_domain_v1",
        "classification": "REDUCED_COMPLETE_LIU_ACTION_DOMAIN / OUR_RECONSTRUCTION_FOR_TRACTABLE_VALIDATION",
        "note": "NOT the original full Liu action cardinality. Uses N=30/K=28 vs Liu N=100/K=21.",
        "n": N,
        "k": K,
        "c_n_k": len(validator_sets),
        "top_k_configs_per_protocol": TOP_K_CONFIGS,
        "total_actions": len(actions),
        "protocol_configs": {
            p: [{"block_size_mb": sb, "block_interval_s": ti, "source_reward": r,
                 "source_des_t_f": tf, "source_action_id": aid}
                for sb, ti, r, tf, aid in configs[p]]
            for p in PROTOCOLS
        },
        "field_descriptions": {
            "action_id": "Unique integer ID for this action (0-indexed, row-major: validator_sets outer, configs inner)",
            "protocol": "Consensus protocol: PBFT / ZYZZYVA / LIU_QUORUM",
            "block_size_mb": "Block size S_B in MB (from paper grid: 0.2-step multiples)",
            "block_interval_s": "Block interval T_I in seconds (from paper grid: 0.5-step multiples)",
            "validator_ids": "JSON list of K validator node IDs (sorted ascending)",
            "validator_mask": "JSON list of N binary values: 1 if node i in validator set",
            "source_sweep_action_id": "action_id from controlled_full_results.csv that generated this (S_B, T_I)",
            "source_sweep_reward": "reward_des_liu_objective of source row",
            "source_sweep_des_t_f": "des_t_f of source row",
        },
    }
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return metadata


if __name__ == "__main__":
    meta = build_domain()
    print(f"Total actions: {meta['total_actions']}")
    print(f"Validator sets: {meta['c_n_k']}")
    for p, cfgs in meta["protocol_configs"].items():
        print(f"  {p}: {[(c['block_size_mb'], c['block_interval_s'], round(c['source_reward'],1)) for c in cfgs]}")
    print(f"Artifacts written to: {OUT_DIR}")
    print("Gate: REDUCED_COMPLETE_ACTION_DOMAIN_VERIFIED — pending micro DRL gate")
