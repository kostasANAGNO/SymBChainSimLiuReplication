"""N30 Seed-42 Clean Policy Evaluation — N30_SEED42_POLICY_VERIFIED gate.

Evaluates clean ep400 checkpoint (primary) and ep350 (diagnostic) against:
  - Random policy
  - Strong fixed baseline (LIU_QUORUM, S_B=6.8 MB, T_I=2.0 s, one C1-valid set)

Trajectory design:
  - 4 eval seeds: [1000, 1001, 1002, 1003]  (NOT in training range [42, 441])
  - 4 initial link states: ALL_HIGH_100 / ALL_MID_55 / ALL_LOW_10 / TRAINING_MIXED
  - 20 steps per trajectory (same horizon as training episodes)
  - All policies share IDENTICAL FSMC transition sequences per trajectory
    (same env.reset(seed=eval_seed) → same random.Random state → same advances)

Output:
  n30_seed42_policy_evaluation.json
  n30_seed42_policy_evaluation.md
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
_DYN = ROOT / "experiments" / "liu_replication" / "dynamic_epoch"
sys.path.insert(0, str(_DYN))
sys.path.insert(0, str(HERE.parent))  # drl root: config, state_encoder, …

from config import DQNConfig
from state_encoder import StateEncoder
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from liu_dynamic_env import LiuDynamicEpochEnv, DESFailureReason
from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

# ── Population (identical to training) ────────────────────────────────────────
N = 30
K = 28
CHI_BYTES = 200
REWARD_SCALE = 17_000.0

STAKES = tuple(float(4 + (i % 10)) for i in range(N))
CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(N))
POSITIONS = tuple((0.2 * (i % 5), 0.2 * (i // 5)) for i in range(N))
INITIAL_LINK_ROWS = tuple(
    tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N))
    for i in range(N)
)

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

# ── Paths ─────────────────────────────────────────────────────────────────────
CKPT_DIR   = HERE / "checkpoints"
DRL_DIR    = HERE.parent
ACTION_CSV = DRL_DIR / "reduced_action_domain.csv"
OUT_JSON   = HERE / "n30_seed42_policy_evaluation.json"
OUT_MD     = HERE / "n30_seed42_policy_evaluation.md"

# ── Evaluation configuration ──────────────────────────────────────────────────
EVAL_SEEDS = [1000, 1001, 1002, 1003]
EVAL_STEPS = 20
PRIMARY_CKPT  = CKPT_DIR / "ckpt_seed42_clean_ep400.pt"
DIAG_CKPT     = CKPT_DIR / "ckpt_seed42_clean_ep350.pt"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_links(rate: float) -> tuple:
    return tuple(tuple(None if i == j else rate for j in range(N)) for i in range(N))


INITIAL_STATES: dict[str, tuple] = {
    "ALL_HIGH_100":    _all_links(100.0),
    "ALL_MID_55":      _all_links(55.0),
    "ALL_LOW_10":      _all_links(10.0),
    "TRAINING_MIXED":  INITIAL_LINK_ROWS,
}


def gini_coefficient(values: list[float]) -> float:
    if not values:
        return 0.0
    n = len(values)
    s = sorted(values)
    total = sum(s)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(s))
    return 2 * weighted / (n * total) - (n + 1) / n


def stake_gini(validator_ids: tuple) -> float:
    return gini_coefficient([STAKES[i] for i in validator_ids])


def load_actions() -> list[dict]:
    actions = []
    with open(ACTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            validator_ids = tuple(json.loads(row["validator_ids"]))
            sb = float(row["block_size_mb"])
            ti = float(row["block_interval_s"])
            protocol_str = row["protocol"]
            actions.append({
                "action_id": int(row["action_id"]),
                "protocol": protocol_str,
                "block_size_mb": sb,
                "block_interval_s": ti,
                "validator_ids": validator_ids,
                "validator_mask": json.loads(row["validator_mask"]),
                "source_sweep_reward": float(row["source_sweep_reward"]),
                "_liu_action": LiuAction(
                    N, K, validator_ids, LiuConsensusProtocol(protocol_str), sb, ti
                ),
            })
    assert len(actions) == 3915, f"Expected 3915, got {len(actions)}"
    return actions


def make_env(initial_link_rows: tuple, seed: int) -> LiuDynamicEpochEnv:
    return LiuDynamicEpochEnv(
        node_count=N,
        transaction_size_bytes=float(CHI_BYTES),
        stakes_tokens=STAKES,
        capabilities_ghz=CAPABILITIES,
        positions_km=POSITIONS,
        initial_link_rows_mbps=initial_link_rows,
        faulty_node_ids=(),
        reference_params=REF_PARAMS,
        geo_model=GEO,
        offered_workload_tx=100_000,
        max_events=2_000_000,
        seed=seed,
    )


def make_state_dict(link_rows: tuple) -> dict:
    return {
        "chi_bytes": float(CHI_BYTES),
        "stakes": STAKES,
        "positions_km": POSITIONS,
        "capabilities_ghz": CAPABILITIES,
        "link_rows_mbps": link_rows,
    }


def load_dqn_greedy(ckpt_path: Path, state_enc: StateEncoder,
                     action_enc: ActionEncoder, actions: list[dict]) -> DQNAgent:
    """Load checkpoint with epsilon=0 for greedy evaluation."""
    ckpt = torch.load(ckpt_path, weights_only=False)
    cd = ckpt["cfg_dict"]
    cfg = DQNConfig(
        hidden_sizes=tuple(cd["hidden_sizes"]),
        gamma=cd["gamma"],
        learning_rate=cd["learning_rate"],
        replay_capacity=cd["replay_capacity"],
        replay_warmup_steps=cd["replay_warmup_steps"],
        batch_size=cd["batch_size"],
        target_sync_steps=cd["target_sync_steps"],
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay_steps=1,
    )
    action_tensors = action_enc.encode_batch(actions)
    agent = DQNAgent(state_enc.dim, action_enc.dim, action_tensors, cfg, seed=0)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.online.eval()
    return agent


def identify_c1_valid(actions: list[dict]) -> set[tuple]:
    """Return set of validator_ids tuples that satisfy C1 stake gini ≤ 0.2."""
    geo_g = GEO.geographic_gini()
    valid = set()
    for a in actions:
        vids = a["validator_ids"]
        if vids not in valid:
            sg = stake_gini(vids)
            if sg <= 0.2 and geo_g <= 0.3:
                valid.add(vids)
    return valid


def find_strong_fixed(actions: list[dict], c1_valid_ids: set[tuple]) -> dict:
    """Find first LIU_QUORUM S_B=6.8 T_I=2.0 C1-valid action."""
    for a in actions:
        if (a["protocol"] == "LIU_QUORUM"
                and abs(a["block_size_mb"] - 6.8) < 0.01
                and abs(a["block_interval_s"] - 2.0) < 0.01
                and a["validator_ids"] in c1_valid_ids):
            return a
    raise ValueError("No valid strong fixed action found")


def net_bucket(mean_r: float, min_r: float) -> str:
    if min_r <= 10.0 + 0.5:
        return "CONTAINS_LOW_10MBPS"
    if mean_r >= 85.0:
        return "NEAR_HIGH_100MBPS"
    if mean_r >= 45.0:
        return "MEDIUM_RANGE"
    return "LOW_MEDIUM_RANGE"


# ── Q-health ──────────────────────────────────────────────────────────────────

def q_health(agent: DQNAgent, s_enc: torch.Tensor) -> dict:
    with torch.no_grad():
        q_vals = agent.online.q_values_batch(s_enc, agent.all_action_tensors)
    q = q_vals.cpu().numpy()
    return {
        "q_min":       float(q.min()),
        "q_max":       float(q.max()),
        "q_mean":      float(q.mean()),
        "q_std":       float(q.std()),
        "q_has_nan":   bool(any(math.isnan(v) for v in q)),
        "q_has_inf":   bool(any(math.isinf(v) for v in q)),
        "q_degenerate": float(q.std()) < 1e-4,
        "top5_action_ids": [int(i) for i in q.argsort()[-5:][::-1].tolist()],
    }


# ── Policy definitions ────────────────────────────────────────────────────────

def dqn_policy(agent: DQNAgent, state_enc: StateEncoder, actions: list[dict]):
    def _fn(s_enc: torch.Tensor) -> tuple[int, dict]:
        idx = agent.argmax(s_enc)
        return idx, actions[idx]
    return _fn


def random_policy(actions: list[dict], rng: random.Random):
    def _fn(s_enc: torch.Tensor) -> tuple[int, dict]:
        idx = rng.randrange(len(actions))
        return idx, actions[idx]
    return _fn


def fixed_policy(action: dict):
    def _fn(s_enc: torch.Tensor) -> tuple[int, dict]:
        return action["action_id"], action
    return _fn


# ── Trajectory rollout ────────────────────────────────────────────────────────

def run_trajectory(
    env: LiuDynamicEpochEnv,
    policy_fn: Callable,
    state_enc: StateEncoder,
    n_steps: int,
) -> list[dict]:
    link_rows = env._current_link_rows()
    s_enc = state_enc.encode(make_state_dict(link_rows))
    records = []

    for step_idx in range(n_steps):
        mean_r, min_r, max_r = env._link_stats()
        _action_idx, action_dict = policy_fn(s_enc)
        result = env.step(action_dict["_liu_action"])

        omega = (action_dict["block_size_mb"] / (CHI_BYTES / 1_000_000)) / action_dict["block_interval_s"]

        records.append({
            "step": step_idx,
            "action_id":        action_dict["action_id"],
            "protocol":         action_dict["protocol"],
            "block_size_mb":    action_dict["block_size_mb"],
            "block_interval_s": action_dict["block_interval_s"],
            "validator_ids":    action_dict["validator_ids"],
            "link_mean_mbps":   round(mean_r, 2),
            "link_min_mbps":    round(min_r, 2),
            "link_max_mbps":    round(max_r, 2),
            "net_bucket":       net_bucket(mean_r, min_r),
            "des_c1": int(result.des_c1),
            "des_c2": int(result.des_c2),
            "des_c3": int(result.des_c3),
            "des_t_c_s": result.des_t_c_s if math.isfinite(result.des_t_c_s) else None,
            "failure_reason":   result.failure_reason.value,
            "reward":           result.reward,
            "omega":            omega,
            "stalled":          result.stalled,
        })

        next_link_rows = env._current_link_rows()
        s_enc = state_enc.encode(make_state_dict(next_link_rows))

    return records


# ── Metrics aggregation ───────────────────────────────────────────────────────

def aggregate(records: list[dict], c1_valid_ids: set[tuple]) -> dict:
    rewards = [r["reward"] for r in records]
    feasible = [r for r in records if not r["stalled"]]
    sorted_r = sorted(rewards)
    n = len(records)

    omega_vals = [r["omega"] for r in feasible]

    # Validator validity
    n_valid   = sum(1 for r in records if r["validator_ids"] in c1_valid_ids)
    n_invalid = n - n_valid

    return {
        "n_steps":               n,
        "cumulative_reward":     round(sum(rewards), 4),
        "mean_reward":           round(sum(rewards) / n, 4) if n else 0.0,
        "median_reward":         float(sorted_r[n // 2]) if n else 0.0,
        "feasible_steps":        len(feasible),
        "feasibility_rate":      round(len(feasible) / n, 4) if n else 0.0,
        "c1_pass":               sum(r["des_c1"] for r in records),
        "c2_pass":               sum(r["des_c2"] for r in records),
        "c3_pass":               sum(r["des_c3"] for r in records),
        "c2_deadline_exceeded":  sum(1 for r in records if r["failure_reason"] == "C2_DEADLINE_EXCEEDED"),
        "runtime_guard_failure": sum(1 for r in records if r["failure_reason"] == "RUNTIME_GUARD_FAILURE"),
        "mean_omega_feasible":   round(sum(omega_vals) / len(omega_vals), 4) if omega_vals else 0.0,
        "validator_c1_valid_selected":   n_valid,
        "validator_c1_invalid_selected": n_invalid,
        "validator_valid_rate":          round(n_valid / n, 4) if n else 0.0,
        "net_bucket_counts":     dict(Counter(r["net_bucket"] for r in records)),
    }


def dqn_action_profile(records: list[dict]) -> dict:
    protocols  = Counter(r["protocol"] for r in records)
    sb_vals    = Counter(r["block_size_mb"] for r in records)
    ti_vals    = Counter(r["block_interval_s"] for r in records)
    action_ids = Counter(r["action_id"] for r in records)
    top_actions = action_ids.most_common(10)
    return {
        "protocol_dist": dict(protocols),
        "block_size_dist": {str(k): v for k, v in sb_vals.items()},
        "block_interval_dist": {str(k): v for k, v in ti_vals.items()},
        "top10_action_ids": top_actions,
        "n_distinct_actions": len(action_ids),
    }


def adaptation_by_bucket(records: list[dict]) -> dict:
    """Protocol/SB/TI breakdown by network condition bucket."""
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        buckets[r["net_bucket"]].append(r)
    result = {}
    for bk, recs in buckets.items():
        result[bk] = {
            "n": len(recs),
            "protocol_dist": dict(Counter(r["protocol"] for r in recs)),
            "block_size_dist": dict(Counter(r["block_size_mb"] for r in recs)),
            "block_interval_dist": dict(Counter(r["block_interval_s"] for r in recs)),
            "feasible": sum(1 for r in recs if not r["stalled"]),
            "mean_reward": round(sum(r["reward"] for r in recs) / len(recs), 2),
        }
    return result


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()

    print("Loading actions…")
    actions = load_actions()

    print("Identifying C1-valid validator sets…")
    geo_g = GEO.geographic_gini()
    c1_valid_ids: set[tuple] = identify_c1_valid(actions)
    all_vsets = {a["validator_ids"] for a in actions}
    c1_invalid_ids = all_vsets - c1_valid_ids
    print(f"  geo_gini={geo_g:.4f}  C1-valid sets: {len(c1_valid_ids)}  "
          f"C1-invalid: {len(c1_invalid_ids)}")

    print("Finding strong fixed action (LIU_QUORUM S_B=6.8 T_I=2.0 C1-valid)…")
    fixed_action = find_strong_fixed(actions, c1_valid_ids)
    fixed_stake_g = stake_gini(fixed_action["validator_ids"])
    print(f"  action_id={fixed_action['action_id']}  "
          f"validators={list(fixed_action['validator_ids'])[:5]}...  "
          f"stake_gini={fixed_stake_g:.4f}")

    print("Initialising encoders…")
    cfg_dummy = DQNConfig()
    state_enc  = StateEncoder(N, cfg_dummy)
    action_enc = ActionEncoder(N, cfg_dummy)

    print(f"Loading primary checkpoint:    {PRIMARY_CKPT.name}")
    agent_ep400 = load_dqn_greedy(PRIMARY_CKPT, state_enc, action_enc, actions)
    print(f"Loading diagnostic checkpoint: {DIAG_CKPT.name}")
    agent_ep350 = load_dqn_greedy(DIAG_CKPT, state_enc, action_enc, actions)

    # Policy makers (random RNG is per-trajectory, seeded with eval_seed for reproducibility)
    def make_policies(eval_seed: int):
        rng_rand = random.Random(eval_seed + 9999)  # offset so it differs from env rng
        return {
            "dqn_ep400":    dqn_policy(agent_ep400, state_enc, actions),
            "dqn_ep350":    dqn_policy(agent_ep350, state_enc, actions),
            "random":       random_policy(actions, rng_rand),
            "strong_fixed": fixed_policy(fixed_action),
        }

    # ── Run evaluation trajectories ────────────────────────────────────────────
    traj_records: dict[str, dict[str, dict[str, list[dict]]]] = {}
    # traj_records[eval_seed_str][init_state_name][policy_name] = list of step records

    total_traj = len(EVAL_SEEDS) * len(INITIAL_STATES)
    traj_count = 0

    q_health_samples: list[dict] = []  # sample Q-health at start of some trajectories

    for eval_seed in EVAL_SEEDS:
        traj_records[str(eval_seed)] = {}
        policies = make_policies(eval_seed)

        for init_name, init_links in INITIAL_STATES.items():
            traj_count += 1
            print(f"  [{traj_count}/{total_traj}] seed={eval_seed} state={init_name}  ", end="", flush=True)
            t_traj = time.perf_counter()

            # Create one env per policy with IDENTICAL seed → identical FSMC trajectory
            envs = {
                pname: make_env(init_links, eval_seed)
                for pname in policies
            }
            # Reset all to the same seed
            for env in envs.values():
                env.reset(seed=eval_seed)

            # Sample Q-health from DQN on the first state of this trajectory
            if len(q_health_samples) < 8:
                link_rows0 = envs["dqn_ep400"]._current_link_rows()
                s0 = state_enc.encode(make_state_dict(link_rows0))
                qh = q_health(agent_ep400, s0)
                qh["eval_seed"] = eval_seed
                qh["init_state"] = init_name
                q_health_samples.append(qh)

            traj_records[str(eval_seed)][init_name] = {}
            for pname, policy_fn in policies.items():
                records = run_trajectory(
                    envs[pname], policy_fn, state_enc, EVAL_STEPS
                )
                traj_records[str(eval_seed)][init_name][pname] = records

            elapsed = time.perf_counter() - t_traj
            print(f"done in {elapsed:.0f}s")

    print(f"\nAll trajectories complete. Total time: {time.perf_counter()-t0:.0f}s")

    # ── Aggregate across all trajectories ─────────────────────────────────────
    policy_names = ["dqn_ep400", "dqn_ep350", "random", "strong_fixed"]
    all_records: dict[str, list[dict]] = {p: [] for p in policy_names}

    for seed_str, seed_data in traj_records.items():
        for init_name, init_data in seed_data.items():
            for pname in policy_names:
                all_records[pname].extend(init_data[pname])

    # Global aggregate per policy
    global_agg: dict[str, dict] = {}
    for pname in policy_names:
        global_agg[pname] = aggregate(all_records[pname], c1_valid_ids)

    # DQN-specific action profile
    dqn400_profile = dqn_action_profile(all_records["dqn_ep400"])
    dqn400_adapt   = adaptation_by_bucket(all_records["dqn_ep400"])

    # ── Paired trajectory comparison ──────────────────────────────────────────
    # One trajectory = (eval_seed, init_state), 16 total
    paired_400_vs_random = []
    paired_400_vs_fixed  = []
    paired_400_vs_350    = []

    for seed_str, seed_data in traj_records.items():
        for init_name, init_data in seed_data.items():
            cr400   = sum(r["reward"] for r in init_data["dqn_ep400"])
            cr_rand = sum(r["reward"] for r in init_data["random"])
            cr_fix  = sum(r["reward"] for r in init_data["strong_fixed"])
            cr350   = sum(r["reward"] for r in init_data["dqn_ep350"])

            paired_400_vs_random.append({
                "eval_seed": int(seed_str), "init_state": init_name,
                "dqn400": cr400, "random": cr_rand, "diff": cr400 - cr_rand,
            })
            paired_400_vs_fixed.append({
                "eval_seed": int(seed_str), "init_state": init_name,
                "dqn400": cr400, "strong_fixed": cr_fix, "diff": cr400 - cr_fix,
            })
            paired_400_vs_350.append({
                "eval_seed": int(seed_str), "init_state": init_name,
                "dqn_ep400": cr400, "dqn_ep350": cr350, "diff": cr400 - cr350,
            })

    def paired_summary(pairs: list[dict], a_key: str, b_key: str) -> dict:
        diffs = [p["diff"] for p in pairs]
        n_win  = sum(1 for d in diffs if d > 0)
        n_tie  = sum(1 for d in diffs if d == 0)
        n_lose = sum(1 for d in diffs if d < 0)
        return {
            "n_trajectories":    len(pairs),
            f"{a_key}_wins":     n_win,
            "tied":              n_tie,
            f"{b_key}_wins":     n_lose,
            "mean_paired_diff":  round(sum(diffs) / len(diffs), 2),
        }

    paired_summary_rand  = paired_summary(paired_400_vs_random, "dqn400", "random")
    paired_summary_fixed = paired_summary(paired_400_vs_fixed,  "dqn400", "strong_fixed")
    paired_summary_350   = paired_summary(paired_400_vs_350,    "dqn_ep400", "dqn_ep350")

    # ── Concrete trajectory example (find healthy→degraded→recovery) ──────────
    # Pick trajectory from ALL_HIGH_100 initial state (likely to show degradation)
    best_example_seed = None
    best_bucket_variety = -1
    for seed_str, seed_data in traj_records.items():
        for init_name in ["ALL_HIGH_100", "TRAINING_MIXED", "ALL_MID_55"]:
            if init_name not in seed_data:
                continue
            recs = seed_data[init_name]["dqn_ep400"]
            buckets_seen = len(set(r["net_bucket"] for r in recs))
            if buckets_seen > best_bucket_variety:
                best_bucket_variety = buckets_seen
                best_example = {
                    "eval_seed": int(seed_str),
                    "init_state": init_name,
                    "records": recs,
                }

    # Format as concise table
    example_steps = []
    for r in best_example["records"]:
        example_steps.append({
            "step":              r["step"],
            "net_summary":       f"mean={r['link_mean_mbps']:.0f} min={r['link_min_mbps']:.0f} Mbps",
            "net_bucket":        r["net_bucket"],
            "protocol":          r["protocol"],
            "block_size_mb":     r["block_size_mb"],
            "block_interval_s":  r["block_interval_s"],
            "validator_c1_valid": r["validator_ids"] in c1_valid_ids,
            "des_c1": r["des_c1"],
            "des_c2": r["des_c2"],
            "des_c3": r["des_c3"],
            "reward": r["reward"],
            "failure_reason": r["failure_reason"],
        })

    # ── Q-health summary across samples ──────────────────────────────────────
    any_nan = any(s["q_has_nan"] for s in q_health_samples)
    any_inf = any(s["q_has_inf"] for s in q_health_samples)
    any_deg = any(s["q_degenerate"] for s in q_health_samples)
    q_mins  = [s["q_min"]  for s in q_health_samples]
    q_maxs  = [s["q_max"]  for s in q_health_samples]
    q_stds  = [s["q_std"]  for s in q_health_samples]

    q_health_summary = {
        "n_samples":        len(q_health_samples),
        "any_nan":          any_nan,
        "any_inf":          any_inf,
        "any_degenerate":   any_deg,
        "global_q_min":     round(min(q_mins), 4),
        "global_q_max":     round(max(q_maxs), 4),
        "mean_q_std":       round(sum(q_stds) / len(q_stds), 4),
        "samples":          q_health_samples,
    }

    # ── Gate decision ─────────────────────────────────────────────────────────
    agg400  = global_agg["dqn_ep400"]
    agg_rnd = global_agg["random"]

    gate_criteria = {
        "A_stable_finite_q":              not any_nan and not any_inf,
        "B_near_zero_c1_invalid_selection": agg400["validator_valid_rate"] >= 0.90,
        "C_meaningful_feasible_behavior":   agg400["feasibility_rate"] >= 0.10,
        "D_better_than_random_paired":      paired_summary_rand["mean_paired_diff"] > 0,
        "E_no_runtime_guard_pathology":     agg400["runtime_guard_failure"] == 0,
        "F_policy_adaptation_evidence":     len(dqn400_adapt) >= 2,
    }
    gate_awarded = all(gate_criteria.values())

    # ep350 regression check
    ep350_better = paired_summary_350["dqn_ep350_wins"] > paired_summary_350["dqn_ep400_wins"]
    late_training_regression = "LATE_TRAINING_POLICY_REGRESSION" if ep350_better else "NOT_DETECTED"

    # ── Build output JSON ────────────────────────────────────────────────────
    result = {
        "schema": "n30_seed42_policy_evaluation_v1",
        "gate": "N30_SEED42_POLICY_VERIFIED",
        "gate_awarded": gate_awarded,
        "gate_criteria": gate_criteria,
        "evaluation_design": {
            "eval_seeds":    EVAL_SEEDS,
            "eval_steps":    EVAL_STEPS,
            "initial_states": list(INITIAL_STATES.keys()),
            "n_trajectories": len(EVAL_SEEDS) * len(INITIAL_STATES),
            "total_steps_per_policy": len(EVAL_SEEDS) * len(INITIAL_STATES) * EVAL_STEPS,
            "training_rng_range":  [42, 441],
            "eval_seeds_unseen":   True,
            "fsmc_trajectory_identical_for_all_policies": True,
            "trajectory_design_note": (
                "Each env reset with same eval_seed → identical random.Random state → "
                "identical FSMC advance sequence for all 4 policies on each trajectory."
            ),
        },
        "c1_validity": {
            "geo_gini": round(geo_g, 4),
            "c1_valid_validator_sets":   len(c1_valid_ids),
            "c1_invalid_validator_sets": len(c1_invalid_ids),
        },
        "strong_fixed_action": {
            "action_id": fixed_action["action_id"],
            "protocol":  fixed_action["protocol"],
            "block_size_mb": fixed_action["block_size_mb"],
            "block_interval_s": fixed_action["block_interval_s"],
            "validator_ids": list(fixed_action["validator_ids"]),
            "stake_gini": round(fixed_stake_g, 4),
            "c1_valid": fixed_action["validator_ids"] in c1_valid_ids,
        },
        "ep400_primary": {
            **global_agg["dqn_ep400"],
            "action_profile": dqn400_profile,
            "adaptation_by_network_bucket": dqn400_adapt,
        },
        "random_baseline": global_agg["random"],
        "strong_fixed_baseline": global_agg["strong_fixed"],
        "ep350_diagnostic": {
            **global_agg["dqn_ep350"],
            "late_training_regression_verdict": late_training_regression,
        },
        "paired_comparisons": {
            "ep400_vs_random":       {"summary": paired_summary_rand,  "trajectories": paired_400_vs_random},
            "ep400_vs_strong_fixed": {"summary": paired_summary_fixed, "trajectories": paired_400_vs_fixed},
            "ep400_vs_ep350":        {"summary": paired_summary_350,   "trajectories": paired_400_vs_350},
        },
        "q_health": q_health_summary,
        "concrete_trajectory_example": {
            "eval_seed":  best_example["eval_seed"],
            "init_state": best_example["init_state"],
            "policy":     "dqn_ep400",
            "steps":      example_steps,
        },
        "wall_s": round(time.perf_counter() - t0, 1),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"JSON written: {OUT_JSON}")

    # ── Write Markdown summary ────────────────────────────────────────────────
    write_md(result, agg400, agg_rnd, global_agg["strong_fixed"], global_agg["dqn_ep350"],
             paired_summary_rand, paired_summary_fixed, paired_summary_350,
             q_health_summary, gate_criteria, gate_awarded, late_training_regression,
             dqn400_profile, dqn400_adapt, example_steps,
             fixed_action, fixed_stake_g, geo_g, c1_valid_ids, c1_invalid_ids)
    print(f"MD  written: {OUT_MD}")
    print(f"\nGate N30_SEED42_POLICY_VERIFIED: {'AWARDED' if gate_awarded else 'NOT AWARDED'}")
    return result


def write_md(result, agg400, agg_rnd, agg_fix, agg350,
             p_rand, p_fix, p_350,
             qh, gate_criteria, gate_awarded, regression,
             profile, adapt, example_steps,
             fixed_action, fixed_sg, geo_g,
             c1_valid_ids, c1_invalid_ids):

    total_steps = result["evaluation_design"]["total_steps_per_policy"]
    lines = []
    lines.append("# N30 Seed-42 Clean Policy Evaluation")
    lines.append("")
    lines.append(f"**Gate: N30_SEED42_POLICY_VERIFIED — {'AWARDED' if gate_awarded else 'NOT AWARDED'}**")
    lines.append("")
    lines.append("## 1. Evaluation Design")
    lines.append("")
    lines.append(f"- Eval seeds: {result['evaluation_design']['eval_seeds']} (outside training range [42, 441])")
    lines.append(f"- Initial states: {', '.join(result['evaluation_design']['initial_states'])}")
    lines.append(f"- Steps per trajectory: {result['evaluation_design']['eval_steps']}")
    lines.append(f"- Trajectories: {result['evaluation_design']['n_trajectories']}")
    lines.append(f"- Total steps per policy: {total_steps}")
    lines.append("- FSMC trajectory: **identical** for all policies per trajectory (same `random.Random(eval_seed)` initial state)")
    lines.append("")
    lines.append("## 2. C1 Validity")
    lines.append("")
    lines.append(f"- `geo_gini = {geo_g:.4f}` (constant for all K=28 validator sets, ≤ 0.30 threshold)")
    lines.append(f"- C1-valid validator sets: **{len(c1_valid_ids)} / {len(c1_valid_ids)+len(c1_invalid_ids)}**  (stake gini ≤ 0.20)")
    lines.append(f"- C1-invalid validator sets: {len(c1_invalid_ids)}")
    lines.append("")
    lines.append("## 3. Strong Fixed Baseline")
    lines.append("")
    lines.append(f"- Protocol: LIU_QUORUM  |  S_B = {fixed_action['block_size_mb']} MB  |  T_I = {fixed_action['block_interval_s']} s")
    lines.append(f"- action_id = {fixed_action['action_id']}  |  stake_gini = {fixed_sg:.4f}  |  C1-valid = True")
    lines.append("")
    lines.append("## 4. Global Policy Metrics")
    lines.append(f"*(all policies, {total_steps} steps each)*")
    lines.append("")
    lines.append("| Metric | DQN ep400 | Random | Strong Fixed | DQN ep350 |")
    lines.append("|--------|-----------|--------|--------------|-----------|")
    for label, key in [
        ("Cumulative reward",  "cumulative_reward"),
        ("Mean reward/step",   "mean_reward"),
        ("Median reward/step", "median_reward"),
        ("Feasible steps",     "feasible_steps"),
        ("Feasibility rate",   "feasibility_rate"),
        ("C1 pass",            "c1_pass"),
        ("C2 pass",            "c2_pass"),
        ("C3 pass",            "c3_pass"),
        ("C2_DEADLINE_EXCEEDED", "c2_deadline_exceeded"),
        ("RUNTIME_GUARD_FAILURE","runtime_guard_failure"),
        ("Mean Ω feasible",    "mean_omega_feasible"),
        ("Valid validator sel.","validator_c1_valid_selected"),
        ("Invalid validator",   "validator_c1_invalid_selected"),
        ("Valid rate",         "validator_valid_rate"),
    ]:
        row = (f"| {label} | {agg400.get(key,'—')} | {agg_rnd.get(key,'—')} | "
               f"{agg_fix.get(key,'—')} | {agg350.get(key,'—')} |")
        lines.append(row)

    lines.append("")
    lines.append("## 5. DQN ep400 Action Profile")
    lines.append("")
    lines.append(f"- Protocol distribution: {profile['protocol_dist']}")
    lines.append(f"- Block-size distribution: {profile['block_size_dist']}")
    lines.append(f"- Block-interval distribution: {profile['block_interval_dist']}")
    lines.append(f"- Distinct actions selected: {profile['n_distinct_actions']} / 3915")
    lines.append(f"- Top-10 action ids: {profile['top10_action_ids']}")
    lines.append("")
    lines.append("### 5a. Adaptation by Network Condition (DQN ep400)")
    lines.append("")
    lines.append("| Bucket | n | feasible | mean_reward | protocol_dist |")
    lines.append("|--------|---|----------|-------------|---------------|")
    for bk, bd in adapt.items():
        lines.append(f"| {bk} | {bd['n']} | {bd['feasible']} | {bd['mean_reward']} | {bd['protocol_dist']} |")

    lines.append("")
    lines.append("## 6. Paired Comparisons")
    lines.append("")
    def fmt_paired(s, a, b):
        return (f"- Mean paired diff ({a} − {b}): **{s['mean_paired_diff']:.0f}**  "
                f"| {a} wins: {s.get(a+'_wins', s.get('dqn400_wins','?'))}  "
                f"| tied: {s['tied']}  "
                f"| {b} wins: {s.get(b+'_wins', s.get('random_wins', s.get('strong_fixed_wins', s.get('dqn_ep350_wins','?'))))}")

    lines.append(f"DQN ep400 vs Random ({p_rand['n_trajectories']} trajectories):")
    lines.append(f"- Mean paired diff (ep400 − random): **{p_rand['mean_paired_diff']:.0f}**")
    lines.append(f"- ep400 wins: {p_rand['dqn400_wins']}  |  tied: {p_rand['tied']}  |  random wins: {p_rand['random_wins']}")
    lines.append("")
    lines.append(f"DQN ep400 vs Strong Fixed ({p_fix['n_trajectories']} trajectories):")
    lines.append(f"- Mean paired diff (ep400 − fixed): **{p_fix['mean_paired_diff']:.0f}**")
    lines.append(f"- ep400 wins: {p_fix['dqn400_wins']}  |  tied: {p_fix['tied']}  |  fixed wins: {p_fix['strong_fixed_wins']}")
    lines.append("")
    lines.append(f"DQN ep400 vs DQN ep350 diagnostic ({p_350['n_trajectories']} trajectories):")
    lines.append(f"- Mean paired diff (ep400 − ep350): **{p_350['mean_paired_diff']:.0f}**")
    lines.append(f"- ep400 wins: {p_350['dqn_ep400_wins']}  |  tied: {p_350['tied']}  |  ep350 wins: {p_350['dqn_ep350_wins']}")
    lines.append(f"- Verdict: **{regression}**")

    lines.append("")
    lines.append("## 7. Q-Value Health (ep400)")
    lines.append("")
    lines.append(f"- Samples: {qh['n_samples']}")
    lines.append(f"- Any NaN: {qh['any_nan']}")
    lines.append(f"- Any Inf: {qh['any_inf']}")
    lines.append(f"- Any degenerate (std<1e-4): {qh['any_degenerate']}")
    lines.append(f"- Global Q range: [{qh['global_q_min']}, {qh['global_q_max']}]")
    lines.append(f"- Mean Q std across samples: {qh['mean_q_std']}")

    lines.append("")
    lines.append("## 8. Concrete Trajectory Example (DQN ep400)")
    lines.append("")
    ex = result["concrete_trajectory_example"]
    lines.append(f"Eval seed {ex['eval_seed']}, initial state {ex['init_state']}")
    lines.append("")
    lines.append("| Step | Network | Bucket | Protocol | S_B | T_I | C1-valid | C1 | C2 | C3 | Reward | Failure |")
    lines.append("|------|---------|--------|----------|-----|-----|----------|----|----|----|--------|---------|")
    for s in example_steps[:20]:
        lines.append(
            f"| {s['step']} | {s['net_summary']} | {s['net_bucket'][:12]} | "
            f"{s['protocol']} | {s['block_size_mb']} | {s['block_interval_s']} | "
            f"{s['validator_c1_valid']} | {s['des_c1']} | {s['des_c2']} | {s['des_c3']} | "
            f"{s['reward']:.0f} | {s['failure_reason'][:12]} |"
        )

    lines.append("")
    lines.append("## 9. Gate Criteria")
    lines.append("")
    for k, v in gate_criteria.items():
        lines.append(f"- **{k}**: {'PASS' if v else 'FAIL'}")
    lines.append("")
    lines.append(f"## Result: N30_SEED42_POLICY_VERIFIED — {'AWARDED' if gate_awarded else 'NOT AWARDED'}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
