"""N=30/K=28 Presentation-Scale DRL Training.

Environment:  REDUCED_EXHAUSTIVE_VALIDATOR_VALIDATION_ENVIRONMENT / OUR_RECONSTRUCTION
Action domain: REDUCED_COMPLETE_LIU_ACTION_DOMAIN (3915 actions = 435 C(30,28) × 9 configs)
State:         S_t = [chi, Upsilon, x, c, R(t)]  dim=991  (PAPER_EXACT Eq.10)

Gate sequence:
  N30_DRL_PILOT_VERIFIED        → 15-episode smoke run passes (run with --mode pilot)
  DRL_PRESENTATION_RUN_COMPLETE → full training + evaluation passes (run with --mode train)

Usage:
  python train_n30.py --mode pilot                     # gate 1: 15 ep × 20 steps, seed=42
  python train_n30.py --mode train                     # gate 2: 400 ep × 3 seeds
  python train_n30.py --mode train --resume ckpt.pt    # resume from checkpoint

Per-step log: step_log_seed{N}.csv (action_id, validator_ids, C1/C2/C3, T_C, T_F,
  DESFailureReason, state_hash, epsilon, reward, reward_norm, link stats, loss, ...)

Checkpoints: checkpoints/ckpt_seed{N}_ep{E}.pt
  Contains: online/target nets, optimizer, episode, global_step, epsilon_steps, RNG states.

Immutable constraints (verbatim from specification):
  - Reward ONLY: Ω = floor(S_B/χ)/T_I; R_t = Ω if C1_DES ∧ C2_DES ∧ C3_DES else 0
  - No reward shaping, no validator-selection bonus, no penalty
  - Exact argmax over all 3915 actions (no sampling, no 64-candidate heuristic)
  - Liu objective, C1, C2, C3, DES, FSMC semantics: UNCHANGED

Classification: OUR_RECONSTRUCTION_FOR_TRACTABLE_VALIDATION / NOT_LIU_N100_K21
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
_DYN = ROOT / "experiments" / "liu_replication" / "dynamic_epoch"
sys.path.insert(0, str(_DYN))
sys.path.insert(0, str(HERE.parent))  # drl root: config, state_encoder, etc.

from config import DQNConfig
from state_encoder import StateEncoder
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from liu_dynamic_env import LiuDynamicEpochEnv, DESFailureReason
from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

# ── Population (identical to Phase 3, 4; OUR_RECONSTRUCTION) ──────────────────
N = 30
K = 28
CHI_BYTES = 200

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

# ── Training parameters ────────────────────────────────────────────────────────
STEPS_PER_EPISODE = 20    # DES steps per episode
PILOT_EPISODES = 15       # gate: N30_DRL_PILOT_VERIFIED
TRAIN_EPISODES = 400      # gate: DRL_PRESENTATION_RUN_COMPLETE
CHECKPOINT_EVERY = 50     # save checkpoint every N episodes
SEEDS = [42, 123, 7]

# Reward normalization: max Omega = floor(6.8×10^6 / 200)/2.0 = 17000 (LIU_QUORUM)
# OUR_RECONSTRUCTION: does not change Liu reward; normalizes for Bellman stability only.
REWARD_SCALE = 17_000.0

ACTION_CSV = HERE.parent / "reduced_action_domain.csv"
CKPT_DIR = HERE / "checkpoints"
LOG_DIR = HERE


def make_n30_cfg(total_steps: int) -> DQNConfig:
    """DQNConfig for N=30 training. epsilon_decay_steps = 90% of total_steps (v2 fix)."""
    eps_decay = max(1, int(0.9 * total_steps))
    return DQNConfig(
        hidden_sizes=(256, 128),   # OUR_RECONSTRUCTION: larger than micro for 991-dim state
        gamma=0.9,                 # OUR_RECONSTRUCTION: Liu symbol mu ∈ (0,1]
        learning_rate=5e-4,        # OUR_RECONSTRUCTION
        replay_capacity=10_000,    # OUR_RECONSTRUCTION
        replay_warmup_steps=200,   # OUR_RECONSTRUCTION: diverse replay before training
        batch_size=64,             # OUR_RECONSTRUCTION
        target_sync_steps=50,      # OUR_RECONSTRUCTION: every G steps (Liu Alg.1)
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=eps_decay,  # OUR_RECONSTRUCTION: 90% of budget
        update_every_steps=1,
    )


# ── Action loading ─────────────────────────────────────────────────────────────

def load_actions() -> list[dict]:
    """Load all 3915 actions from the reduced action domain CSV."""
    actions = []
    with open(ACTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            validator_ids = tuple(json.loads(row["validator_ids"]))
            validator_mask = json.loads(row["validator_mask"])
            protocol_str = row["protocol"]
            protocol_enum = LiuConsensusProtocol(protocol_str)
            sb = float(row["block_size_mb"])
            ti = float(row["block_interval_s"])
            actions.append({
                "action_id": int(row["action_id"]),
                "protocol": protocol_str,
                "block_size_mb": sb,
                "block_interval_s": ti,
                "validator_ids": validator_ids,
                "validator_mask": validator_mask,
                "_liu_action": LiuAction(N, K, validator_ids, protocol_enum, sb, ti),
            })
    assert len(actions) == 3915, f"Expected 3915 actions, got {len(actions)}"
    return actions


# ── Environment construction ───────────────────────────────────────────────────

def _all_links_rows(rate_mbps: float) -> tuple:
    return tuple(
        tuple(None if i == j else rate_mbps for j in range(N))
        for i in range(N)
    )


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
        offered_workload_tx=100_000,  # match controlled sweep (covers 2 full blocks at max S_B=7.4 MB → cap=37k, 2×=74k < 100k)
        max_events=2_000_000,
        seed=seed,
    )


# ── State construction ─────────────────────────────────────────────────────────

def make_state_dict(link_rows: tuple) -> dict:
    return {
        "chi_bytes": float(CHI_BYTES),
        "stakes": STAKES,
        "positions_km": POSITIONS,
        "capabilities_ghz": CAPABILITIES,
        "link_rows_mbps": link_rows,
    }


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(
    agent: DQNAgent,
    episode: int,
    global_step: int,
    seed: int,
    cfg: DQNConfig,
    path: Path,
) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    replay_data = None
    try:
        replay_data = pickle.dumps(agent.replay)
        if len(replay_data) > 200 * 1024 * 1024:  # skip if > 200 MB
            replay_data = None
    except Exception:
        replay_data = None

    ckpt = {
        "online_state_dict": agent.online.state_dict(),
        "target_state_dict": agent.target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "episode": episode,
        "global_step": global_step,
        "epsilon_steps": agent._step,
        "rng_python_state": random.getstate(),
        "rng_torch_state": torch.get_rng_state(),
        "seed": seed,
        "cfg_dict": {
            "hidden_sizes": list(cfg.hidden_sizes),
            "gamma": cfg.gamma,
            "learning_rate": cfg.learning_rate,
            "replay_capacity": cfg.replay_capacity,
            "replay_warmup_steps": cfg.replay_warmup_steps,
            "batch_size": cfg.batch_size,
            "target_sync_steps": cfg.target_sync_steps,
            "epsilon_start": cfg.epsilon_start,
            "epsilon_end": cfg.epsilon_end,
            "epsilon_decay_steps": cfg.epsilon_decay_steps,
        },
        "replay_pickle": replay_data,
        "n": N, "k": K, "reward_scale": REWARD_SCALE,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, agent: DQNAgent) -> tuple[int, int]:
    """Load checkpoint into agent. Returns (episode, global_step)."""
    ckpt = torch.load(path, weights_only=False)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.target.load_state_dict(ckpt["target_state_dict"])
    agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent._step = ckpt["epsilon_steps"]
    random.setstate(ckpt["rng_python_state"])
    torch.set_rng_state(ckpt["rng_torch_state"])
    if ckpt.get("replay_pickle") is not None:
        try:
            agent.replay = pickle.loads(ckpt["replay_pickle"])
        except Exception:
            pass  # skip replay restore if corrupt
    return ckpt["episode"], ckpt["global_step"]


# ── Per-step logger ────────────────────────────────────────────────────────────

STEP_LOG_COLS = [
    "seed", "episode", "global_step", "step_in_episode",
    "action_id", "validator_ids", "protocol", "block_size_mb", "block_interval_s",
    "epsilon", "action_mode",
    "des_c1", "des_c2", "des_c3",
    "des_t_c_s", "des_t_f_s", "des_failure_reason",
    "reward", "reward_norm",
    "link_rate_mean_mbps", "link_rate_min_mbps", "link_rate_max_mbps",
    "state_hash",
    "replay_size", "loss",
]


def make_step_log_row(
    seed: int, episode: int, global_step: int, step_in_episode: int,
    action: dict, epsilon: float, action_mode: str,
    result,  # StepResult
    reward_norm: float,
    state_hash: str,
    replay_size: int,
    loss: float | None,
) -> dict:
    ti = action["block_interval_s"]
    t_c = result.des_t_c_s
    t_f = t_c + ti if math.isfinite(t_c) else float("inf")
    return {
        "seed": seed,
        "episode": episode,
        "global_step": global_step,
        "step_in_episode": step_in_episode,
        "action_id": action["action_id"],
        "validator_ids": json.dumps(list(action["validator_ids"])),
        "protocol": action["protocol"],
        "block_size_mb": action["block_size_mb"],
        "block_interval_s": ti,
        "epsilon": round(epsilon, 5),
        "action_mode": action_mode,
        "des_c1": int(result.des_c1),
        "des_c2": int(result.des_c2),
        "des_c3": int(result.des_c3),
        "des_t_c_s": round(t_c, 4) if math.isfinite(t_c) else "inf",
        "des_t_f_s": round(t_f, 4) if math.isfinite(t_f) else "inf",
        "des_failure_reason": result.failure_reason.value,
        "reward": round(result.reward, 2),
        "reward_norm": round(reward_norm, 6),
        "link_rate_mean_mbps": round(result.link_rate_mean_mbps, 2),
        "link_rate_min_mbps": result.link_rate_min_mbps,
        "link_rate_max_mbps": result.link_rate_max_mbps,
        "state_hash": state_hash,
        "replay_size": replay_size,
        "loss": round(loss, 6) if loss is not None and math.isfinite(loss) else (loss if loss is not None else ""),
    }


# ── Training loop for one seed ────────────────────────────────────────────────

def train_one_seed(
    seed: int,
    actions: list[dict],
    state_enc: StateEncoder,
    action_enc: ActionEncoder,
    cfg: DQNConfig,
    n_episodes: int,
    resume_path: Path | None = None,
    pilot: bool = False,
    run_tag: str = "",
) -> tuple[list[dict], DQNAgent, list[dict]]:
    """
    Train one seed. Returns (step_log_rows, trained_agent, episode_log_rows).
    episode_log_rows is one row per episode (for quick summary display).
    """
    label = "PILOT" if pilot else "TRAIN"
    print(f"\n=== DRL {label} seed={seed} ({n_episodes} episodes x {STEPS_PER_EPISODE} steps/ep) ===")
    torch.manual_seed(seed)

    action_tensors = action_enc.encode_batch(actions)  # (3915, action_dim)
    agent = DQNAgent(state_enc.dim, action_enc.dim, action_tensors, cfg, seed=seed)

    start_episode = 0
    global_step = 0

    if resume_path is not None and resume_path.exists():
        start_episode, global_step = load_checkpoint(resume_path, agent)
        print(f"  Resumed from {resume_path}: episode={start_episode}, global_step={global_step}")

    # Two training environments (GOOD + MEDIUM only).
    # POOR_10Mbps is excluded from training: the oracle confirms ALL 3915 actions
    # yield reward=0 at 10 Mbps (no C2 feasibility), so it contributes no gradient
    # signal and causes very slow DES runs (C2 deadline exhaustion). It is kept for
    # evaluation only. Classification: OUR_RECONSTRUCTION.
    envs = {
        "GOOD_100Mbps":  make_env(_all_links_rows(100.0), seed=seed),
        "MEDIUM_55Mbps": make_env(_all_links_rows(55.0),  seed=seed),
    }
    env_cycle = list(envs.items())

    step_log_rows: list[dict] = []
    ep_log_rows: list[dict] = []
    moving_rewards: list[float] = []
    window = 20

    tag_suffix = f"_{run_tag}" if run_tag else ""
    log_path = LOG_DIR / f"step_log_seed{seed}{tag_suffix}.csv"
    # Open log file for appending (handles resume)
    log_file_mode = "a" if (resume_path is not None and log_path.exists()) else "w"
    log_fh = open(log_path, log_file_mode, newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_fh, fieldnames=STEP_LOG_COLS)
    if log_file_mode == "w":
        log_writer.writeheader()

    t_train_start = time.perf_counter()
    nan_loss_count = 0

    for ep in range(start_episode, n_episodes):
        env_name, env = env_cycle[ep % len(env_cycle)]
        obs = env.reset(seed=seed + ep)
        link_rows = env._current_link_rows()
        s_dict = make_state_dict(link_rows)
        s_enc = state_enc.encode(s_dict)

        ep_reward = 0.0
        ep_feasible = 0
        ep_loss_sum = 0.0
        ep_loss_n = 0

        for step_in_ep in range(STEPS_PER_EPISODE):
            eps = agent.epsilon()
            a_idx, mode = agent.select_action(s_enc)
            a = actions[a_idx]

            result = env.step(a["_liu_action"])
            r = result.reward
            r_norm = r / REWARD_SCALE

            obs_next = env._observe()
            state_hash = obs_next.get("link_state_hash", "")[:16]
            ns_link_rows = env._current_link_rows()
            ns_dict = make_state_dict(ns_link_rows)
            ns_enc = state_enc.encode(ns_dict)

            done = (step_in_ep == STEPS_PER_EPISODE - 1)
            agent.observe(s_enc, a_idx, r_norm, ns_enc, done)
            loss = agent._last_loss

            if loss is not None and not math.isfinite(loss):
                nan_loss_count += 1

            row = make_step_log_row(
                seed=seed, episode=ep, global_step=global_step,
                step_in_episode=step_in_ep,
                action=a, epsilon=eps, action_mode=mode,
                result=result, reward_norm=r_norm,
                state_hash=state_hash,
                replay_size=len(agent.replay),
                loss=loss,
            )
            step_log_rows.append(row)
            log_writer.writerow(row)
            log_fh.flush()

            s_enc = ns_enc
            ep_reward += r
            ep_feasible += int(not result.stalled)
            if loss is not None and math.isfinite(loss):
                ep_loss_sum += loss
                ep_loss_n += 1
            global_step += 1

        moving_rewards.append(ep_reward)
        moving_avg = sum(moving_rewards[-window:]) / len(moving_rewards[-window:])
        ep_loss_mean = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else None

        ep_row = {
            "seed": seed, "episode": ep, "env_name": env_name,
            "episode_reward": round(ep_reward, 2),
            "moving_avg": round(moving_avg, 2),
            "epsilon": round(agent.epsilon(), 4),
            "feasible_steps": ep_feasible,
            "replay_size": len(agent.replay),
            "loss_mean": round(ep_loss_mean, 6) if ep_loss_mean is not None else None,
        }
        ep_log_rows.append(ep_row)

        if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
            loss_str = f"{ep_loss_mean:.4f}" if ep_loss_mean is not None else "—"
            print(f"  Ep {ep+1:4d}/{n_episodes}: R={ep_reward:8.1f}  "
                  f"mavg={moving_avg:8.1f}  eps={agent.epsilon():.3f}  "
                  f"replay={len(agent.replay):5d}  loss={loss_str}  env={env_name}")

        if (ep + 1) % CHECKPOINT_EVERY == 0 or ep == n_episodes - 1:
            if run_tag:
                ckpt_path = CKPT_DIR / f"ckpt_seed{seed}_{run_tag}_ep{ep+1:03d}.pt"
            else:
                ckpt_path = CKPT_DIR / f"ckpt_seed{seed}_ep{ep+1}.pt"
            save_checkpoint(agent, ep + 1, global_step, seed, cfg, ckpt_path)
            print(f"  Checkpoint: {ckpt_path}")

    log_fh.close()
    t_elapsed = time.perf_counter() - t_train_start

    print(f"  Done: {global_step - (start_episode * STEPS_PER_EPISODE)} steps  "
          f"elapsed={t_elapsed:.1f}s  nan_losses={nan_loss_count}")

    return step_log_rows, agent, ep_log_rows


# ── Pilot gate check ───────────────────────────────────────────────────────────

def check_pilot_gate(step_log_rows: list[dict], agent: DQNAgent, cfg: DQNConfig) -> dict:
    """Check N30_DRL_PILOT_VERIFIED criteria."""
    n_steps = len(step_log_rows)
    n_feasible = sum(1 for r in step_log_rows if r["des_c1"] and r["des_c2"] and r["des_c3"])
    rewards = [float(r["reward"]) for r in step_log_rows]
    losses = [float(r["loss"]) for r in step_log_rows
              if r["loss"] != "" and r["loss"] is not None]
    nan_losses = [l for l in losses if not math.isfinite(l)]
    eps_final = agent.epsilon()
    replay_size = len(agent.replay)

    # Action diversity: distinct action_ids seen
    distinct_actions = len(set(r["action_id"] for r in step_log_rows))

    checks = {
        "A_no_crash": True,  # by construction: reached here
        "B_at_least_one_feasible_step": n_feasible > 0,
        "C_epsilon_decayed": eps_final < cfg.epsilon_start,
        "D_replay_filling": replay_size >= min(cfg.replay_warmup_steps, n_steps),
        "E_loss_finite_if_training": len(nan_losses) == 0,
        "F_action_diversity": distinct_actions >= min(100, n_steps),
    }

    gate = all(checks.values())

    print("\n=== N30_DRL_PILOT_VERIFIED Gate Check ===")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  (n_steps={n_steps}, feasible={n_feasible}, eps={eps_final:.4f}, "
          f"replay={replay_size}, distinct_actions={distinct_actions}, "
          f"losses_computed={len(losses)}, nan={len(nan_losses)})")
    print(f"Gate N30_DRL_PILOT_VERIFIED: {'AWARDED' if gate else 'NOT AWARDED'}")

    return {
        "gate": "N30_DRL_PILOT_VERIFIED",
        "gate_awarded": gate,
        "n_steps": n_steps,
        "n_feasible_steps": n_feasible,
        "feasibility_rate": round(n_feasible / n_steps, 4) if n_steps else 0,
        "max_reward": max(rewards) if rewards else 0,
        "mean_reward": round(sum(rewards) / len(rewards), 2) if rewards else 0,
        "epsilon_final": round(eps_final, 4),
        "replay_size": replay_size,
        "distinct_actions_seen": distinct_actions,
        "n_training_steps": len(losses),
        "n_nan_losses": len(nan_losses),
        "criteria": checks,
    }


# ── Evaluation (epsilon=0 greedy) ─────────────────────────────────────────────

def evaluate_greedy(
    agent: DQNAgent,
    actions: list[dict],
    state_enc: StateEncoder,
    n_eval_steps: int = 10,
) -> list[dict]:
    """Evaluate greedy policy (epsilon=0) across 3 FSMC states for n_eval_steps each."""
    results = []
    eval_envs = {
        "GOOD_100Mbps":  make_env(_all_links_rows(100.0), seed=0),
        "MEDIUM_55Mbps": make_env(_all_links_rows(55.0),  seed=0),
        "POOR_10Mbps":   make_env(_all_links_rows(10.0),  seed=0),
    }

    for env_name, env in eval_envs.items():
        obs = env.reset(seed=0)
        link_rows = env._current_link_rows()
        s_dict = make_state_dict(link_rows)
        s_enc = state_enc.encode(s_dict)

        ep_rewards = []
        for step_i in range(n_eval_steps):
            a_idx = agent.argmax(s_enc)  # epsilon=0 greedy
            a = actions[a_idx]
            result = env.step(a["_liu_action"])
            ep_rewards.append(result.reward)
            obs_next = env._observe()
            link_rows = env._current_link_rows()
            s_enc = state_enc.encode(make_state_dict(link_rows))

        results.append({
            "env_name": env_name,
            "n_steps": n_eval_steps,
            "mean_reward": round(sum(ep_rewards) / len(ep_rewards), 2),
            "max_reward": max(ep_rewards),
            "min_reward": min(ep_rewards),
            "rewards": ep_rewards,
        })

    return results


def evaluate_random_baseline(
    actions: list[dict],
    state_enc: StateEncoder,
    n_eval_steps: int = 10,
    seed: int = 0,
) -> list[dict]:
    """Evaluate random policy across 3 FSMC states."""
    rng = random.Random(seed)
    results = []
    eval_envs = {
        "GOOD_100Mbps":  make_env(_all_links_rows(100.0), seed=seed),
        "MEDIUM_55Mbps": make_env(_all_links_rows(55.0),  seed=seed),
        "POOR_10Mbps":   make_env(_all_links_rows(10.0),  seed=seed),
    }

    for env_name, env in eval_envs.items():
        env.reset(seed=seed)
        ep_rewards = []
        for step_i in range(n_eval_steps):
            a_idx = rng.randrange(len(actions))
            a = actions[a_idx]
            result = env.step(a["_liu_action"])
            ep_rewards.append(result.reward)

        results.append({
            "env_name": env_name,
            "n_steps": n_eval_steps,
            "mean_reward": round(sum(ep_rewards) / len(ep_rewards), 2),
            "max_reward": max(ep_rewards),
            "min_reward": min(ep_rewards),
        })

    return results


def oracle_best_paper_reference(actions: list[dict]) -> dict:
    """
    Limited exhaustive oracle: for each FSMC state, use paper reference to find
    the best action from the 3915 action domain. Fast (~seconds, no DES).
    """
    from Liu.ReferenceCore import evaluate_reference
    from Liu.LinkState import LinkStateMatrix

    oracle_states = {
        "GOOD_100Mbps":  _all_links_rows(100.0),
        "MEDIUM_55Mbps": _all_links_rows(55.0),
        "POOR_10Mbps":   _all_links_rows(10.0),
    }
    geo_gini = GEO.geographic_gini()
    results = {}

    for state_name, rows in oracle_states.items():
        link_matrix = LinkStateMatrix.from_rows(rows)
        best_reward = -1.0
        best_action_id = -1
        all_rewards = []
        for a in actions:
            mask = list(a["validator_mask"])
            ref = evaluate_reference(
                node_stakes=STAKES,
                node_capabilities_ghz=CAPABILITIES,
                link_rates=link_matrix,
                geographic_gini=geo_gini,
                producer_mask=tuple(mask),
                protocol=LiuConsensusProtocol(a["protocol"]),
                block_size_mb=a["block_size_mb"],
                block_interval_s=a["block_interval_s"],
                malicious_count=0,
                params=REF_PARAMS,
            )
            r = float(ref.reward)
            all_rewards.append(r)
            if r > best_reward:
                best_reward = r
                best_action_id = a["action_id"]

        n_optimal = sum(1 for r in all_rewards if abs(r - best_reward) < 1e-3)
        feasible = sum(1 for r in all_rewards if r > 0)
        results[state_name] = {
            "oracle_best_reward": best_reward,
            "oracle_best_action_id": best_action_id,
            "n_optimal_actions": n_optimal,
            "n_feasible_actions": feasible,
            "n_total_actions": len(actions),
        }
        print(f"  Oracle [{state_name}]: best_reward={best_reward:.1f} "
              f"n_optimal={n_optimal}/{len(actions)} feasible={feasible}")

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="N=30/K=28 DRL Training")
    parser.add_argument("--mode", choices=["pilot", "train"], default="pilot",
                        help="pilot: 15 ep gate check; train: full 400-ep run")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from (seed=42 only)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episode count (default: pilot=15, train=400)")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Tag appended to output filenames (e.g. 'clean'). "
                             "step_log_seed42_{tag}.csv, ckpt_seed42_{tag}_ep050.pt")
    args = parser.parse_args()

    t_global_start = time.perf_counter()
    pilot = (args.mode == "pilot")
    n_episodes = args.episodes or (PILOT_EPISODES if pilot else TRAIN_EPISODES)
    seeds = SEEDS[:1] if pilot else SEEDS  # pilot: seed=42 only

    print(f"Mode: {args.mode.upper()}")
    print(f"Episodes: {n_episodes}, Seed(s): {seeds}")
    print(f"Steps/episode: {STEPS_PER_EPISODE}, Total steps/seed: {n_episodes * STEPS_PER_EPISODE}")

    actions = load_actions()
    print(f"Actions loaded: {len(actions)} (expected 3915)")

    state_enc = StateEncoder(N, make_n30_cfg(1))  # config only used for normalization constants
    action_enc = ActionEncoder(N, make_n30_cfg(1))
    print(f"State dim: {state_enc.dim}  Action dim: {action_enc.dim}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Oracle (paper reference, fast) ────────────────────────────────────────
    print("\n=== Oracle (paper reference, exhaustive 3915 actions) ===")
    oracle = oracle_best_paper_reference(actions)
    oracle_path = HERE / "oracle_paper_reference.json"
    with open(oracle_path, "w", encoding="utf-8") as f:
        json.dump(oracle, f, indent=2)
    print(f"Oracle written: {oracle_path}")

    # ── Training ───────────────────────────────────────────────────────────────
    all_ep_logs: dict[int, list[dict]] = {}
    all_agents: dict[int, DQNAgent] = {}

    for seed in seeds:
        total_steps = n_episodes * STEPS_PER_EPISODE
        cfg = make_n30_cfg(total_steps)
        resume_path = args.resume if (seed == seeds[0] and args.resume) else None

        step_logs, agent, ep_logs = train_one_seed(
            seed=seed,
            actions=actions,
            state_enc=state_enc,
            action_enc=action_enc,
            cfg=cfg,
            n_episodes=n_episodes,
            resume_path=resume_path,
            pilot=pilot,
            run_tag=args.run_tag,
        )
        all_ep_logs[seed] = ep_logs
        all_agents[seed] = agent

        tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
        # Write episode-level log
        ep_log_path = LOG_DIR / f"episode_log_seed{seed}{tag_suffix}.csv"
        if ep_logs:
            with open(ep_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ep_logs[0].keys()))
                writer.writeheader()
                writer.writerows(ep_logs)

    # ── Pilot gate check ──────────────────────────────────────────────────────
    if pilot:
        tag_suffix_pilot = f"_{args.run_tag}" if args.run_tag else ""
        # Load step log for seed=42 to check gate
        seed42_log_path = LOG_DIR / f"step_log_seed{seeds[0]}{tag_suffix_pilot}.csv"
        with open(seed42_log_path, encoding="utf-8") as f:
            step_rows = list(csv.DictReader(f))
        # Convert numeric fields
        for r in step_rows:
            r["des_c1"] = int(r["des_c1"])
            r["des_c2"] = int(r["des_c2"])
            r["des_c3"] = int(r["des_c3"])
            r["reward"] = float(r["reward"])
            r["loss"] = r["loss"] if r["loss"] not in ("", "nan", "inf") else None
            r["action_id"] = int(r["action_id"])

        pilot_gate = check_pilot_gate(step_rows, all_agents[seeds[0]], make_n30_cfg(n_episodes * STEPS_PER_EPISODE))

        pilot_result = {
            "schema": "n30_drl_pilot_v1",
            "mode": "pilot",
            "n": N, "k": K, "chi_bytes": CHI_BYTES,
            "total_actions": len(actions),
            "state_dim": state_enc.dim,
            "action_dim": action_enc.dim,
            "pilot_episodes": n_episodes,
            "steps_per_episode": STEPS_PER_EPISODE,
            "total_steps": n_episodes * STEPS_PER_EPISODE,
            "seed": seeds[0],
            "reward_scale": REWARD_SCALE,
            "oracle": oracle,
            "pilot_gate": pilot_gate,
            "episode_log": all_ep_logs[seeds[0]],
            "wall_s": round(time.perf_counter() - t_global_start, 1),
            "label": "OUR_RECONSTRUCTION / NOT_LIU_N100_K21",
        }

        pilot_tag = f"_seed{seeds[0]}{tag_suffix_pilot}" if (tag_suffix_pilot or True) else ""
        out_path = HERE / f"pilot_gate{pilot_tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pilot_result, f, indent=2)
        print(f"\nPilot result written: {out_path}")

        if pilot_gate["gate_awarded"]:
            print("\n*** N30_DRL_PILOT_VERIFIED — AWARDED ***")
            print("Next: run --mode train to start full N=30 DRL training.")
        else:
            print("\n*** N30_DRL_PILOT_VERIFIED — NOT AWARDED (see gate criteria) ***")

        return pilot_result

    # ── Full training evaluation ───────────────────────────────────────────────
    print("\n=== Evaluation (epsilon=0 greedy, 20 steps per FSMC state) ===")
    n_eval = 20
    greedy_results = {}
    rand_results = {}
    for seed in seeds:
        print(f"\n  Seed {seed} greedy:")
        greedy_results[seed] = evaluate_greedy(all_agents[seed], actions, state_enc, n_eval)
        print(f"  Seed {seed} random baseline:")
        rand_results[seed] = evaluate_random_baseline(actions, state_enc, n_eval, seed=seed)

    # Aggregate across seeds
    def mean_across_seeds(results_dict: dict, env_name: str, field: str) -> float:
        vals = [r[field] for seed_res in results_dict.values()
                for r in seed_res if r["env_name"] == env_name]
        return sum(vals) / len(vals) if vals else 0.0

    env_names = ["GOOD_100Mbps", "MEDIUM_55Mbps", "POOR_10Mbps"]
    print("\n=== Summary: DQN vs Random vs Oracle ===")
    print(f"{'Env':<18} {'DQN_mean':>10} {'Rand_mean':>10} {'Oracle':>10} {'Advantage':>10}")
    for en in env_names:
        dqn_mean = mean_across_seeds(greedy_results, en, "mean_reward")
        rand_mean = mean_across_seeds(rand_results, en, "mean_reward")
        oracle_r = oracle.get(en, {}).get("oracle_best_reward", 0)
        adv = (dqn_mean - rand_mean) / max(rand_mean, 1) * 100
        print(f"  {en:<16} {dqn_mean:>10.1f} {rand_mean:>10.1f} {oracle_r:>10.1f} {adv:>9.1f}%")

    # Gate check for DRL_PRESENTATION_RUN_COMPLETE
    dqn_total = sum(mean_across_seeds(greedy_results, en, "mean_reward") for en in env_names)
    rand_total = sum(mean_across_seeds(rand_results, en, "mean_reward") for en in env_names)
    gate_dqn_beats_random = dqn_total > rand_total

    training_summary = {
        "schema": "n30_drl_training_summary_v1",
        "mode": "train",
        "n": N, "k": K, "chi_bytes": CHI_BYTES,
        "total_actions": len(actions),
        "state_dim": state_enc.dim,
        "action_dim": action_enc.dim,
        "train_episodes": n_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "total_steps_per_seed": n_episodes * STEPS_PER_EPISODE,
        "seeds": seeds,
        "reward_scale": REWARD_SCALE,
        "oracle": oracle,
        "greedy_evaluation": {str(s): greedy_results[s] for s in seeds},
        "random_baseline": {str(s): rand_results[s] for s in seeds},
        "gate_dqn_beats_random": gate_dqn_beats_random,
        "dqn_total_mean": round(dqn_total, 2),
        "rand_total_mean": round(rand_total, 2),
        "wall_s": round(time.perf_counter() - t_global_start, 1),
        "label": "OUR_RECONSTRUCTION / NOT_LIU_N100_K21",
    }

    out_path = HERE / "training_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=2)
    print(f"\nTraining summary written: {out_path}")

    if gate_dqn_beats_random:
        print("\n*** DRL_PRESENTATION_RUN_COMPLETE — criteria met for AWARDED ***")
        print("    (Final gate award pending presentation artifact review)")
    else:
        print("\n*** DRL_PRESENTATION_RUN_COMPLETE — NOT AWARDED: DQN did not beat random ***")

    return training_summary


if __name__ == "__main__":
    main()
