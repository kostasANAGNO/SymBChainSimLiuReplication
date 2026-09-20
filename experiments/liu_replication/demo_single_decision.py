"""Liu Replication — Single-Decision Demo.

Demonstrates one DRL decision using the final ep400 DQN checkpoint:
  S(t)  → DQN greedy argmax → A(t) → SymBChainSim DES → C1/C2/C3 → reward

Usage:
    python experiments/liu_replication/demo_single_decision.py
    python experiments/liu_replication/demo_single_decision.py --compare

The demo uses the SAME population fixture, action domain, environment,
and checkpoint as the final N30 policy evaluation (eval_n30_seed42.py).
No scientific logic is duplicated here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch

# ── Path setup (same pattern as eval_n30_seed42.py) ──────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                           # repo root
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
_DYN = ROOT / "experiments" / "liu_replication" / "dynamic_epoch"
_DRL = ROOT / "experiments" / "liu_replication" / "drl"
sys.path.insert(0, str(_DYN))
sys.path.insert(0, str(_DRL))

from config import DQNConfig
from state_encoder import StateEncoder
from action_encoder import ActionEncoder
from dqn_agent import DQNAgent
from liu_dynamic_env import LiuDynamicEpochEnv, DESFailureReason
from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

# ── Population fixture (identical to train_n30 / eval_n30_seed42) ─────────────
N = 30
K = 28
CHI_BYTES = 200
REWARD_SCALE = 17_000.0

STAKES       = tuple(float(4 + (i % 10)) for i in range(N))
CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(N))
POSITIONS    = tuple((0.2 * (i % 5), 0.2 * (i // 5)) for i in range(N))

# Default initial state: ALL_HIGH_100 (deterministic, favourable for presentation)
INITIAL_LINK_ROWS_HIGH = tuple(
    tuple(None if i == j else 100.0 for j in range(N))
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

CKPT_PATH   = _DRL / "n30" / "checkpoints" / "ckpt_seed42_clean_ep400.pt"
ACTION_CSV  = _DRL / "reduced_action_domain.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_actions() -> list[dict]:
    actions = []
    with open(ACTION_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            validator_ids = tuple(json.loads(row["validator_ids"]))
            sb = float(row["block_size_mb"])
            ti = float(row["block_interval_s"])
            proto = row["protocol"]
            actions.append({
                "action_id":       int(row["action_id"]),
                "protocol":        proto,
                "block_size_mb":   sb,
                "block_interval_s": ti,
                "validator_ids":   validator_ids,
                "validator_mask":  json.loads(row["validator_mask"]),
                "_liu_action":     LiuAction(N, K, validator_ids,
                                             LiuConsensusProtocol(proto), sb, ti),
            })
    assert len(actions) == 3915
    return actions


def _load_agent(actions: list[dict]) -> tuple[DQNAgent, StateEncoder, ActionEncoder]:
    ckpt = torch.load(CKPT_PATH, weights_only=False)
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
    state_enc  = StateEncoder(N, cfg)
    action_enc = ActionEncoder(N, cfg)
    action_tensors = action_enc.encode_batch(actions)
    agent = DQNAgent(state_enc.dim, action_enc.dim, action_tensors, cfg, seed=0)
    agent.online.load_state_dict(ckpt["online_state_dict"])
    agent.online.eval()
    return agent, state_enc, action_enc


def _make_env(initial_link_rows: tuple, seed: int = 1000) -> LiuDynamicEpochEnv:
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


def _state_dict(link_rows: tuple) -> dict:
    return {
        "chi_bytes":       float(CHI_BYTES),
        "stakes":          STAKES,
        "positions_km":    POSITIONS,
        "capabilities_ghz": CAPABILITIES,
        "link_rows_mbps":  link_rows,
    }


def _sep(label: str = "") -> None:
    print()
    if label:
        print(f"{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
    else:
        print(f"{'─'*60}")


def _omega(sb_mb: float, ti_s: float) -> float:
    return math.floor(sb_mb * 1_000_000 / CHI_BYTES) / ti_s


def _link_stats(link_rows: tuple) -> tuple[float, float, float]:
    rates = [link_rows[i][j] for i in range(N) for j in range(N) if i != j and link_rows[i][j] is not None]
    return sum(rates) / len(rates), min(rates), max(rates)


# ── Single run ────────────────────────────────────────────────────────────────

def run_single(
    actions: list[dict],
    agent: DQNAgent,
    state_enc: StateEncoder,
    action_enc: ActionEncoder,
    initial_link_rows: tuple,
    env_seed: int = 1000,
    *,
    verbose: bool = True,
) -> dict:
    """Execute one DQN decision against the real DES. Returns result dict."""
    env = _make_env(initial_link_rows, seed=env_seed)
    obs = env.reset(seed=env_seed)

    # Encode the current state
    current_link_rows = tuple(
        tuple(None if i == j else initial_link_rows[i][j] for j in range(N))
        for i in range(N)
    )
    mean_r, min_r, max_r = _link_stats(initial_link_rows)
    sd = _state_dict(initial_link_rows)
    s_enc = state_enc.encode(sd)

    # DQN greedy argmax
    with torch.no_grad():
        q_vals = agent.online.q_values_batch(s_enc, agent.all_action_tensors)
    best_idx = int(torch.argmax(q_vals).item())
    best_q   = float(q_vals[best_idx].item())
    chosen   = actions[best_idx]

    if verbose:
        _sep("CURRENT STATE  S(t)")
        print(f"  Transaction size χ : {CHI_BYTES} B              (Table I, PAPER_EXACT)")
        print(f"  Nodes N            : {N}   Validators K: {K}    (OUR_RECONSTRUCTION)")
        print(f"  Link rate mean     : {mean_r:.1f} Mbps")
        print(f"  Link rate min      : {min_r:.1f} Mbps")
        print(f"  Link rate max      : {max_r:.1f} Mbps")
        stake_mean = sum(STAKES) / len(STAKES)
        cap_mean   = sum(CAPABILITIES) / len(CAPABILITIES)
        print(f"  Stakes (mean)      : {stake_mean:.1f} token  range [{min(STAKES):.0f},{max(STAKES):.0f}]")
        print(f"  Capabilities (mean): {cap_mean:.1f} GHz   range [{min(CAPABILITIES):.0f},{max(CAPABILITIES):.0f}]")
        print(f"  State encoding dim : {state_enc.dim}")

        _sep("DQN DECISION  A(t)   [ep400 checkpoint, ε=0 greedy]")
        print(f"  Action ID          : {chosen['action_id']} / 3915")
        print(f"  Protocol δ         : {chosen['protocol']}")
        print(f"  Block size S_B     : {chosen['block_size_mb']:.1f} MB")
        print(f"  Block interval T_I : {chosen['block_interval_s']:.1f} s")
        n_validators = len(chosen["validator_ids"])
        v_sample = list(chosen["validator_ids"])[:6]
        print(f"  Validators (K={n_validators}) : {v_sample} ... (first 6 shown)")
        omega_val = _omega(chosen["block_size_mb"], chosen["block_interval_s"])
        print(f"  Nominal Ω          : {omega_val:,.0f} TPS   (= floor(S_B/χ)/T_I)")
        print(f"  Predicted Q-value  : {best_q:.4f}  (normalised by {REWARD_SCALE:,.0f})")

        _sep("SYMBCHAINSIM EXECUTION")
        print("  Running DES consensus simulation ... ", end="", flush=True)

    result = env.step(chosen["_liu_action"])

    if verbose:
        tc_str = f"{result.des_t_c_s*1000:.2f} ms" if math.isfinite(result.des_t_c_s) else "∞ (deadline exceeded)"
        tf_str = (f"{(chosen['block_interval_s'] + result.des_t_c_s)*1000:.2f} ms"
                  if math.isfinite(result.des_t_c_s) else "—")
        print("done")
        print(f"  T_C_DES            : {tc_str}")
        print(f"  T_F_DES (T_I+T_C)  : {tf_str}")
        print(f"  C2 deadline (ω·T_I): {REF_PARAMS.finality_multiplier_omega * chosen['block_interval_s'] * 1000:.0f} ms")
        print(f"  Stake Gini G(Υ)    : {result.stake_gini:.4f}  (threshold η_s={REF_PARAMS.stake_gini_threshold_eta_s})")
        print(f"  Geo Gini G(λ)      : {result.geo_gini:.4f}  (threshold η_l={REF_PARAMS.geographic_gini_threshold_eta_l})")
        print(f"  Failure reason     : {result.failure_reason.value}")

        _sep("CONSTRAINT CHECK")
        c1 = "✓ PASS" if result.des_c1 else "✗ FAIL"
        c2 = "✓ PASS" if result.des_c2 else "✗ FAIL"
        c3 = "✓ PASS" if result.des_c3 else "✗ FAIL"
        print(f"  C1 Decentralization : {c1}  (G(Υ)≤{REF_PARAMS.stake_gini_threshold_eta_s} ∧ G(λ)≤{REF_PARAMS.geographic_gini_threshold_eta_l})")
        print(f"  C2 Finality         : {c2}  (T_F ≤ ω·T_I = {REF_PARAMS.finality_multiplier_omega}×{chosen['block_interval_s']:.1f}s)")
        print(f"  C3 Fault tolerance  : {c3}  (f=0 ≤ F^δ)")

        _sep("RESULT")
        print(f"  Ω (Liu throughput) : {omega_val:,.0f} TPS")
        print(f"  Liu reward R(t)    : {result.reward:,.0f}  (= Ω if C1∧C2∧C3 else 0)")

        # Next state summary
        obs_next = env._observe()
        _sep("NEXT STATE  S(t+1)  [after FSMC advance]")
        print(f"  Link rate mean     : {obs_next['link_rate_mean_mbps']:.1f} Mbps")
        print(f"  Link rate min      : {obs_next['link_rate_min_mbps']:.1f} Mbps")
        print(f"  Link rate max      : {obs_next['link_rate_max_mbps']:.1f} Mbps")
        print(f"  Link state hash    : {obs_next['link_state_hash'][:16]}...")
        print()

    return {
        "action_id":      chosen["action_id"],
        "protocol":       chosen["protocol"],
        "block_size_mb":  chosen["block_size_mb"],
        "block_interval_s": chosen["block_interval_s"],
        "c1": result.des_c1,
        "c2": result.des_c2,
        "c3": result.des_c3,
        "reward": result.reward,
        "omega": omega_val,
        "t_c_des_s": result.des_t_c_s,
        "failure_reason": result.failure_reason.value,
    }


# ── Compare mode ──────────────────────────────────────────────────────────────

def _pairwise_gini(values: list[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    return sum(abs(a - b) for a in values for b in values) / (2 * n * total)


def run_compare(
    actions: list[dict],
    agent: DQNAgent,
    state_enc: StateEncoder,
    initial_link_rows: tuple,
    env_seed: int = 1000,
) -> None:
    """Compare DQN greedy / random / strong-fixed on the same starting state."""

    # Find strong-fixed action (LIU_QUORUM S_B=6.8 T_I=2.0 first C1-valid set)
    strong_fixed = None
    for a in actions:
        if (a["protocol"] == "LIU_QUORUM"
                and abs(a["block_size_mb"] - 6.8) < 0.01
                and abs(a["block_interval_s"] - 2.0) < 0.01):
            stakes_k = [STAKES[i] for i in a["validator_ids"]]
            if _pairwise_gini(stakes_k) <= 0.2:
                strong_fixed = a
                break

    if strong_fixed is None:
        # Fallback: any LIU_QUORUM 6.8/2.0
        for a in actions:
            if a["protocol"] == "LIU_QUORUM" and abs(a["block_size_mb"] - 6.8) < 0.01:
                strong_fixed = a
                break

    # Random action (seeded deterministically)
    rng = random.Random(env_seed)
    random_action = actions[rng.randrange(len(actions))]

    # DQN action
    sd = _state_dict(initial_link_rows)
    s_enc_t = state_enc.encode(sd)
    with torch.no_grad():
        q_vals = agent.online.q_values_batch(s_enc_t, agent.all_action_tensors)
    dqn_idx = int(torch.argmax(q_vals).item())
    dqn_action = actions[dqn_idx]

    policies = [
        ("DQN greedy", dqn_action),
        ("Random",     random_action),
        ("Strong fixed", strong_fixed),
    ]

    results = []
    for label, a in policies:
        env = _make_env(initial_link_rows, seed=env_seed)
        env.reset(seed=env_seed)
        r = env.step(a["_liu_action"])
        omega = _omega(a["block_size_mb"], a["block_interval_s"])
        results.append({
            "policy":    label,
            "protocol":  a["protocol"],
            "S_B":       a["block_size_mb"],
            "T_I":       a["block_interval_s"],
            "C1":        "PASS" if r.des_c1 else "FAIL",
            "C2":        "PASS" if r.des_c2 else "FAIL",
            "C3":        "PASS" if r.des_c3 else "FAIL",
            "reward":    r.reward,
            "omega":     omega,
        })

    _sep("POLICY COMPARISON  (same S(t), same env seed)")
    hdr = f"  {'Policy':<14} {'Protocol':<12} {'S_B':>5} {'T_I':>4} {'C1':>4} {'C2':>4} {'C3':>4}  {'Reward':>8}  {'Ω':>8}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in results:
        print(f"  {r['policy']:<14} {r['protocol']:<12} {r['S_B']:>5.1f} {r['T_I']:>4.1f} "
              f"{r['C1']:>4} {r['C2']:>4} {r['C3']:>4}  {r['reward']:>8,.0f}  {r['omega']:>8,.0f}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Liu replication — single-decision demo")
    parser.add_argument(
        "--compare", action="store_true",
        help="Run DQN / random / strong-fixed on the same state and print a comparison table.",
    )
    parser.add_argument(
        "--network", choices=["high", "mid", "low", "mixed"],
        default="high",
        help="Initial network condition: high=100Mbps, mid=55Mbps, low=10Mbps, mixed=training-mixed.",
    )
    args = parser.parse_args()

    _MIXED = tuple(
        tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N))
        for i in range(N)
    )
    RATE_MAP = {
        "high":  100.0,
        "mid":   55.0,
        "low":   10.0,
    }

    if args.network == "mixed":
        initial_links = _MIXED
    else:
        rate = RATE_MAP[args.network]
        initial_links = tuple(
            tuple(None if i == j else rate for j in range(N))
            for i in range(N)
        )

    print()
    print("  Liu et al. 2019 — DRL Blockchain Optimisation")
    print("  SymBChainSim Replication  |  ep400 DQN checkpoint  |  N=30 K=28")
    print(f"  Network: {args.network.upper()}  |  Action domain: 3915  |  Seed: 1000")

    print(f"\n  Loading action domain ... ", end="", flush=True)
    actions = _load_actions()
    print(f"{len(actions)} actions loaded")

    print(f"  Loading DQN checkpoint  ... ", end="", flush=True)
    agent, state_enc, action_enc = _load_agent(actions)
    print(f"done  ({CKPT_PATH.name})")

    if args.compare:
        run_compare(actions, agent, state_enc, initial_links)
    else:
        run_single(actions, agent, state_enc, action_enc, initial_links, verbose=True)


if __name__ == "__main__":
    main()
