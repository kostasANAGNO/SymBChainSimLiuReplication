"""Liu 2019 Fig. 5 at paper scale (N=100, K=21), DES-evaluated, node-score validator selection.

    python fig5_n100.py --scheme proposed --episodes 20000 --out runs/fig5_n100 [--val-explore uniform|stake_window]
Same schemes/reward/episode definition as fig5_convergence.py. Resumes from <out>/<tag>.ckpt.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import fig5_convergence as f5          # cfg grid, scheme masks, log truncation
from common_n100 import *             # noqa: F401,F403
from factored_dqn import DQNHyper, action_features
from state_encoder import StateEncoder
from topk_dqn import TopKDQN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["proposed", "no_cas", "fixed_block_size", "fixed_block_interval"])
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/fig5_n100")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--val-explore", choices=["uniform", "stake_window"], default="uniform")
    ap.add_argument("--replay", type=int, default=6000)
    ap.add_argument("--link-profile", choices=sorted(LINK_PROFILES), default="default", help="FSMC link-rate profile (sensitivity run)")
    ap.add_argument("--node-attrs", action="store_true", help="v3: score nodes from their attributes (stake, GHz, x, y)")
    ap.add_argument("--eps-decay", type=int, default=5000, help="episodes over which epsilon decays 1.0 -> 0.02")
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.scheme}_seed{a.seed}"
    log_path, ckpt_path = out / f"{tag}.csv", out / f"{tag}.ckpt"

    enc = StateEncoder(N100)
    hp = DQNHyper(replay_capacity=a.replay, eps_decay_episodes=a.eps_decay)
    order = np.argsort(np.array(STAKES100))
    attrs = None
    if a.node_attrs:
        attrs = torch.tensor([[STAKES100[i] / 50.0, CAPS100[i] / 30.0, POS100[i][0], POS100[i][1]] for i in range(N100)], dtype=torch.float32)
    agent = TopKDQN(enc.dim, f5.N_CFG, N100, K21, f5.scheme_mask(a.scheme), hp, a.seed,
                    action_features(f5.cfg_decode, f5.N_CFG), stake_order=order, val_explore=a.val_explore, node_attrs=attrs)
    env = make_env100(a.seed, link_profile=a.link_profile)
    start = 0
    if ckpt_path.exists():
        ck = pickle.loads(ckpt_path.read_bytes())
        agent.online.load_state_dict(ck["online"]); agent.target.load_state_dict(ck["target"]); agent.opt.load_state_dict(ck["opt"])
        agent.replay, agent.updates, agent.rng = ck["replay"], ck["updates"], ck["rng"]
        env._fsmc, env._rng, env._step_count = ck["fsmc"], ck["env_rng"], ck["env_steps"]
        start = ck["episode"]
        f5._truncate_log(log_path, start)
        print(f"resumed {tag} at episode {start}", flush=True)

    cols = ["episode", "reward", "epsilon", "mode", "protocol", "block_size_mb", "block_interval_s", "stake_gini",
            "c1", "c2", "c3", "failure", "loss", "step_s"]
    new = not log_path.exists() or start == 0
    f = open(log_path, "w" if new else "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    state = lambda: enc.encode({"chi_bytes": 200.0, "stakes": STAKES100, "positions_km": POS100, "capabilities_ghz": CAPS100,
                                "link_rows_mbps": env._current_link_rows()})
    s = state()
    for ep in range(start, a.episodes):
        t0 = time.time()
        cfg, vals, mode = agent.act(s, ep)
        p, sb, ti = f5.cfg_decode(cfg)
        res = env.step(make_action100(vals, p, sb, ti))
        s2 = state()
        loss = agent.observe(s, cfg, vals, res.reward, s2)
        w.writerow({"episode": ep, "reward": round(res.reward, 3), "epsilon": round(agent.epsilon(ep), 4), "mode": mode,
                    "protocol": PROTOCOL_NAMES[p], "block_size_mb": sb, "block_interval_s": ti, "stake_gini": round(res.stake_gini, 4),
                    "c1": int(res.des_c1), "c2": int(res.des_c2), "c3": int(res.des_c3), "failure": res.failure_reason.value,
                    "loss": "" if loss is None else round(loss, 6), "step_s": round(time.time() - t0, 3)})
        s = s2
        if (ep + 1) % 50 == 0:
            f.flush()
        if (ep + 1) % a.ckpt_every == 0 or ep + 1 == a.episodes:
            ck = {"online": agent.online.state_dict(), "target": agent.target.state_dict(), "opt": agent.opt.state_dict(),
                  "replay": agent.replay, "updates": agent.updates, "rng": agent.rng, "fsmc": env._fsmc,
                  "env_rng": env._rng, "env_steps": env._step_count, "episode": ep + 1}
            tmp = ckpt_path.with_suffix(".tmp"); tmp.write_bytes(pickle.dumps(ck)); tmp.replace(ckpt_path)
            print(f"[{tag}] ep {ep+1}/{a.episodes} eps={agent.epsilon(ep):.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
