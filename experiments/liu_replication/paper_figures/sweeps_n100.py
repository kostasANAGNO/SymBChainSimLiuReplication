"""Liu 2019 Fig. 7-10 at paper scale (N=100, K=21), every point evaluated on the DES.

Same sweeps, schemes, budgets and protocol as sweeps.py (warm start across sweep values, greedy evaluation
over fresh DES epochs with identical seeds for all schemes), but with the Table I population and the
node-attribute validator head (topk_dqn.TopKQNetNF). Validator exploration defaults to `stake_window`
(an exploration prior, see topk_dqn.py / the N=100 pilot: uniform exploration never sees a C1-valid set).

    python sweeps_n100.py --fig fig7 --scheme proposed --episodes 300 --out runs/sweeps_n100
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sweeps as sw                      # sweep definitions, cfg grid (S up to 10 MB), scheme masks
from common_n100 import *               # noqa: F401,F403
from factored_dqn import DQNHyper, action_features
from state_encoder import StateEncoder
from topk_dqn import TopKDQN

BASE_AVG_GHZ100 = sum(CAPS100) / len(CAPS100)


def setting(fig: str, value: float) -> dict:
    p = dict(sw.DEFAULTS)
    p[sw.SWEEPS[fig]["param"]] = value
    return {**p, "capability_scale": p["avg_ghz"] / BASE_AVG_GHZ100,
            "offered_tx": max(100_000, math.ceil(2.5 * p["s_max_mb"] * 1e6 / p["chi_bytes"]))}


def node_attrs(scale: float) -> torch.Tensor:
    return torch.tensor([[STAKES100[i] / 50.0, CAPS100[i] * scale / 30.0, POS100[i][0], POS100[i][1]] for i in range(N100)], dtype=torch.float32)


def run_episode(env, agent, enc, chi, scale, ep, train):
    obs = lambda: enc.encode({"chi_bytes": float(chi), "stakes": STAKES100, "positions_km": POS100,
                              "capabilities_ghz": tuple(c * scale for c in CAPS100), "link_rows_mbps": env._current_link_rows()})
    s = obs()
    cfg, vals, _ = agent.act(s, ep) if train else (*agent.greedy(s), "greedy")
    p, sb, ti = sw.cfg_decode(cfg)
    res = env.step(make_action100(vals, p, sb, ti))
    if train:
        agent.observe(s, cfg, vals, res.reward, obs())
    return res, ti


def run_job(fig: str, scheme: str, a) -> None:
    spec = sw.SWEEPS[fig]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{fig}_{scheme}"
    res_path, w_path = out / f"{tag}.jsonl", out / f"{tag}.weights"
    done = {json.loads(l)["value"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
    enc = StateEncoder(N100)
    feats = action_features(sw.cfg_decode, sw.N_CFG)
    order = np.argsort(np.array(STAKES100))
    prev = torch.load(w_path, weights_only=True) if w_path.exists() else None
    for i, v in enumerate(spec["values"]):
        if v in done:
            continue
        st = setting(fig, v)
        budget = a.episodes * (a.first_mult if prev is None else 1)
        hp = DQNHyper(gamma=0.0 if scheme == "static" else 0.9, eps_start=1.0 if prev is None else 0.3, eps_end=0.02,
                      eps_decay_episodes=max(1, budget // 2), warmup=min(200, max(8, budget // 4)), replay_capacity=a.replay)
        agent = TopKDQN(enc.dim, sw.N_CFG, N100, K21, sw.scheme_mask(scheme, st["s_max_mb"]), hp, a.seed + i, feats,
                        stake_order=order, val_explore=a.val_explore, node_attrs=node_attrs(st["capability_scale"]))
        if prev is not None:
            sd = {k: x for k, x in prev.items() if k != "attrs"}      # node attributes depend on the sweep value
            agent.online.load_state_dict(sd, strict=False); agent.target.load_state_dict(sd, strict=False)
        kw = dict(chi_bytes=int(st["chi_bytes"]), omega=st["omega"], offered_tx=st["offered_tx"], capability_scale=st["capability_scale"],
                  link_profile=a.link_profile)
        env = make_env100(a.seed + i, **kw)
        t0, tail = time.time(), []
        for ep in range(budget):
            res, _ = run_episode(env, agent, enc, st["chi_bytes"], st["capability_scale"], ep, True)
            tail.append(res.reward)
        prev = {k: x.clone() for k, x in agent.online.state_dict().items()}
        ev = make_env100(10_000 + i, **kw)                          # same eval seed for every scheme
        rewards, ttfs, c1 = [], [], []
        for _ in range(a.eval_epochs):
            res, ti = run_episode(ev, agent, enc, st["chi_bytes"], st["capability_scale"], 0, False)
            rewards.append(res.reward); c1.append(res.des_c1)
            if res.reward > 0 and math.isfinite(res.des_t_c_s):
                ttfs.append(ti + res.des_t_c_s)
        row = {"fig": fig, "scheme": scheme, "param": spec["param"], "value": v, "throughput": float(np.mean(rewards)),
               "ttf": float(np.mean(ttfs)) if ttfs else None, "feasible_frac": float(np.mean([r > 0 for r in rewards])),
               "c1_frac": float(np.mean(c1)), "train_tail_mean": float(np.mean(tail[-100:])), "episodes": budget,
               "eval_epochs": a.eval_epochs, "wall_s": round(time.time() - t0, 1), "scale": "N100_K21", "link_profile": a.link_profile}
        with open(res_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save(prev, w_path)
        print(json.dumps(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True, choices=list(sw.SWEEPS))
    ap.add_argument("--scheme", required=True, choices=sw.SCHEMES)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--first-mult", type=int, default=3)
    ap.add_argument("--eval-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--replay", type=int, default=3000)
    ap.add_argument("--val-explore", choices=["uniform", "stake_window"], default="stake_window")
    ap.add_argument("--link-profile", choices=sorted(LINK_PROFILES), default="default")
    ap.add_argument("--out", default="runs/sweeps_n100")
    a = ap.parse_args()
    torch.set_num_threads(1)
    run_job(a.fig, a.scheme, a)


if __name__ == "__main__":
    main()
