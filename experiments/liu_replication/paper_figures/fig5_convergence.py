"""Liu 2019 Fig. 5 -- convergence of four DRL schemes, evaluated on the SymBChainSim DES.

One *episode* = one decision epoch (the paper does not define episode length); the FSMC link
state keeps evolving across episodes (no reset). Reward per episode is the DES-realised
throughput Omega if C1&C2&C3 hold, else 0.

Schemes:  proposed | no_cas (PBFT only) | fixed_block_size (4 MB) | fixed_block_interval (1 s)

    python fig5_convergence.py --scheme proposed --episodes 20000 --out runs/fig5
Resumes automatically from <out>/<scheme>_seed<seed>.ckpt when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import *  # noqa: F401,F403
from factored_dqn import DQNHyper, FactoredDQN, action_features
from state_encoder import StateEncoder

SB_GRID = block_size_grid(8.0)
N_CFG = len(PROTOCOLS) * len(SB_GRID) * len(T_I_GRID)


def cfg_index(p: int, sb_i: int, ti_i: int) -> int:
    return (p * len(SB_GRID) + sb_i) * len(T_I_GRID) + ti_i


def cfg_decode(c: int) -> tuple[int, float, float]:
    p, rem = divmod(c, len(SB_GRID) * len(T_I_GRID))
    sb_i, ti_i = divmod(rem, len(T_I_GRID))
    return p, SB_GRID[sb_i], T_I_GRID[ti_i]


def scheme_mask(scheme: str) -> np.ndarray:
    m = np.ones(N_CFG, dtype=bool)
    for c in range(N_CFG):
        p, sb, ti = cfg_decode(c)
        if scheme == "no_cas" and p != 0:
            m[c] = False
        if scheme == "fixed_block_size" and abs(sb - 4.0) > 1e-9:
            m[c] = False
        if scheme == "fixed_block_interval" and abs(ti - 1.0) > 1e-9:
            m[c] = False
    return m


def state_dict_of(link_rows) -> dict:
    return base.make_state_dict(link_rows)


def _truncate_log(path: Path, start: int) -> None:
    """Drop CSV rows for episodes >= start (the log can run ahead of the last checkpoint)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        fields, keep = rd.fieldnames, [r for r in rd if int(r["episode"]) < start]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=["proposed", "no_cas", "fixed_block_size", "fixed_block_interval"])
    ap.add_argument("--episodes", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/fig5")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--arch", choices=["features", "tabular"], default="features")
    a = ap.parse_args()
    torch.set_num_threads(1)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.scheme}_seed{a.seed}"
    log_path, ckpt_path = out / f"{tag}.csv", out / f"{tag}.ckpt"

    enc = StateEncoder(N)
    hp = DQNHyper()
    feats = vmask = None
    if a.arch == "features":
        feats = action_features(cfg_decode, N_CFG)
        vmask = torch.tensor([[1.0 if i in vs else 0.0 for i in range(N)] for vs in VALIDATOR_SETS])
    agent = FactoredDQN(enc.dim, N_CFG, len(VALIDATOR_SETS), scheme_mask(a.scheme), hp, a.seed, feats, vmask)
    env = make_env(a.seed)
    start = 0
    if ckpt_path.exists():
        ck = pickle.loads(ckpt_path.read_bytes())
        agent.online.load_state_dict(ck["online"]); agent.target.load_state_dict(ck["target"]); agent.opt.load_state_dict(ck["opt"])
        agent.replay, agent.updates, agent.rng = ck["replay"], ck["updates"], ck["rng"]
        env._fsmc, env._rng, env._step_count = ck["fsmc"], ck["env_rng"], ck["env_steps"]
        start = ck["episode"]
        _truncate_log(log_path, start)   # rows past the checkpoint will be re-run
        print(f"resumed {tag} at episode {start}", flush=True)

    wpath = out / f"{tag}.weights.pt"
    if start == 0 and wpath.exists():
        # Fallback resume (container recycled): weights only. Fresh Adam/replay/FSMC; CSV rows past the
        # snapshot are dropped so the log stays consistent with the resumed policy.
        snap = torch.load(wpath, weights_only=True)
        sd = {k: v.float() for k, v in snap["online"].items()}
        agent.online.load_state_dict(sd); agent.target.load_state_dict(sd)
        start = snap["episode"]
        env = make_env(a.seed + start)
        _truncate_log(log_path, start)
        print(f"resumed {tag} from weights snapshot at episode {start}", flush=True)

    cols = ["episode", "reward", "epsilon", "mode", "protocol", "block_size_mb", "block_interval_s", "validator_set",
            "c1", "c2", "c3", "failure", "loss", "step_s"]
    new = not log_path.exists() or start == 0
    f = open(log_path, "w" if new else "a", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()

    def link_state() -> torch.Tensor:
        return enc.encode(state_dict_of(env._current_link_rows()))

    s = link_state()
    for ep in range(start, a.episodes):
        t0 = time.time()
        cfg, val, mode = agent.act(s, ep)
        p, sb, ti = cfg_decode(cfg)
        res = env.step(make_action(val, p, sb, ti))
        s2 = link_state()
        loss = agent.observe(s, cfg, val, res.reward, s2)
        w.writerow({"episode": ep, "reward": round(res.reward, 3), "epsilon": round(agent.epsilon(ep), 4), "mode": mode,
                    "protocol": PROTOCOL_NAMES[p], "block_size_mb": sb, "block_interval_s": ti, "validator_set": val,
                    "c1": int(res.des_c1), "c2": int(res.des_c2), "c3": int(res.des_c3), "failure": res.failure_reason.value,
                    "loss": "" if loss is None else round(loss, 6), "step_s": round(time.time() - t0, 3)})
        s = s2
        if (ep + 1) % 50 == 0:
            f.flush()
        if (ep + 1) % a.ckpt_every == 0 or ep + 1 == a.episodes:
            ck = {"online": agent.online.state_dict(), "target": agent.target.state_dict(), "opt": agent.opt.state_dict(),
                  "replay": agent.replay, "updates": agent.updates, "rng": agent.rng, "fsmc": env._fsmc,
                  "env_rng": env._rng, "env_steps": env._step_count, "episode": ep + 1}
            tmp = ckpt_path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(ck))
            tmp.replace(ckpt_path)
            print(f"[{tag}] ep {ep+1}/{a.episodes} eps={agent.epsilon(ep):.3f}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
