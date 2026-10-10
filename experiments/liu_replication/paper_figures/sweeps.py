"""Liu 2019 Fig. 7-10 -- throughput / finality sweeps, every point evaluated on the DES.

Fig. 7  throughput vs finality threshold omega          (omega = 2..20)
Fig. 8  throughput vs average transaction size chi      (100..550 B)
Fig. 9  average TTF vs average validator compute        (10..30 GHz)
Fig. 10 throughput vs block-size limit S_dot            (1..10 MB)

Schemes: proposed | no_cas | fixed_block_size (4 MB) | fixed_block_interval (1 s) | static.

Per (figure, scheme) one *job* walks the sweep values in order. At each value the agent is trained
for `--episodes` DES epochs (warm-started from the previous value's weights; the first value gets
`--first-mult` x episodes), then the greedy policy is evaluated for `--eval-epochs` fresh DES epochs
(same seeds for every scheme). Results are appended to <out>/<fig>_<scheme>.jsonl and the job resumes
where it stopped.

`static` ("existing static scheme": decisions maximise the immediate reward) is a gamma = 0 DQN with
the full action space -- OUR_RECONSTRUCTION, the paper does not define its optimiser.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common import *  # noqa: F401,F403
from factored_dqn import DQNHyper, FactoredDQN, action_features
from state_encoder import StateEncoder

S_HEAD_MAX_MB = 10.0                       # head covers the largest S_dot of Fig. 10
SB_GRID = block_size_grid(S_HEAD_MAX_MB)   # 50 values
N_CFG = len(PROTOCOLS) * len(SB_GRID) * len(T_I_GRID)
SCHEMES = ("proposed", "no_cas", "fixed_block_size", "fixed_block_interval", "static")

SWEEPS = {
    "fig7": {"param": "omega", "values": [float(v) for v in range(2, 21, 2)], "metric": "throughput"},
    "fig8": {"param": "chi_bytes", "values": [float(v) for v in range(100, 551, 50)], "metric": "throughput"},
    "fig9": {"param": "avg_ghz", "values": [float(v) for v in range(10, 31, 2)], "metric": "ttf"},
    "fig10": {"param": "s_max_mb", "values": [float(v) for v in range(1, 11)], "metric": "throughput"},
}
DEFAULTS = {"omega": 6.0, "chi_bytes": 200.0, "avg_ghz": 20.0, "s_max_mb": 8.0}  # Table I / population mean
BASE_AVG_GHZ = sum(base.CAPABILITIES) / len(base.CAPABILITIES)


def setting(fig: str, value: float) -> dict:
    p = dict(DEFAULTS)
    p[SWEEPS[fig]["param"]] = value
    return {**p, "capability_scale": p["avg_ghz"] / BASE_AVG_GHZ,
            "offered_tx": max(100_000, math.ceil(2.5 * p["s_max_mb"] * 1e6 / p["chi_bytes"]))}


def cfg_decode(c: int) -> tuple[int, float, float]:
    p, rem = divmod(c, len(SB_GRID) * len(T_I_GRID))
    sb_i, ti_i = divmod(rem, len(T_I_GRID))
    return p, SB_GRID[sb_i], T_I_GRID[ti_i]


def scheme_mask(scheme: str, s_max_mb: float) -> np.ndarray:
    m = np.zeros(N_CFG, dtype=bool)
    for c in range(N_CFG):
        p, sb, ti = cfg_decode(c)
        ok = sb <= s_max_mb + 1e-9
        if scheme == "no_cas":
            ok &= p == 0
        elif scheme == "fixed_block_size":
            ok &= abs(sb - min(4.0, s_max_mb)) < 1e-9
        elif scheme == "fixed_block_interval":
            ok &= abs(ti - 1.0) < 1e-9
        m[c] = ok
    return m


def run_episode(env, agent, enc, ep: int, train: bool):
    s = enc.encode(base.make_state_dict(env._current_link_rows()))
    cfg, val, _ = agent.act(s, ep) if train else (*agent.greedy(s), "greedy")
    p, sb, ti = cfg_decode(cfg)
    res = env.step(make_action(val, p, sb, ti))
    if train:
        s2 = enc.encode(base.make_state_dict(env._current_link_rows()))
        agent.observe(s, cfg, val, res.reward, s2)
    return res, ti


def run_job(fig: str, scheme: str, a) -> None:
    sw = SWEEPS[fig]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{fig}_{scheme}"
    res_path, w_path = out / f"{tag}.jsonl", out / f"{tag}.weights"
    done = {json.loads(l)["value"] for l in res_path.read_text().splitlines()} if res_path.exists() else set()
    enc = StateEncoder(N)
    feats = action_features(cfg_decode, N_CFG)
    vmask = torch.tensor([[1.0 if i in vs else 0.0 for i in range(N)] for vs in VALIDATOR_SETS])
    prev = torch.load(w_path, weights_only=True) if w_path.exists() else None

    for i, v in enumerate(sw["values"]):
        if v in done:
            continue
        st = setting(fig, v)
        ep_budget = a.episodes * (a.first_mult if prev is None else 1)
        hp = DQNHyper(gamma=0.0 if scheme == "static" else 0.9,
                      eps_start=1.0 if prev is None else 0.3, eps_end=0.02,
                      eps_decay_episodes=max(1, ep_budget // 2), warmup=min(200, max(8, ep_budget // 4)))
        agent = FactoredDQN(enc.dim, N_CFG, len(VALIDATOR_SETS), scheme_mask(scheme, st["s_max_mb"]), hp, a.seed + i, feats, vmask)
        if prev is not None:
            agent.online.load_state_dict(prev)
            agent.target.load_state_dict(prev)
        env = make_env(a.seed + i, chi_bytes=st["chi_bytes"], omega=st["omega"],
                       capability_scale=st["capability_scale"], offered_tx=st["offered_tx"])
        t0, tail = time.time(), []
        for ep in range(ep_budget):
            res, _ = run_episode(env, agent, enc, ep, train=True)
            tail.append(res.reward)
        prev = {k: x.clone() for k, x in agent.online.state_dict().items()}

        ev = make_env(10_000 + i, chi_bytes=st["chi_bytes"], omega=st["omega"],
                      capability_scale=st["capability_scale"], offered_tx=st["offered_tx"])  # same seed for all schemes
        rewards, ttfs = [], []
        for _ in range(a.eval_epochs):
            res, ti = run_episode(ev, agent, enc, 0, train=False)
            rewards.append(res.reward)
            if res.reward > 0 and math.isfinite(res.des_t_c_s):
                ttfs.append(ti + res.des_t_c_s)      # T_F = T_I + T_C (Eq. 6)
        row = {"fig": fig, "scheme": scheme, "param": sw["param"], "value": v,
               "throughput": float(np.mean(rewards)), "ttf": float(np.mean(ttfs)) if ttfs else None,
               "feasible_frac": float(np.mean([r > 0 for r in rewards])), "train_tail_mean": float(np.mean(tail[-100:])),
               "episodes": ep_budget, "eval_epochs": a.eval_epochs, "wall_s": round(time.time() - t0, 1)}
        with open(res_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save(prev, w_path)
        print(json.dumps(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", required=True, choices=list(SWEEPS))
    ap.add_argument("--scheme", required=True, choices=SCHEMES)
    ap.add_argument("--episodes", type=int, default=1000, help="training epochs per sweep value")
    ap.add_argument("--first-mult", type=int, default=3, help="first value trains this many times longer")
    ap.add_argument("--eval-epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/sweeps")
    a = ap.parse_args()
    torch.set_num_threads(1)
    run_job(a.fig, a.scheme, a)


if __name__ == "__main__":
    main()
