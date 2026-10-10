"""DQN over Liu's joint action A = [a, delta, S_B, T_I] without enumerating it.

The full action space is 435 validator sets x 3 protocols x 40 block sizes x 20 intervals
(~10^6 actions), so Q(S, A) is parameterised as

    Q(S, A) = q_cfg(S)[delta, S_B, T_I] + q_val(S)[a]

i.e. a shared state trunk with a joint (delta, S_B, T_I) head (2400 outputs: S_B and T_I
interact strongly through Omega and C2, so they are NOT factored further) and an additive
validator-set head (435 outputs). argmax over A is exact and cheap:
argmax_cfg + argmax_val. Scheme variants (fixed block size, ...) only mask the cfg head.

Classification: OUR_RECONSTRUCTION (Liu specifies Q(S,A) with a DNN but no architecture).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DQNHyper:
    hidden: tuple[int, int] = (256, 128)
    gamma: float = 0.9            # Liu's mu in (0,1]; value not given
    lr: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 20_000
    warmup: int = 200
    target_sync: int = 100        # "every G steps" in Algorithm 1; G not given
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_episodes: int = 5_000
    reward_scale: float = 17_000.0  # reward normalisation for stable Bellman targets only
    grad_clip: float = 5.0


class FactoredQNet(nn.Module):
    def __init__(self, state_dim: int, n_cfg: int, n_val: int, hidden=(256, 128)) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.cfg_head = nn.Linear(hidden[1], n_cfg)
        self.val_head = nn.Linear(hidden[1], n_val)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        return self.cfg_head(h), self.val_head(h)


def action_features(cfg_decode, n_cfg: int, protocols: int = 3) -> torch.Tensor:
    """Smooth features of a (delta, S_B, T_I) configuration so Q can generalise across neighbours.

    s = S_B/10 MB, t = T_I/10 s, log(S_B/T_I) ~ log Omega; per-protocol copies of the continuous terms.
    """
    rows = []
    for c in range(n_cfg):
        p, sb, ti = cfg_decode(c)
        s, t = sb / 10.0, ti / 10.0
        lr = math.log(sb / ti) / 4.0
        oh = [1.0 if p == k else 0.0 for k in range(protocols)]
        cont = [s, t, lr, s * s, t * t, s * t]
        rows.append(oh + cont + [o * x for o in oh for x in (s, t, lr)])
    return torch.tensor(rows, dtype=torch.float32)


class FeatureQNet(nn.Module):
    """Q(S, A) = u(h(S)) . phi(feat(cfg)) + b(feat(cfg))  +  w(h(S)) . member(validator set).

    Same interface as FactoredQNet (returns Q over all cfgs and all validator sets) but the action
    enters through features, as in Liu's Q(S, A): nearby (S_B, T_I) share statistics and a validator
    set is scored by the nodes it contains (435 sets = 30 node weights).
    """

    def __init__(self, state_dim: int, feats: torch.Tensor, vmask: torch.Tensor, hidden=(256, 128), d: int = 64) -> None:
        super().__init__()
        self.register_buffer("feats", feats)    # (n_cfg, f)
        self.register_buffer("vmask", vmask)    # (n_val, N) 0/1 membership
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(feats.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d + 1))
        self.u = nn.Linear(hidden[1], d)
        self.w = nn.Linear(hidden[1], vmask.shape[1])

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        ph = self.phi(self.feats)                              # (n_cfg, d+1)
        qc = self.u(h) @ ph[:, :-1].T + ph[:, -1]              # (B, n_cfg)
        qv = self.w(h) @ self.vmask.T                          # (B, n_val)
        return qc, qv


class Replay:
    def __init__(self, capacity: int, state_dim: int) -> None:
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float16)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float16)
        self.cfg = np.zeros(capacity, dtype=np.int64)
        self.val = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.n = 0
        self.i = 0

    def push(self, s, cfg, val, r, s2) -> None:
        self.s[self.i], self.s2[self.i] = s.numpy(), s2.numpy()
        self.cfg[self.i], self.val[self.i], self.r[self.i] = cfg, val, r
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b: int, rng: random.Random):
        idx = [rng.randrange(self.n) for _ in range(b)]
        t = torch.from_numpy
        return (t(self.s[idx].astype(np.float32)), t(self.cfg[idx]), t(self.val[idx]), t(self.r[idx]), t(self.s2[idx].astype(np.float32)))


class FactoredDQN:
    def __init__(self, state_dim: int, n_cfg: int, n_val: int, cfg_mask: np.ndarray, hp: DQNHyper, seed: int,
                 feats: torch.Tensor | None = None, vmask: torch.Tensor | None = None) -> None:
        self.hp, self.n_cfg, self.n_val = hp, n_cfg, n_val
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        make = (lambda: FeatureQNet(state_dim, feats, vmask, hp.hidden)) if feats is not None else (lambda: FactoredQNet(state_dim, n_cfg, n_val, hp.hidden))
        self.online, self.target = make(), make()
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=hp.lr)
        self.replay = Replay(hp.replay_capacity, state_dim)
        self.allowed = np.flatnonzero(cfg_mask)           # indices of permitted (delta,S_B,T_I)
        self.neg_mask = torch.full((n_cfg,), float("-inf"))
        self.neg_mask[torch.from_numpy(self.allowed)] = 0.0
        self.updates = 0

    def epsilon(self, episode: int) -> float:
        f = min(1.0, episode / self.hp.eps_decay_episodes)
        return self.hp.eps_start + f * (self.hp.eps_end - self.hp.eps_start)

    def act(self, s: torch.Tensor, episode: int) -> tuple[int, int, str]:
        if self.rng.random() < self.epsilon(episode):
            return int(self.allowed[self.rng.randrange(len(self.allowed))]), self.rng.randrange(self.n_val), "explore"
        return (*self.greedy(s), "greedy")

    @torch.no_grad()
    def greedy(self, s: torch.Tensor) -> tuple[int, int]:
        qc, qv = self.online(s.unsqueeze(0))
        return int((qc[0] + self.neg_mask).argmax()), int(qv[0].argmax())

    def observe(self, s, cfg, val, r, s2) -> float | None:
        self.replay.push(s, cfg, val, r / self.hp.reward_scale, s2)
        if self.replay.n < self.hp.warmup:
            return None
        sb, cb, vb, rb, s2b = self.replay.sample(self.hp.batch_size, self.rng)
        with torch.no_grad():
            tc, tv = self.target(s2b)
            y = rb + self.hp.gamma * ((tc + self.neg_mask).max(1).values + tv.max(1).values)
        qc, qv = self.online(sb)
        q = qc.gather(1, cb[:, None]).squeeze(1) + qv.gather(1, vb[:, None]).squeeze(1)
        loss = torch.mean((y - q) ** 2)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.hp.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % self.hp.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
