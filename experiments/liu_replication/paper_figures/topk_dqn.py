"""DQN with node-score validator selection, for paper-scale N=100, K=21 (C(100,21) sets: no enumeration).

    Q(S, A) = u(h(S)) . phi(feat(delta, S_B, T_I)) + b(feat)  +  w(h(S)) . a        a in {0,1}^N, sum(a) = K

argmax over A = argmax over cfg  +  sum of the K largest node scores w(h(S)).
Exploration draws a uniform random K-subset by default; `val_explore="stake_window"` instead draws K nodes
adjacent in stake order (an exploration prior -- NOT a change of reward, but it injects knowledge that
stake-homogeneous sets satisfy C1, so use it only when stated).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from factored_dqn import DQNHyper


class TopKQNet(nn.Module):
    def __init__(self, state_dim: int, feats: torch.Tensor, n_nodes: int, hidden=(256, 128), d: int = 64) -> None:
        super().__init__()
        self.register_buffer("feats", feats)
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(feats.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, d + 1))
        self.u = nn.Linear(hidden[1], d)
        self.w = nn.Linear(hidden[1], n_nodes)

    def forward(self, s):
        h = self.trunk(s)
        ph = self.phi(self.feats)
        return self.u(h) @ ph[:, :-1].T + ph[:, -1], self.w(h)          # (B, n_cfg), (B, N)


class TopKQNetNF(TopKQNet):
    """Node scores computed from node ATTRIBUTES (stake, GHz, x, y) with a network shared across nodes:
    score_i = g([P h(S), attr_i]). Lets the agent generalise 'nodes with similar stake are good' to nodes it
    has never selected (the plain per-node head cannot)."""

    def __init__(self, state_dim, feats, node_attrs: torch.Tensor, hidden=(256, 128), d: int = 64, p: int = 32) -> None:
        super().__init__(state_dim, feats, node_attrs.shape[0], hidden, d)
        del self.w
        self.register_buffer("attrs", node_attrs)               # (N, a), normalised
        self.proj = nn.Linear(hidden[1], p)
        self.node_mlp = nn.Sequential(nn.Linear(p + node_attrs.shape[1], 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, s):
        h = self.trunk(s)
        ph = self.phi(self.feats)
        qc = self.u(h) @ ph[:, :-1].T + ph[:, -1]
        z = self.proj(h)                                         # (B, p)
        B, N = z.shape[0], self.attrs.shape[0]
        x = torch.cat([z[:, None, :].expand(B, N, -1), self.attrs[None].expand(B, N, -1)], dim=-1)
        return qc, self.node_mlp(x).squeeze(-1)                  # (B, n_cfg), (B, N)


class MaskReplay:
    def __init__(self, cap: int, state_dim: int, n_nodes: int) -> None:
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, state_dim), np.float16); self.s2 = np.zeros((cap, state_dim), np.float16)
        self.cfg = np.zeros(cap, np.int64); self.a = np.zeros((cap, n_nodes), np.float32); self.r = np.zeros(cap, np.float32)

    def push(self, s, cfg, mask, r, s2) -> None:
        self.s[self.i], self.s2[self.i], self.cfg[self.i], self.a[self.i], self.r[self.i] = s.numpy(), s2.numpy(), cfg, mask, r
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b: int, rng: random.Random):
        ix = [rng.randrange(self.n) for _ in range(b)]
        t = torch.from_numpy
        return t(self.s[ix].astype(np.float32)), t(self.cfg[ix]), t(self.a[ix]), t(self.r[ix]), t(self.s2[ix].astype(np.float32))


class TopKDQN:
    def __init__(self, state_dim, n_cfg, n_nodes, k, cfg_mask, hp: DQNHyper, seed, feats, stake_order=None, val_explore="uniform", node_attrs=None):
        self.hp, self.n_nodes, self.k, self.val_explore, self.stake_order = hp, n_nodes, k, val_explore, stake_order
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        make = (lambda: TopKQNetNF(state_dim, feats, node_attrs, hp.hidden)) if node_attrs is not None else (lambda: TopKQNet(state_dim, feats, n_nodes, hp.hidden))
        self.online, self.target = make(), make()
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=hp.lr)
        self.replay = MaskReplay(hp.replay_capacity, state_dim, n_nodes)
        self.allowed = np.flatnonzero(cfg_mask)
        self.neg = torch.full((n_cfg,), float("-inf")); self.neg[torch.from_numpy(self.allowed)] = 0.0
        self.updates = 0

    def epsilon(self, ep: int) -> float:
        return self.hp.eps_start + min(1.0, ep / self.hp.eps_decay_episodes) * (self.hp.eps_end - self.hp.eps_start)

    def _explore_set(self) -> list[int]:
        if self.val_explore == "stake_window":
            i = self.rng.randrange(self.n_nodes - self.k + 1)
            return [int(x) for x in self.stake_order[i:i + self.k]]
        return self.rng.sample(range(self.n_nodes), self.k)

    def act(self, s, ep):
        if self.rng.random() < self.epsilon(ep):
            return int(self.allowed[self.rng.randrange(len(self.allowed))]), self._explore_set(), "explore"
        return (*self.greedy(s), "greedy")

    @torch.no_grad()
    def greedy(self, s):
        qc, w = self.online(s.unsqueeze(0))
        return int((qc[0] + self.neg).argmax()), [int(i) for i in torch.topk(w[0], self.k).indices]

    def observe(self, s, cfg, validators, r, s2):
        mask = np.zeros(self.n_nodes, np.float32); mask[validators] = 1.0
        self.replay.push(s, cfg, mask, r / self.hp.reward_scale, s2)
        if self.replay.n < self.hp.warmup:
            return None
        sb, cb, ab, rb, s2b = self.replay.sample(self.hp.batch_size, self.rng)
        with torch.no_grad():
            tc, tw = self.target(s2b)
            y = rb + self.hp.gamma * ((tc + self.neg).max(1).values + torch.topk(tw, self.k, dim=1).values.sum(1))
        qc, w = self.online(sb)
        q = qc.gather(1, cb[:, None]).squeeze(1) + (w * ab).sum(1)
        loss = torch.mean((y - q) ** 2)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.hp.grad_clip)
        self.opt.step()
        self.updates += 1
        if self.updates % self.hp.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())
