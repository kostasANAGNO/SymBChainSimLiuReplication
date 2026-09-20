"""DQN agent: epsilon-greedy policy, exact argmax, Bellman target update.

Implements Q-learning with:
  - Exact argmax over all legal actions (not 64 sampled candidates)
  - Bellman target: y = R + gamma * max_A' Q_target(S', A') [non-terminal]
                    y = R                                    [terminal]
  - Target network updated every target_sync_steps optimizer steps
  - Epsilon-greedy exploration with linear schedule
  - Deterministic tie-break: smallest action_id when Q values are equal

Classification:
  Q-learning update rule: PAPER_EXACT (Eq.12 / Algorithm 1)
  Exact argmax over reduced domain: OUR_RECONSTRUCTION (practical realization of A* = argmax Q)
  Target network: OUR_RECONSTRUCTION (not specified in paper; standard DQN stability technique)
  Epsilon schedule: OUR_RECONSTRUCTION (paper names epsilon-greedy, no schedule given)
  gamma=0.9: OUR_RECONSTRUCTION (paper gives symbol mu in (0,1], no value)
"""
from __future__ import annotations

import copy
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from config import DQNConfig, DEFAULT_CONFIG
from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """Online DQN agent with exact argmax over the legal action domain."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        all_action_tensors: torch.Tensor,
        cfg: DQNConfig = DEFAULT_CONFIG,
        seed: int = 42,
    ) -> None:
        """
        Args:
            state_dim:           Dimension of the state encoding.
            action_dim:          Dimension of the action encoding.
            all_action_tensors:  (A, action_dim) — all legal action encodings, pre-computed.
            cfg:                 Hyperparameter configuration.
            seed:                RNG seed for exploration.
        """
        self.cfg = cfg
        self._rng = random.Random(seed)
        torch.manual_seed(seed)

        self.online = QNetwork(state_dim, action_dim, cfg)
        self.target = copy.deepcopy(self.online)
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.replay = ReplayBuffer(cfg.replay_capacity, seed=seed)

        # Pre-computed action encodings — shape (A, action_dim)
        self.all_action_tensors = all_action_tensors
        self.n_actions = all_action_tensors.shape[0]

        self._step = 0            # environment steps
        self._opt_step = 0        # optimizer steps
        self._last_loss: float | None = None  # loss from most recent train step

    # ── Epsilon schedule ───────────────────────────────────────────────────────

    def epsilon(self, step: int | None = None) -> float:
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay_steps."""
        t = self._step if step is None else step
        cfg = self.cfg
        ratio = min(1.0, t / max(1, cfg.epsilon_decay_steps))
        return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state_enc: torch.Tensor) -> tuple[int, str]:
        """Epsilon-greedy action selection.

        Returns:
            action_idx: Index into self.all_action_tensors (= action_id).
            mode:       "greedy" or "explore"
        """
        if self._rng.random() < self.epsilon():
            idx = self._rng.randrange(self.n_actions)
            return idx, "explore"
        return self.argmax(state_enc), "greedy"

    @torch.no_grad()
    def argmax(self, state_enc: torch.Tensor) -> int:
        """Exact argmax over all legal actions using the online network.

        Returns:
            action_idx — index of highest predicted Q value.
            Deterministic tie-break: smallest action_id.
        """
        self.online.eval()
        q_vals = self.online.q_values_batch(state_enc, self.all_action_tensors)  # (A,)
        self.online.train()
        best_idx = int(torch.argmax(q_vals).item())
        return best_idx

    @torch.no_grad()
    def max_next_q(self, next_state_enc: torch.Tensor) -> float:
        """max_A' Q_target(S', A') over all legal actions using the target network."""
        self.target.eval()
        q_vals = self.target.q_values_batch(next_state_enc, self.all_action_tensors)  # (A,)
        return float(q_vals.max().item())

    # ── Training ───────────────────────────────────────────────────────────────

    def observe(
        self,
        state_enc: torch.Tensor,
        action_idx: int,
        reward: float,
        next_state_enc: torch.Tensor,
        done: bool,
    ) -> None:
        """Store transition and optionally train."""
        action_enc = self.all_action_tensors[action_idx]
        self.replay.push(state_enc, action_enc, reward, next_state_enc, done)
        self._step += 1

        if len(self.replay) < self.cfg.replay_warmup_steps:
            return
        if self._step % self.cfg.update_every_steps == 0:
            self._last_loss = self._train_step()

    def _train_step(self) -> float | None:
        """One Bellman update step. Returns loss or None if replay too small."""
        if len(self.replay) < self.cfg.replay_warmup_steps:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.cfg.batch_size)

        # Bellman targets: y = R + gamma * max_A' Q_target(S', A') * (1 - done)
        with torch.no_grad():
            next_q_vals = []
            for ns in next_states:
                q_ns = self.target.q_values_batch(ns, self.all_action_tensors)
                next_q_vals.append(q_ns.max())
            next_q = torch.stack(next_q_vals)
            targets = rewards + self.cfg.gamma * next_q * (~dones).float()

        # Online Q estimates
        self.online.train()
        q_pred = self.online(states, actions).squeeze(1)

        loss = nn.functional.mse_loss(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.gradient_clip_norm)
        self.optimizer.step()
        self._opt_step += 1

        if self._opt_step % self.cfg.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def train_step_explicit(self) -> float | None:
        """Force one training step (used for supervised sanity check)."""
        return self._train_step()
