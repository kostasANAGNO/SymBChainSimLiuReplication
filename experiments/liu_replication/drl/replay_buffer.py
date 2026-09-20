"""Replay buffer for DQN transitions.

Stores (state_enc, action_enc, reward, next_state_enc, done) tuples.
Uniform random sampling.

Classification: OUR_RECONSTRUCTION — paper does not specify replay mechanics,
though replay + target network are standard DQN practice (DQN Mnih 2015).
"""
from __future__ import annotations

import random
from collections import deque

import torch


class ReplayBuffer:
    """Fixed-capacity deque replay buffer with uniform sampling."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self._buf: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state_enc: torch.Tensor,
        action_enc: torch.Tensor,
        reward: float,
        next_state_enc: torch.Tensor,
        done: bool,
    ) -> None:
        """Store one transition. Older entries are evicted when at capacity."""
        self._buf.append((
            state_enc.detach().clone(),
            action_enc.detach().clone(),
            float(reward),
            next_state_enc.detach().clone(),
            bool(done),
        ))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """Sample a batch uniformly at random.

        Returns:
            states      (B, state_dim)
            actions     (B, action_dim)
            rewards     (B,)
            next_states (B, state_dim)
            dones       (B,)  bool
        """
        batch = self._rng.sample(self._buf, min(batch_size, len(self._buf)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool),
        )

    def __len__(self) -> int:
        return len(self._buf)
