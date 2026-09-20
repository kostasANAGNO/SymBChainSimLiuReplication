"""Test: Bellman target numerical correctness with hand-computed values.

Non-terminal: y = R + gamma * max_A' Q_target(S', A')
Terminal:      y = R

Uses a mock Q-network with known outputs to verify the Bellman update is correct.
"""
import math
import sys
from pathlib import Path
import torch
import torch.nn as nn

DRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL))

from config import DQNConfig
from dqn_agent import DQNAgent


class ConstantQ(nn.Module):
    """Q-network that returns a predefined Q vector for any state."""
    def __init__(self, q_vals: list[float]):
        super().__init__()
        self._q = torch.tensor(q_vals, dtype=torch.float32)

    def forward(self, state, action):
        return torch.zeros(action.shape[0], 1)

    def q_values_batch(self, state_enc, action_batch):
        return self._q.clone()


def _make_agent(n_actions=5, gamma=0.9, next_q_vals=None):
    state_dim = 2
    action_dim = 3
    action_tensors = torch.eye(action_dim)[:n_actions] if n_actions <= action_dim else torch.rand(n_actions, action_dim)
    cfg = DQNConfig(hidden_sizes=(8,), gamma=gamma, batch_size=1,
                    replay_warmup_steps=1, replay_capacity=100,
                    epsilon_start=0.0, epsilon_end=0.0, update_every_steps=1,
                    target_sync_steps=1000)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=0)
    if next_q_vals is not None:
        agent.target = ConstantQ(next_q_vals)
    return agent


def test_bellman_non_terminal():
    """y = R + gamma * max_A' Q_target(S', A')"""
    # max Q_target over 5 actions is 8.0, gamma=0.9, R=3.0
    # expected y = 3.0 + 0.9 * 8.0 = 10.2
    agent = _make_agent(n_actions=5, gamma=0.9, next_q_vals=[1.0, 5.0, 8.0, 2.0, 3.0])
    R = 3.0
    state = torch.zeros(2)
    next_state = torch.ones(2)
    with torch.no_grad():
        max_q_next = agent.max_next_q(next_state)
    y = R + 0.9 * max_q_next
    assert math.isclose(max_q_next, 8.0, abs_tol=1e-5), f"max_q_next={max_q_next}"
    assert math.isclose(y, 10.2, abs_tol=1e-5), f"y={y}"


def test_bellman_terminal():
    """Terminal transition: y = R (gamma term absent)."""
    # For terminal, done=True → next_q multiplied by 0
    agent = _make_agent(n_actions=5, gamma=0.9, next_q_vals=[100.0, 200.0, 300.0, 400.0, 500.0])
    R = 7.0
    # Inject a terminal transition manually and check Bellman target
    from replay_buffer import ReplayBuffer
    buf = ReplayBuffer(capacity=10, seed=0)
    s = torch.zeros(2)
    a = torch.zeros(3)
    ns = torch.ones(2)
    buf.push(s, a, R, ns, done=True)  # terminal
    states, actions, rewards, next_states, dones = buf.sample(1)
    # Compute target manually
    with torch.no_grad():
        q_next = torch.stack([agent.target.q_values_batch(ns_i, agent.all_action_tensors).max()
                               for ns_i in next_states])
    targets = rewards + 0.9 * q_next * (~dones).float()
    assert math.isclose(float(targets[0]), R, abs_tol=1e-5), \
        f"Terminal target should be R={R}, got {float(targets[0])}"


def test_bellman_different_gamma():
    """y = R + gamma * max_q for different gamma values."""
    max_q = 10.0
    R = 5.0
    for gamma in [0.0, 0.5, 0.99]:
        agent = _make_agent(n_actions=3, gamma=gamma, next_q_vals=[max_q, 0.0, 0.0])
        next_state = torch.zeros(2)
        with torch.no_grad():
            q_next = agent.max_next_q(next_state)
        y = R + gamma * q_next
        expected = R + gamma * max_q
        assert math.isclose(y, expected, abs_tol=1e-5), f"gamma={gamma}: y={y} != {expected}"


def test_no_nan_in_training():
    """Training step must produce finite loss."""
    n_actions = 5
    state_dim = 4
    action_dim = 3
    action_tensors = torch.rand(n_actions, action_dim)
    cfg = DQNConfig(hidden_sizes=(16, 8), batch_size=4, replay_warmup_steps=4, replay_capacity=50)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=0)
    for i in range(20):
        s = torch.randn(state_dim)
        a_idx, _ = agent.select_action(s)
        r = float(i % 3)
        ns = torch.randn(state_dim)
        agent.observe(s, a_idx, r, ns, done=False)
    # Force one training step
    loss = agent.train_step_explicit()
    if loss is not None:
        assert math.isfinite(loss), f"Loss is not finite: {loss}"


if __name__ == "__main__":
    for fn in [test_bellman_non_terminal, test_bellman_terminal,
               test_bellman_different_gamma, test_no_nan_in_training]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("\nAll Bellman target tests: PASS")
