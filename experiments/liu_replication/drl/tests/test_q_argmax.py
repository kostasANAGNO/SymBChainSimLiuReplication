"""Test: Q-network argmax correctness.

Constructs fake known Q outputs and proves that epsilon=0 selects
the true maximum-Q candidate.
"""
import sys
from pathlib import Path
import torch

DRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL))

from config import DEFAULT_CONFIG, DQNConfig
from q_network import QNetwork
from dqn_agent import DQNAgent
from state_encoder import StateEncoder
from action_encoder import ActionEncoder

N = 6


def _make_fake_action_tensors(n_actions: int, action_dim: int) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.rand(n_actions, action_dim)


def test_argmax_selects_max_q():
    """With a known Q function, argmax must return the action with highest Q."""
    n_actions = 10
    state_dim = 1
    action_dim = 5
    action_tensors = _make_fake_action_tensors(n_actions, action_dim)

    cfg = DQNConfig(hidden_sizes=(8, 4), epsilon_start=0.0, epsilon_end=0.0, replay_warmup_steps=1)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=0)

    # Force the network to return a known Q vector by replacing it with a linear model
    # that maps each action encoding to a preset value
    target_q = torch.zeros(n_actions)
    target_q[3] = 100.0   # action 3 has the best Q
    target_q[7] = 50.0

    class FixedQ(torch.nn.Module):
        def __init__(self, q_vals):
            super().__init__()
            self._q = q_vals  # (A,)
        def forward(self, state, action):
            # action is (B, action_dim); identify which action by its index in the batch
            B = action.shape[0]
            result = []
            for i in range(B):
                # find which action tensor this is (by closest match)
                diffs = (action_tensors - action[i]).abs().sum(dim=1)
                idx = int(diffs.argmin().item())
                result.append(self._q[idx].unsqueeze(0))
            return torch.stack(result)
        def q_values_batch(self, state_enc, action_batch):
            return torch.tensor([self._q[i] for i in range(len(action_batch))])

    agent.online = FixedQ(target_q)
    state_enc = torch.zeros(state_dim)
    best = agent.argmax(state_enc)
    assert best == 3, f"argmax should return 3 (Q=100), got {best}"


def test_argmax_tie_break_smallest_id():
    """When Q values are equal, smallest action_id wins."""
    n_actions = 5
    state_dim = 1
    action_dim = 4
    action_tensors = torch.ones(n_actions, action_dim)  # all identical

    cfg = DQNConfig(hidden_sizes=(4,), epsilon_start=0.0, epsilon_end=0.0, replay_warmup_steps=1)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=0)

    class EqualQ(torch.nn.Module):
        def forward(self, state, action): return torch.zeros(action.shape[0], 1)
        def q_values_batch(self, state_enc, action_batch): return torch.zeros(action_batch.shape[0])

    agent.online = EqualQ()
    best = agent.argmax(torch.zeros(state_dim))
    assert best == 0, f"Tie-break should return smallest action_id=0, got {best}"


def test_epsilon_zero_always_greedy():
    """epsilon=0 must always select argmax."""
    n_actions = 20
    state_dim = 4
    action_dim = 3
    action_tensors = torch.rand(n_actions, action_dim)

    cfg = DQNConfig(hidden_sizes=(8,), epsilon_start=0.0, epsilon_end=0.0, replay_warmup_steps=1)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=42)

    state_enc = torch.randn(state_dim)
    expected = agent.argmax(state_enc)
    for _ in range(20):
        idx, mode = agent.select_action(state_enc)
        assert mode == "greedy", "epsilon=0 must always be greedy"
        assert idx == expected


def test_epsilon_one_explores():
    """epsilon=1 must produce exploration (different actions across many draws)."""
    n_actions = 50
    state_dim = 4
    action_dim = 3
    action_tensors = torch.rand(n_actions, action_dim)

    cfg = DQNConfig(hidden_sizes=(8,), epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1, replay_warmup_steps=1)
    agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=0)
    state_enc = torch.randn(state_dim)

    selected = set()
    for _ in range(100):
        idx, mode = agent.select_action(state_enc)
        assert mode == "explore"
        selected.add(idx)
    assert len(selected) > 1, "epsilon=1 should explore multiple actions"


def test_fixed_seed_reproducible():
    """Same seed → same exploration sequence."""
    n_actions = 10
    state_dim = 2
    action_dim = 2
    action_tensors = torch.rand(n_actions, action_dim)

    cfg = DQNConfig(hidden_sizes=(4,), epsilon_start=1.0, epsilon_end=1.0, epsilon_decay_steps=1, replay_warmup_steps=1)

    def _run(seed):
        agent = DQNAgent(state_dim, action_dim, action_tensors, cfg, seed=seed)
        return [agent.select_action(torch.zeros(state_dim))[0] for _ in range(20)]

    assert _run(99) == _run(99), "Same seed must produce same exploration sequence"
    assert _run(1) != _run(2), "Different seeds should (very likely) differ"


if __name__ == "__main__":
    for fn in [test_argmax_selects_max_q, test_argmax_tie_break_smallest_id,
               test_epsilon_zero_always_greedy, test_epsilon_one_explores,
               test_fixed_seed_reproducible]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("\nAll Q-argmax tests: PASS")
