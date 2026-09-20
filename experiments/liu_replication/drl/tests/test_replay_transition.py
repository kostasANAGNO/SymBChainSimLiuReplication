"""Test: replay buffer round-trip and sampling."""
import sys
from pathlib import Path
import torch

DRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL))

from replay_buffer import ReplayBuffer


def test_replay_push_and_len():
    buf = ReplayBuffer(capacity=10, seed=0)
    assert len(buf) == 0
    for i in range(5):
        buf.push(torch.zeros(4), torch.ones(3), float(i), torch.ones(4), i % 2 == 0)
    assert len(buf) == 5


def test_replay_capacity_eviction():
    buf = ReplayBuffer(capacity=5, seed=0)
    for i in range(10):
        buf.push(torch.tensor([float(i)]), torch.zeros(2), float(i), torch.zeros(1), False)
    assert len(buf) == 5


def test_replay_sample_shape():
    buf = ReplayBuffer(capacity=100, seed=42)
    state_dim, action_dim = 6, 4
    for i in range(50):
        buf.push(torch.rand(state_dim), torch.rand(action_dim), float(i), torch.rand(state_dim), False)
    states, actions, rewards, next_states, dones = buf.sample(16)
    assert states.shape == (16, state_dim)
    assert actions.shape == (16, action_dim)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, state_dim)
    assert dones.shape == (16,)
    assert dones.dtype == torch.bool


def test_replay_round_trip_values():
    """Values pushed must be retrievable exactly (with appropriate batch size)."""
    buf = ReplayBuffer(capacity=1, seed=0)
    s = torch.tensor([1.0, 2.0, 3.0])
    a = torch.tensor([0.5])
    r = 42.0
    ns = torch.tensor([4.0, 5.0, 6.0])
    done = True
    buf.push(s, a, r, ns, done)
    states, actions, rewards, next_states, dones = buf.sample(1)
    assert torch.allclose(states[0], s)
    assert torch.allclose(actions[0], a)
    assert float(rewards[0]) == r
    assert torch.allclose(next_states[0], ns)
    assert bool(dones[0]) == done


def test_replay_deterministic_sampling():
    """Same seed → same sample."""
    buf1 = ReplayBuffer(capacity=100, seed=42)
    buf2 = ReplayBuffer(capacity=100, seed=42)
    for i in range(30):
        t = torch.tensor([float(i)])
        buf1.push(t, t, float(i), t, False)
        buf2.push(t, t, float(i), t, False)
    s1, *_ = buf1.sample(10)
    s2, *_ = buf2.sample(10)
    assert torch.equal(s1, s2), "Same seed must produce same sample"


if __name__ == "__main__":
    for fn in [test_replay_push_and_len, test_replay_capacity_eviction,
               test_replay_sample_shape, test_replay_round_trip_values,
               test_replay_deterministic_sampling]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("\nAll replay buffer tests: PASS")
