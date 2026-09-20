"""Test: state and action encoding determinism and shape."""
import sys
from pathlib import Path
import torch

DRL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL))

from config import DEFAULT_CONFIG
from state_encoder import StateEncoder
from action_encoder import ActionEncoder

N = 6

def _make_state(rate: float = 100.0) -> dict:
    return {
        "chi_bytes": 200.0,
        "stakes": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "positions_km": [(0.2,0.0),(0.4,0.0),(0.6,0.0),(0.2,0.5),(0.4,0.5),(0.6,0.5)],
        "capabilities_ghz": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        "link_rows_mbps": tuple(
            tuple(None if i == j else rate for j in range(N))
            for i in range(N)
        ),
    }

def _make_action(proto="PBFT", sb=1.0, ti=2.0, vids=(0,1,2,3)) -> dict:
    mask = [1 if i in set(vids) else 0 for i in range(N)]
    return {"protocol": proto, "block_size_mb": sb, "block_interval_s": ti,
            "validator_mask": mask, "validator_ids": list(vids)}


def test_state_encoder_dim():
    enc = StateEncoder(N)
    expected = 1 + N + 2*N + N + N*(N-1)  # 1+6+12+6+30=55
    assert enc.dim == expected, f"Expected {expected}, got {enc.dim}"


def test_state_encoding_shape():
    enc = StateEncoder(N)
    state = _make_state()
    t = enc.encode(state)
    assert t.shape == (enc.dim,)
    assert t.dtype == torch.float32


def test_state_encoding_deterministic():
    enc = StateEncoder(N)
    state = _make_state(rate=55.0)
    assert torch.equal(enc.encode(state), enc.encode(state))


def test_state_different_rates_differ():
    enc = StateEncoder(N)
    t1 = enc.encode(_make_state(rate=100.0))
    t2 = enc.encode(_make_state(rate=10.0))
    assert not torch.equal(t1, t2), "Different link rates should produce different encodings"


def test_action_encoder_dim():
    enc = ActionEncoder(N)
    expected = N + 3 + 1 + 1  # 6+3+1+1=11
    assert enc.dim == expected


def test_action_encoding_shape():
    enc = ActionEncoder(N)
    a = _make_action()
    t = enc.encode(a)
    assert t.shape == (enc.dim,)
    assert t.dtype == torch.float32


def test_action_encoding_deterministic():
    enc = ActionEncoder(N)
    a = _make_action()
    assert torch.equal(enc.encode(a), enc.encode(a))


def test_different_protocols_differ():
    enc = ActionEncoder(N)
    a1 = _make_action(proto="PBFT")
    a2 = _make_action(proto="ZYZZYVA")
    assert not torch.equal(enc.encode(a1), enc.encode(a2))


def test_different_validators_differ():
    enc = ActionEncoder(N)
    a1 = _make_action(vids=(0,1,2,3))
    a2 = _make_action(vids=(2,3,4,5))
    assert not torch.equal(enc.encode(a1), enc.encode(a2))


def test_normalization_range():
    enc_s = StateEncoder(N)
    enc_a = ActionEncoder(N)
    s = enc_s.encode(_make_state(rate=100.0))
    a = enc_a.encode(_make_action(sb=8.0, ti=10.0))
    # All values should be in [0, 1] after normalization
    assert float(s.min()) >= -0.01
    assert float(s.max()) <= 1.01
    assert float(a.min()) >= -0.01
    assert float(a.max()) <= 1.01


if __name__ == "__main__":
    for fn in [test_state_encoder_dim, test_state_encoding_shape, test_state_encoding_deterministic,
               test_state_different_rates_differ, test_action_encoder_dim, test_action_encoding_shape,
               test_action_encoding_deterministic, test_different_protocols_differ,
               test_different_validators_differ, test_normalization_range]:
        fn()
        print(f"{fn.__name__}: PASS")
    print("\nAll encoding tests: PASS")
