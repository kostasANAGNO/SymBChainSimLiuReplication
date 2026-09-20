"""Test: reduced action domain structure and uniqueness."""
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DRL = ROOT / "experiments" / "liu_replication" / "drl"
sys.path.insert(0, str(DRL))

from build_action_domain import build_domain, _all_validator_sets, _select_top_configs, SWEEP_CSV, N, K


def test_validator_sets_count():
    vsets = _all_validator_sets(30, 28)
    assert len(vsets) == 435, f"Expected C(30,28)=435, got {len(vsets)}"


def test_validator_sets_are_k_subsets():
    vsets = _all_validator_sets(30, 28)
    for vs in vsets:
        assert len(vs) == 28
        assert all(0 <= v < 30 for v in vs)
        assert vs == tuple(sorted(vs)), "Validator set not sorted"


def test_top_configs_per_protocol():
    configs = _select_top_configs(SWEEP_CSV)
    for p in ["PBFT", "ZYZZYVA", "LIU_QUORUM"]:
        assert p in configs
        assert len(configs[p]) <= 3
        assert len(configs[p]) >= 1
        # All (S_B, T_I) pairs are distinct
        pairs = [(sb, ti) for sb, ti, *_ in configs[p]]
        assert len(pairs) == len(set(pairs)), f"Duplicate (S_B, T_I) for {p}"


def test_build_domain_produces_correct_count():
    meta = build_domain()
    n_configs = sum(len(cfgs) for cfgs in meta["protocol_configs"].values())
    expected = meta["c_n_k"] * n_configs
    assert meta["total_actions"] == expected, f"Expected {expected}, got {meta['total_actions']}"


def test_all_actions_unique():
    meta = build_domain()
    import csv
    out_csv = DRL / "reduced_action_domain.csv"
    keys = set()
    with open(out_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["protocol"], row["block_size_mb"], row["block_interval_s"], row["validator_ids"])
            assert key not in keys, f"Duplicate action: {key}"
            keys.add(key)
    assert len(keys) == meta["total_actions"]


def test_validator_mask_consistent_with_ids():
    import csv
    out_csv = DRL / "reduced_action_domain.csv"
    with open(out_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = json.loads(row["validator_ids"])
            mask = json.loads(row["validator_mask"])
            assert len(mask) == N
            assert sum(mask) == K
            for i, m in enumerate(mask):
                if i in set(ids):
                    assert m == 1, f"Mask inconsistent at node {i}"
                else:
                    assert m == 0, f"Mask inconsistent at node {i}"


if __name__ == "__main__":
    test_validator_sets_count()
    print("test_validator_sets_count: PASS")
    test_validator_sets_are_k_subsets()
    print("test_validator_sets_are_k_subsets: PASS")
    test_top_configs_per_protocol()
    print("test_top_configs_per_protocol: PASS")
    test_build_domain_produces_correct_count()
    print("test_build_domain_produces_correct_count: PASS")
    test_all_actions_unique()
    print("test_all_actions_unique: PASS")
    test_validator_mask_consistent_with_ids()
    print("test_validator_mask_consistent_with_ids: PASS")
    print("\nAll reduced action domain tests: PASS")
