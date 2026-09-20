"""Tests for the 2400-action controlled full sweep runner (§11: grid shape + resume)."""
import csv
import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
SWEEP_DIR = ROOT / "experiments" / "liu_replication" / "controlled_sweep"


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


class FullGridShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load("controlled_full_runner", SWEEP_DIR / "run_controlled_full.py")

    def test_grid_has_exactly_2400_unique_actions(self):
        grid = self.runner.full_action_grid()
        self.assertEqual(len(grid), 2400)
        self.assertEqual(len(set(grid)), 2400)

    def test_grid_endpoints_and_domain(self):
        grid = self.runner.full_action_grid()
        protocols = {a[0] for a in grid}
        self.assertEqual(protocols, {"PBFT", "ZYZZYVA", "LIU_QUORUM"})
        sbs = sorted({a[1] for a in grid})
        self.assertEqual(sbs[0], 0.2)
        self.assertEqual(sbs[-1], 8.0)
        self.assertEqual(len(sbs), 40)
        tis = sorted({a[2] for a in grid})
        self.assertEqual(tis[0], 0.5)
        self.assertEqual(tis[-1], 10.0)
        self.assertEqual(len(tis), 20)

    def test_grid_is_deterministic(self):
        self.assertEqual(self.runner.full_action_grid(), self.runner.full_action_grid())

    def test_validate_grid_accepts_full_grid(self):
        self.runner._validate_grid(self.runner.full_action_grid())

    def test_validate_grid_rejects_wrong_length(self):
        with self.assertRaisesRegex(AssertionError, "2400"):
            self.runner._validate_grid(self.runner.full_action_grid()[:100])


class FixedS0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load("controlled_full_runner", SWEEP_DIR / "run_controlled_full.py")

    def test_snapshot_has_100_nodes_21_validators_f0(self):
        s = self.runner.build_snapshot()
        self.assertEqual(s.node_count, 100)
        self.assertEqual(len(s.epoch0_validator_ids), 21)
        self.assertEqual(s.faulty_node_ids, ())


class ResumeBehaviorTests(unittest.TestCase):
    """On a tiny subset, previously-SUCCESS rows must be skipped on restart."""

    @classmethod
    def setUpClass(cls):
        cls.runner = _load("controlled_full_runner_for_resume", SWEEP_DIR / "run_controlled_full.py")

    def _tiny_subset(self):
        # 3 cheap actions (small S_B, T_I): near-zero wall time each.
        return (
            ("PBFT", 0.2, 1.0),
            ("ZYZZYVA", 0.2, 1.0),
            ("LIU_QUORUM", 0.2, 1.0),
        )

    def _fresh_csv(self, tmp_path):
        original = self.runner.CSV_PATH
        # Point the runner at a temporary CSV for this test only.
        self.runner.CSV_PATH = tmp_path
        return original

    def test_resume_skips_successful_rows(self):
        tmp = SWEEP_DIR / "resume_test_tmp.csv"
        if tmp.exists():
            tmp.unlink()
        original = self._fresh_csv(tmp)
        try:
            first = self.runner.main(actions_override=self._tiny_subset())
            self.assertEqual(len(first), 3)
            first_walls = {i: float(r["wall_clock_seconds"]) for i, r in first.items()}
            # Sanity: at least one succeeded so resume has something to skip.
            successful = [i for i, r in first.items() if r.get("run_status") == "SUCCESS"]
            self.assertTrue(successful, "no successful action in tiny subset — cannot test resume")

            # Second run must re-read the CSV, skip successful rows, and only re-run failures.
            second = self.runner.main(actions_override=self._tiny_subset())
            for i in successful:
                # Skipped rows keep their prior wall_clock_seconds byte-for-byte.
                self.assertAlmostEqual(
                    float(second[i]["wall_clock_seconds"]),
                    first_walls[i],
                    places=9,
                    msg=f"action {i} was re-executed on resume (wall time changed)",
                )
        finally:
            self.runner.CSV_PATH = original
            if tmp.exists():
                tmp.unlink()

    def test_csv_written_incrementally_after_each_action(self):
        tmp = SWEEP_DIR / "resume_test_incremental_tmp.csv"
        if tmp.exists():
            tmp.unlink()
        original = self._fresh_csv(tmp)
        try:
            self.runner.main(actions_override=self._tiny_subset()[:1])
            self.assertTrue(tmp.exists())
            with tmp.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["action_id"]), 0)
        finally:
            self.runner.CSV_PATH = original
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    unittest.main()
