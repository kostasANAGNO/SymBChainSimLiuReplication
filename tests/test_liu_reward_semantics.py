"""Reward-semantics regression tests (§9): the DES-side Liu-compatible reward reuses
Liu's Omega and is gated only by DES C1/C2/C3, without double-counting T_C_DES.
"""
import sys
import unittest
from pathlib import Path

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Chain.ValidatorSet import ValidatorSet
from Chain.Consensus.LiuRuntime.Common.Roles import LiuZyzzyvaRoles  # noqa: F401 (parity with sibling test files)
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot,
    run_single_action,
)
from Liu.Action import LiuAction
from Liu.DigitalTwinPaired import (
    DES_LIU_OBJECTIVE_POLICY,
    DES_REWARD_POLICY,
    LiuPaperReferenceRecord,
    LiuDigitalTwinDelta,
    evaluate_des_observed,
)
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity


N, K = 100, 21
VALIDATORS = tuple(range(K))


def _snapshot():
    return LiuSingleActionSnapshot(
        node_count=N,
        transaction_size_bytes=200.0,
        stakes_tokens=tuple(25.0 for _ in range(N)),
        capabilities_ghz=tuple(float(10 + (i % 21)) for i in range(N)),
        positions_km=tuple((0.1 * (i % 10), 0.1 * (i // 10)) for i in range(N)),
        link_rows_mbps=tuple(
            tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N))
            for i in range(N)
        ),
        faulty_node_ids=(),
        epoch0_validator_ids=VALIDATORS,
    )


def _run(protocol: LiuConsensusProtocol, sb: float, ti: float):
    return run_single_action(
        _snapshot(),
        LiuAction(N, K, VALIDATORS, protocol, sb, ti),
        ContinuousSpatialIntensityModel(planar_gradient_intensity(K, 0.6), K, lambda_form="g"),
        reference_params=LiuReferenceParameters(2_000_000.0, 1_000_000.0, 100.0, 6.0, 0.2, 0.3, 200),
        stake_gini_threshold=0.2,
        geographic_gini_threshold=0.3,
        finality_multiplier_omega=6.0,
        signature_cycles_alpha=2_000_000.0,
        mac_cycles_beta=1_000_000.0,
        offered_workload_tx=100_000,
        workload_tx_size_mb=0.0002,
        max_events=1_000_000,
    )


class PureEvaluatorSemantics(unittest.TestCase):
    """Tests 1-3, 5, 6, 7 (§9): evaluator-level semantics without running the DES."""

    def _evaluate(self, *, malicious, geographic_gini_lambda, t_c_des_s, block_interval_s,
                  finality_multiplier_omega, throughput_paper_nominal):
        return evaluate_des_observed(
            selected_validator_stakes=[25.0] * K,
            geographic_gini_lambda=geographic_gini_lambda,
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            protocol=LiuConsensusProtocol.PBFT,
            validator_count=K,
            malicious_validator_count=malicious,
            block_interval_s=block_interval_s,
            finality_multiplier_omega=finality_multiplier_omega,
            t_c_des_s=t_c_des_s,
            offered_transactions=1000,
            included_transactions=999,
            finalized_transactions=999,
            measurement_duration_s=1.42,
            throughput_paper_nominal=throughput_paper_nominal,
        )

    def test_reward_des_liu_objective_equals_omega_when_all_des_constraints_pass(self):
        r = self._evaluate(malicious=0, geographic_gini_lambda=0.1, t_c_des_s=0.4,
                           block_interval_s=1.0, finality_multiplier_omega=6.0,
                           throughput_paper_nominal=1000.0)
        self.assertTrue(r.feasible)
        self.assertEqual(r.reward_des_liu_objective, 1000.0)          # Omega, not finalized_tps
        self.assertEqual(r.throughput_paper_nominal, 1000.0)
        self.assertEqual(r.reward_des_liu_objective_policy_version, DES_LIU_OBJECTIVE_POLICY)

    def test_reward_des_liu_objective_is_zero_when_des_c2_fails(self):
        # omega * T_I = 6 * 0.1 = 0.6 s; T_F_DES = 0.1 + 0.7 = 0.8 > 0.6 => C2 fails
        r = self._evaluate(malicious=0, geographic_gini_lambda=0.1, t_c_des_s=0.7,
                           block_interval_s=0.1, finality_multiplier_omega=6.0,
                           throughput_paper_nominal=1000.0)
        self.assertFalse(r.c2_finality_passed)
        self.assertEqual(r.reward_des_liu_objective, 0.0)

    def test_reward_des_liu_objective_is_zero_when_des_c3_fails(self):
        # PBFT with K=21 tolerates 6 malicious; use 7 to violate.
        r = self._evaluate(malicious=7, geographic_gini_lambda=0.1, t_c_des_s=0.4,
                           block_interval_s=1.0, finality_multiplier_omega=6.0,
                           throughput_paper_nominal=1000.0)
        self.assertFalse(r.c3_security_passed)
        self.assertEqual(r.reward_des_liu_objective, 0.0)

    def test_single_epoch_finalized_tps_stays_separately_calculated(self):
        r = self._evaluate(malicious=0, geographic_gini_lambda=0.1, t_c_des_s=0.4,
                           block_interval_s=1.0, finality_multiplier_omega=6.0,
                           throughput_paper_nominal=1000.0)
        # Diagnostic metric: finalized / duration; NOT the Liu objective.
        self.assertAlmostEqual(r.throughput_des_finalized, 999 / 1.42, places=6)
        self.assertAlmostEqual(r.reward_des_realized, 999 / 1.42, places=6)
        # Two different rewards on the same record:
        self.assertNotEqual(r.reward_des_realized, r.reward_des_liu_objective)


class SingleActionIntegration(unittest.TestCase):
    """Tests 1, 2, 4 (§9): end-to-end via SingleActionRuntime."""

    def test_paper_reward_still_equals_omega_when_paper_feasible(self):
        # PBFT 0.2 MB / 1 s is paper-feasible: Omega = 1000, paper reward = 1000.
        r = _run(LiuConsensusProtocol.PBFT, 0.2, 1.0)
        p = r.paper_reference
        self.assertTrue(p.c1_decentralization_passed and p.c2_finality_passed and p.c3_security_passed)
        self.assertEqual(p.throughput_paper_nominal, 1000.0)
        self.assertEqual(p.reward_paper_reference, 1000.0)

    def test_pbft_2mb_1s_regression_paper_c2_fail_des_c2_pass(self):
        """The key semantic regression: paper says infeasible, DES says feasible."""
        r = _run(LiuConsensusProtocol.PBFT, 2.0, 1.0)
        p, d = r.paper_reference, r.des_observed
        self.assertFalse(p.c2_finality_passed)
        self.assertTrue(d.c2_finality_passed)
        # Paper reward = 0 (paper C2 fails), Omega = 10000 (2 MB / 200 B / 1 s).
        self.assertEqual(p.throughput_paper_nominal, 10000.0)
        self.assertEqual(p.reward_paper_reference, 0.0)
        # DES-Liu-objective = Omega gated by DES constraints = 10000 (DES C1/C2/C3 all pass).
        self.assertEqual(d.reward_des_liu_objective, 10000.0)
        # Single-epoch finalized TPS diagnostic still ~1930 (unchanged behavior).
        self.assertGreater(d.reward_des_realized, 1000.0)
        self.assertLess(d.reward_des_realized, 5000.0)
        # And the two rewards on the DES side are meaningfully different.
        self.assertNotAlmostEqual(d.reward_des_liu_objective, d.reward_des_realized, places=0)
        # Delta on the Liu objective is exactly Omega (one side gates it to 0).
        self.assertEqual(r.delta.reward_liu_objective_delta, 10000.0)


class BackwardCompatibility(unittest.TestCase):
    def test_reward_des_realized_and_policy_unchanged(self):
        # Existing DES realized reward field + policy label must keep the historical meaning.
        r = _run(LiuConsensusProtocol.LIU_QUORUM, 0.2, 1.0)
        d = r.des_observed
        self.assertTrue(d.feasible)
        self.assertAlmostEqual(
            d.reward_des_realized,
            d.finalized_transactions / d.measurement_duration_s,
            places=6,
        )
        self.assertEqual(d.reward_policy_version, DES_REWARD_POLICY)


if __name__ == "__main__":
    unittest.main()
