"""Digital-twin single-action closed-loop tests (Layers A-D + causality + determinism).

Proves the smallest real loop: S0 -> A0 -> real SymBChainSim PBFT execution -> measured
outcome -> reward_des_realized -> S1, with the analytical Liu oracle produced beside it.
"""
import sys
import unittest
from pathlib import Path

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Liu.Action import LiuAction
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import (
    CONTINUOUS_ESTIMATOR_VERSION,
    ContinuousSpatialIntensityModel,
    constant_intensity,
    planar_gradient_intensity,
)
from Liu.DigitalTwinPaired import DES_REWARD_POLICY, evaluate_des_observed
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    NETWORK_EVOLUTION_LABEL,
    LiuSingleActionSnapshot,
    _assemble,
    run_single_action,
)
from Chain.Consensus.LiuRuntime.LiuPBFT.Protocol import LiuPBFT
from Chain.Consensus.LiuRuntime.Processing.ProcessingCostModel import LiuProcessingCostModel
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork


N, K = 100, 21
VALIDATORS = tuple(range(K))


def golden_snapshot(*, link_scale=1.0, capability_scale=1.0):
    stakes = tuple(25.0 for _ in range(N))
    caps = tuple(float(10 + (i % 21)) * capability_scale for i in range(N))
    pos = tuple((0.1 * (i % 10), 0.1 * (i // 10)) for i in range(N))
    links = tuple(
        tuple(None if i == j else float(10 + ((i + j) % 91)) * link_scale for j in range(N))
        for i in range(N)
    )
    return LiuSingleActionSnapshot(
        node_count=N,
        transaction_size_bytes=200.0,
        stakes_tokens=stakes,
        capabilities_ghz=caps,
        positions_km=pos,
        link_rows_mbps=links,
        faulty_node_ids=(),
        epoch0_validator_ids=VALIDATORS,
    )


def golden_action():
    return LiuAction(N, K, VALIDATORS, LiuConsensusProtocol.PBFT, 0.2, 1.0)


def golden_geo(slope=0.6):
    return ContinuousSpatialIntensityModel(
        planar_gradient_intensity(K, slope), K, lambda_form=f"planar_gradient s={slope}"
    )


def reference_params():
    return LiuReferenceParameters(2_000_000.0, 1_000_000.0, 100.0, 6.0, 0.2, 0.3, 200)


def run_golden(**overrides):
    snapshot = overrides.pop("snapshot", None) or golden_snapshot()
    geo = overrides.pop("geo", None) or golden_geo()
    return run_single_action(
        snapshot,
        golden_action(),
        geo,
        reference_params=reference_params(),
        stake_gini_threshold=0.2,
        geographic_gini_threshold=0.3,
        finality_multiplier_omega=6.0,
        signature_cycles_alpha=2_000_000.0,
        mac_cycles_beta=1_000_000.0,
        offered_workload_tx=2000,
        workload_tx_size_mb=0.0002,
        max_events=500_000,
        **overrides,
    )


class LayerAPureTests(unittest.TestCase):
    def test_continuous_geo_constant_is_zero_and_normalized(self):
        model = ContinuousSpatialIntensityModel(constant_intensity(K), K, lambda_form="constant")
        self.assertAlmostEqual(model.geographic_gini(), 0.0, places=9)
        self.assertAlmostEqual(model.integrated_intensity(), K, places=6)
        self.assertEqual(model.evaluate().estimator_version, CONTINUOUS_ESTIMATOR_VERSION)

    def test_continuous_geo_gradient_is_nonzero_and_normalized(self):
        model = golden_geo(0.6)
        self.assertGreater(model.geographic_gini(), 0.0)
        self.assertLess(model.geographic_gini(), 0.3)
        self.assertAlmostEqual(model.integrated_intensity(), K, places=6)

    def test_unnormalized_lambda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not normalized to K"):
            ContinuousSpatialIntensityModel(constant_intensity(K + 5), K, lambda_form="wrong")

    def test_des_reward_zero_when_any_constraint_fails(self):
        record = evaluate_des_observed(
            selected_validator_stakes=[25.0] * K,
            geographic_gini_lambda=0.9,  # > threshold -> C1 fails
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            protocol=LiuConsensusProtocol.PBFT,
            validator_count=K,
            malicious_validator_count=0,
            block_interval_s=1.0,
            finality_multiplier_omega=6.0,
            t_c_des_s=0.4,
            offered_transactions=1000,
            included_transactions=999,
            finalized_transactions=999,
            measurement_duration_s=1.4,
        )
        self.assertFalse(record.feasible)
        self.assertEqual(record.reward_des_realized, 0.0)

    def test_des_reward_uses_finalized_tps_when_feasible(self):
        record = evaluate_des_observed(
            selected_validator_stakes=[25.0] * K,
            geographic_gini_lambda=0.1,
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            protocol=LiuConsensusProtocol.PBFT,
            validator_count=K,
            malicious_validator_count=0,
            block_interval_s=1.0,
            finality_multiplier_omega=6.0,
            t_c_des_s=0.4,
            offered_transactions=1000,
            included_transactions=800,
            finalized_transactions=800,
            measurement_duration_s=2.0,
        )
        self.assertTrue(record.feasible)
        self.assertEqual(record.reward_des_realized, 800 / 2.0)
        self.assertEqual(record.reward_policy_version, DES_REWARD_POLICY)


class LayerCWhiteBoxActionApplicationTests(unittest.TestCase):
    def test_action_application_changes_real_runtime_configuration(self):
        snapshot = golden_snapshot()
        action = golden_action()
        assembled = _assemble(
            snapshot,
            action,
            signature_cycles_alpha=2_000_000.0,
            mac_cycles_beta=1_000_000.0,
            request_timeout_s=100.0,
            propagation_delay_s=0.0,
        )
        # Bootstrap epoch 0 to height 1, then apply A0 and inspect real node state.
        from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import _height_finality_record, _run_to_height
        from Chain.TransactionFactory import Transaction, TransactionFactory

        for tx_id in range(2000):
            TransactionFactory.global_mempool.append(Transaction(0, tx_id, 0.0, 0.0002))
        _run_to_height(assembled, 1, 500_000)
        boundary = _height_finality_record(0, 1).finality_time
        assembled.applicator.apply(action, current_finalized_height=1, time=boundary)
        for node in assembled.nodes:
            self.assertEqual(node.reconfiguration_state.configuration.block_size, action.block_size_mb)
            self.assertEqual(node.reconfiguration_state.configuration.block_time, action.block_interval_s)
            self.assertIsInstance(node.cp, LiuPBFT)
        # Only the selected K participate as validators.
        for node in assembled.nodes:
            if node.id in set(action.validator_ids):
                self.assertTrue(node.is_validator)
            else:
                self.assertFalse(node.is_validator)


class LayerDGoldenGateTests(unittest.TestCase):
    def setUp(self):
        self.result = run_golden()
        self.d = self.result.to_dict()

    def test_state_s0_and_next_state_s1_present(self):
        self.assertIn("state_hash", self.d["state"])
        self.assertIn("state_hash", self.d["next_state"])
        self.assertEqual(self.d["next_state"]["network_evolution"], NETWORK_EVOLUTION_LABEL)
        self.assertEqual(len(self.d["state"]["stakes_tokens"]), N)

    def test_real_pbft_reaches_finality_with_transactions(self):
        self.assertGreater(self.d["des_observed"]["finalized_transactions"], 0)

    def test_c1_uses_liu_eq2_and_eq3_not_grid(self):
        des = self.d["des_observed"]
        self.assertEqual(des["stake_gini"], 0.0)  # Eq.2 over equal-stake validators
        self.assertAlmostEqual(des["geographic_gini"], golden_geo().geographic_gini(), places=9)

    def test_c2_uses_des_finality(self):
        des = self.d["des_observed"]
        self.assertAlmostEqual(des["t_f_des_s"], 1.0 + des["t_c_des_s"], places=9)
        self.assertTrue(des["c2_finality_passed"])

    def test_c3_uses_selected_faulty_count(self):
        des = self.d["des_observed"]
        self.assertEqual(des["malicious_validator_count"], 0)
        self.assertEqual(des["tolerated_fault_count"], (K - 1) // 3)

    def test_reward_des_realized_is_finalized_tps(self):
        des = self.d["des_observed"]
        self.assertTrue(des["feasible"])
        self.assertEqual(des["reward_des_realized"], des["throughput_des_finalized"])
        self.assertEqual(
            des["throughput_des_finalized"],
            des["finalized_transactions"] / des["measurement_duration_s"],
        )

    def test_reward_paper_reference_available_independently(self):
        paper = self.d["paper_reference"]
        self.assertEqual(paper["throughput_paper_nominal"], 1000.0)
        self.assertEqual(paper["reward_paper_reference"], 1000.0)

    def test_paired_delta_emitted(self):
        delta = self.d["delta"]
        self.assertIn("consensus_latency_delta_s", delta)
        self.assertIn("reward_delta", delta)
        self.assertTrue(delta["c1_agrees"] and delta["c2_agrees"] and delta["c3_agrees"])


class CausalityTests(unittest.TestCase):
    def test_network_slower_links_do_not_decrease_des_consensus_latency(self):
        fast = run_golden().to_dict()["des_observed"]["t_c_des_s"]
        slow = run_golden(snapshot=golden_snapshot(link_scale=0.5)).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreater(slow, fast)

    def test_compute_lower_capability_strictly_increases_processing_duration(self):
        model = LiuProcessingCostModel(2_000_000.0, 1_000_000.0)
        work = LiuProcessingWork.pbft_prepare_quorum_work(20)
        self.assertGreater(model.duration_s(work, 10.0), model.duration_s(work, 30.0))

    def test_compute_lower_capability_does_not_decrease_des_consensus_latency(self):
        base = run_golden().to_dict()["des_observed"]["t_c_des_s"]
        slower = run_golden(snapshot=golden_snapshot(capability_scale=0.5)).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreaterEqual(slower, base)


class DeterminismTests(unittest.TestCase):
    def test_same_snapshot_and_action_reproduce_identical_paired_result(self):
        first = run_golden().to_dict()
        second = run_golden().to_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
