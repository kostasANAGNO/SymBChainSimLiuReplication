import hashlib
import random
import sys
import unittest
from pathlib import Path

from dataclasses import FrozenInstanceError
from tests.test_liu_dynamic_epochs import DynamicEpochFixture
from tests.test_liu_pbft_runtime import LiuPBFTRuntimeFixture
from tests.test_liu_zyzzyva_runtime import LiuZyzzyvaRuntimeFixture
import numpy as np

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.RuntimeEvaluation import LiuRuntimeConstraintEvaluator, LiuRuntimeConstraintInput, LiuRuntimeExecutionMeasurement, LiuRewardStatus, RuntimeFinalizedBlockMeasurement
from Chain.Consensus.LiuRuntime.Common.StateBuilder import LiuRuntimeStateBuilder
from Chain.Consensus.LiuRuntime.Common.StateEvolution import LiuRuntimeStateEvolution
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import TIMEOUT_POLICY_VERSION
from Chain.Consensus.LiuRuntime.LiuPBFT import Messages as PBFTMessages, Transitions as PBFTTransitions
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase
from Chain.Consensus.LiuRuntime.LiuZyzzyva import Messages as ZyzzyvaMessages, Transitions as ZyzzyvaTransitions
from Chain.Consensus.LiuRuntime.LiuZyzzyva.State import LiuZyzzyvaPhase
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Utils.Instrumentation import InstrumentationCollector, TransactionCreationRecord


from Chain.Consensus.LiuRuntime.Common.Environment import (
    LiuEnvironmentConfig,
    LiuEnvironmentRuntimeAdapter,
    LiuEpochExecutionOutcome,
    LiuRuntimeEnvironment,
)
from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.Network import Network
from Engine.Handler import handle_event
from Liu import (
    AnalyticalConsensusResult,
    GridSpatialIntensityModel,
    LinkStateMatrix,
    LiuAction,
    LiuConsensusProtocol,
    LiuState,
    SpatialProfileSet,
    ThreatScenario,
)
from Liu.LinkFSMC import LinkFSMCState, LinkRateLevels, LinkTransitionMatrix, LinkTransitionTensor
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector


def digest(n):
    return hashlib.sha256(str(n).encode()).hexdigest()

class FixtureEnvironmentAdapter(LiuEnvironmentRuntimeAdapter):
    def __init__(self, config, *, forced_failure=False):
        self.fixture = RuntimeStateFixture(config.fsmc_seed)
        self._time = 20.0
        self.forced_failure = forced_failure
        self.transaction_number = 100
        self._mempool = TransactionFactory.global_mempool
        self._depth_removed = TransactionFactory.depth_removed
        self._produced_tx = TransactionFactory.produced_tx

    def activate(self):
        Network.nodes = list(self.nodes)
        Network.received = {node: set() for node in self.nodes}
        TransactionFactory.nodes = list(self.nodes)
        TransactionFactory.global_mempool = self._mempool
        TransactionFactory.depth_removed = self._depth_removed
        TransactionFactory.produced_tx = self._produced_tx

    def deactivate(self):
        self._mempool = TransactionFactory.global_mempool
        self._depth_removed = TransactionFactory.depth_removed
        self._produced_tx = TransactionFactory.produced_tx

    @property
    def state_evolution(self):
        return self.fixture.evolution

    @property
    def nodes(self):
        return tuple(self.fixture.runtime.nodes)

    @property
    def current_time(self):
        return self._time

    @property
    def current_finalized_height(self):
        return min(node.last_block.depth for node in self.nodes)

    def execute_one_decision_epoch(self, target_height, simulation_deadline):
        before = len(LiuRuntimeInstrumentationCollector.protocol_finalities)
        if self.forced_failure:
            self._time += 1.0
            return LiuEpochExecutionOutcome(
                self._time,
                self.current_finalized_height,
                (),
                "controlled unrecoverable safety failure",
            )
        self.transaction_number += 1
        creation_time = self._time + 0.01
        self.fixture.record_transaction(self.transaction_number, creation_time, 0.002)
        TransactionFactory.global_mempool.append(
            Transaction(0, self.transaction_number, creation_time, 0.001)
        )
        last_event_time = self._time
        for _ in range(5_000):
            if all(node.last_block.depth == target_height for node in self.nodes):
                self._time = last_event_time
                return LiuEpochExecutionOutcome(
                    self._time,
                    target_height,
                    tuple(LiuRuntimeInstrumentationCollector.protocol_finalities[before:]),
                )
            event = self.fixture.runtime.queue.pop_next_event()
            if simulation_deadline is not None and event.time >= simulation_deadline:
                self._time = simulation_deadline
                return LiuEpochExecutionOutcome(
                    self._time,
                    self.current_finalized_height,
                    tuple(LiuRuntimeInstrumentationCollector.protocol_finalities[before:]),
                    externally_truncated=True,
                )
            handle_event(event)
            last_event_time = max(last_event_time, event.time)
        raise AssertionError("fixture runtime did not complete one decision height")

    def analytical_result(self, action, measurement):
        tolerated = 0 if action.consensus_protocol is LiuConsensusProtocol.LIU_QUORUM else (
            action.validator_count - 1
        ) // 3
        return AnalyticalConsensusResult(
            action.consensus_protocol,
            0.25,
            0.25,
            0.5,
            action.block_interval_s,
            action.block_interval_s + 0.5,
            tolerated,
            (("controlled_environment_fixture", True),),
        )


def config(**overrides):
    values = dict(
        runtime_seed=31,
        fsmc_seed=37,
        candidate_seed=41,
        stake_gini_threshold=1.0,
        geographic_gini_threshold=1.0,
        finality_multiplier_omega=100.0,
        max_decision_steps=None,
        max_runtime_time_s=None,
    )
    values.update(overrides)
    return LiuEnvironmentConfig(**values)


def action(protocol=LiuConsensusProtocol.PBFT, validators=(1, 2, 3, 4)):
    return LiuAction(5, 4, validators, protocol, 2.0, 1.0)


class LiuRuntimeEnvironmentTests(unittest.TestCase):
    def environment(self, environment_config=None, *, failure=False):
        environment_config = environment_config or config()
        return LiuRuntimeEnvironment(
            environment_config,
            lambda supplied: FixtureEnvironmentAdapter(supplied, forced_failure=failure),
        )

    def test_reset_returns_deterministic_s0_without_reward(self):
        environment = self.environment()
        first = environment.reset()
        second = environment.reset()
        self.assertEqual(first, second)
        self.assertFalse(hasattr(first, "reward"))
        self.assertEqual(first.state_hash, first.observation.deterministic_hash())

    def test_step_binds_previous_state_action_reward_and_next_state(self):
        environment = self.environment()
        reset = environment.reset()
        selected = action()
        result = environment.step(selected)
        self.assertEqual(result.state_t_hash, reset.state_hash)
        self.assertEqual(result.action_t_hash, selected.deterministic_hash())
        self.assertEqual(result.state_t_plus_1_hash, result.observation.deterministic_hash())
        self.assertGreater(result.reward, 0)
        record = LiuRuntimeInstrumentationCollector.epoch_evaluations[-1]
        self.assertEqual(record.state_hash, result.state_t_hash)
        self.assertEqual(record.action_hash, result.action_t_hash)
        self.assertEqual(record.next_state_hash, result.state_t_plus_1_hash)
        self.assertEqual(environment.runtime.current_finalized_height, 2)

    def test_infeasible_action_does_not_terminate(self):
        environment = self.environment(config(geographic_gini_threshold=0.0))
        environment.reset()
        result = environment.step(action())
        self.assertEqual(result.status.value, "ACTION_INFEASIBLE")
        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.terminated)
        self.assertFalse(result.truncated)

    def test_execution_failure_terminates_without_valid_next_state(self):
        environment = self.environment(failure=True)
        environment.reset()
        result = environment.step(action())
        self.assertEqual(result.status.value, "EXECUTION_FAILED")
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertIsNone(result.observation)
        self.assertIsNone(result.state_t_plus_1_hash)

    def test_max_steps_is_terminal_only_after_completed_step(self):
        environment = self.environment(config(max_decision_steps=1))
        environment.reset()
        result = environment.step(action())
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertIsNotNone(result.observation)

    def test_runtime_budget_is_external_truncation(self):
        environment = self.environment(config(max_runtime_time_s=0.1))
        environment.reset()
        result = environment.step(action())
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(result.status.value, "EXECUTION_FAILED")

    def test_reset_replays_identical_fixed_action_trajectory(self):
        environment = self.environment(config(max_decision_steps=2))

        def run():
            initial = environment.reset()
            first = environment.step(action(LiuConsensusProtocol.PBFT))
            second = environment.step(action(LiuConsensusProtocol.ZYZZYVA, (0, 1, 2, 3)))
            return (
                initial.state_hash,
                first.action_t_hash,
                first.evaluation_hash,
                first.state_t_plus_1_hash,
                second.action_t_hash,
                second.evaluation_hash,
                second.state_t_plus_1_hash,
            )

        self.assertEqual(run(), run())

    def test_all_protocols_and_dynamic_switching_work_across_steps(self):
        environment = self.environment(config(max_decision_steps=3))
        environment.reset()
        rows = (
            environment.step(action(LiuConsensusProtocol.PBFT)),
            environment.step(action(LiuConsensusProtocol.ZYZZYVA, (0, 1, 2, 3))),
            environment.step(action(LiuConsensusProtocol.LIU_QUORUM, (1, 2, 3, 4))),
        )
        self.assertTrue(all(result.observation is not None for result in rows))
        self.assertEqual(environment.runtime.current_finalized_height, 4)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.epoch_evaluations), 3)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.runtime_states), 4)
        self.assertTrue(rows[-1].terminated)

    def test_instances_have_distinct_rng_runtime_epoch_and_instrumentation(self):
        first = self.environment(config(fsmc_seed=1, candidate_seed=2))
        second = self.environment(config(fsmc_seed=3, candidate_seed=4))
        first.reset()
        first_runtime = first.runtime
        second.reset()
        self.assertIsNot(first.instrumentation, second.instrumentation)
        self.assertIsNot(first_runtime, second.runtime)
        self.assertIsNot(first_runtime.state_evolution, second.runtime.state_evolution)
        self.assertNotEqual(first.candidate_rng.getstate(), second.candidate_rng.getstate())
        first_result = first.step(action())
        self.assertIsNotNone(first_result.observation)
        self.assertEqual(first.runtime.current_finalized_height, 2)
        self.assertEqual(second.runtime.current_finalized_height, 1)
        self.assertEqual(len(first.instrumentation.snapshot()), 3)
        self.assertEqual(len(second.instrumentation.snapshot()), 1)

    def test_environment_and_candidate_rng_do_not_consume_global_random(self):
        before = random.getstate()
        environment = self.environment()
        environment.reset()
        environment.step(action())
        self.assertEqual(before, random.getstate())


class EvaluationFixture:
    def __init__(
        self,
        *,
        protocol=LiuConsensusProtocol.PBFT,
        stakes=(1.0, 1.0, 1.0, 1.0, 1.0),
        coordinates=((0.25, 0.5), (1.25, 0.5), (2.25, 0.5), (3.25, 0.5), (3.75, 0.5)),
        malicious=(),
        consensus_latency=0.5,
        interval=1.0,
        block_size=2.0,
        transaction_size=2000.0,
        transaction_ids=(10, 11, 12),
        execution_failure_reason=None,
    ):
        self.protocol = protocol
        self.spatial = SpatialProfileSet.from_coordinates(
            coordinates, region_width_km=4.0, region_height_km=1.0
        )
        links = LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 20.0 for j in range(5)) for i in range(5))
        )
        self.state = LiuState(transaction_size, stakes, self.spatial, (20.0,) * 5, links)
        self.action = LiuAction(5, 4, (0, 1, 2, 3), protocol, block_size, interval)
        blocks = () if execution_failure_reason else (
            RuntimeFinalizedBlockMeasurement(7, 2, digest(1), 11.0, 11.0 + consensus_latency, transaction_ids),
        )
        self.measurement = LiuRuntimeExecutionMeasurement(
            7, protocol, 10.0, 20.0, blocks, execution_failure_reason
        )
        tolerated = 0 if protocol is LiuConsensusProtocol.LIU_QUORUM else 1
        self.analytical = AnalyticalConsensusResult(
            protocol, 0.25, 0.25, 0.5, interval, interval + 0.5, tolerated, (("fixture", True),)
        )
        self.threat = ThreatScenario(5, malicious)
        self.geo = GridSpatialIntensityModel(self.spatial, 1, 4)
        self.evaluator = LiuRuntimeConstraintEvaluator()

    def evaluate(self, *, stake_threshold=1.0, geo_threshold=1.0, omega=10.0):
        return self.evaluator.evaluate(
            LiuRuntimeConstraintInput(
                7,
                self.state,
                self.action,
                self.threat,
                self.geo,
                self.measurement,
                self.analytical,
                stake_threshold,
                geo_threshold,
                omega,
            )
        )


class LiuRuntimeConstraintTests(unittest.TestCase):
    def test_only_stake_gini_fails(self):
        result = EvaluationFixture(stakes=(1, 1, 1, 10, 1)).evaluate(stake_threshold=0.1)
        self.assertEqual(result.constraint_result.failed_constraints, ("stake_gini",))
        self.assertEqual(result.reward_result.status, LiuRewardStatus.ACTION_INFEASIBLE)

    def test_only_geographic_gini_fails(self):
        concentrated = ((0.1, 0.1), (0.2, 0.1), (0.3, 0.1), (0.4, 0.1), (3.5, 0.1))
        result = EvaluationFixture(coordinates=concentrated).evaluate(geo_threshold=0.0)
        self.assertEqual(result.constraint_result.failed_constraints, ("geographic_gini",))

    def test_only_security_fails(self):
        result = EvaluationFixture(malicious=(0, 1)).evaluate()
        self.assertEqual(result.constraint_result.failed_constraints, ("security",))

    def test_only_finality_fails_and_initial_request_is_not_reset(self):
        result = EvaluationFixture(consensus_latency=2.0).evaluate(omega=2.0)
        constraints = result.constraint_result
        self.assertEqual(constraints.failed_constraints, ("des_finality",))
        self.assertEqual(constraints.des_consensus_latency_s, 2.0)
        self.assertEqual(constraints.des_finality_latency_s, 3.0)
        self.assertEqual(constraints.finality_limit_s, 2.0)

    def test_all_pass_reward_and_observed_tps_are_exact(self):
        result = EvaluationFixture(transaction_ids=(1, 2, 3, 4)).evaluate()
        self.assertEqual(result.reward_result.status, LiuRewardStatus.FEASIBLE)
        self.assertEqual(result.reward_result.liu_throughput_tps, 1000.0)
        self.assertEqual(result.reward_result.reward, 1000.0)
        self.assertEqual(result.reward_result.observed_des_tps, 0.4)
        self.assertEqual(result.reward_result.analytical_consensus_latency_s, 0.5)
        self.assertEqual(result.reward_result.analytical_finality_latency_s, 1.5)

    def test_constraint_equality_passes(self):
        fixture = EvaluationFixture(stakes=(0, 0, 0, 1, 1))
        exact_stake = 0.75
        exact_geo = fixture.geo.evaluate(fixture.action.validator_ids).geographic_gini
        result = fixture.evaluate(stake_threshold=exact_stake, geo_threshold=exact_geo, omega=1.5)
        self.assertTrue(result.constraint_result.feasible)

    def test_protocol_fault_bounds(self):
        quorum = EvaluationFixture(protocol=LiuConsensusProtocol.LIU_QUORUM, malicious=(0,)).evaluate()
        self.assertFalse(quorum.constraint_result.security_constraint_passed)
        self.assertEqual(quorum.constraint_result.tolerated_fault_count, 0)
        pbft_at = EvaluationFixture(protocol=LiuConsensusProtocol.PBFT, malicious=(0,)).evaluate()
        pbft_above = EvaluationFixture(protocol=LiuConsensusProtocol.PBFT, malicious=(0, 1)).evaluate()
        zyzzyva_at = EvaluationFixture(protocol=LiuConsensusProtocol.ZYZZYVA, malicious=(0,)).evaluate()
        self.assertTrue(pbft_at.constraint_result.security_constraint_passed)
        self.assertFalse(pbft_above.constraint_result.security_constraint_passed)
        self.assertTrue(zyzzyva_at.constraint_result.security_constraint_passed)

    def test_block_smaller_than_chi_has_zero_capacity_and_reward(self):
        result = EvaluationFixture(block_size=0.2, transaction_size=300_000.0).evaluate()
        self.assertTrue(result.constraint_result.feasible)
        self.assertEqual(result.reward_result.liu_throughput_tps, 0.0)
        self.assertEqual(result.reward_result.reward, 0.0)

    def test_execution_failure_is_not_action_infeasibility(self):
        result = EvaluationFixture(execution_failure_reason="unrecoverable timeout").evaluate()
        self.assertEqual(result.reward_result.status, LiuRewardStatus.EXECUTION_FAILED)
        self.assertNotEqual(result.reward_result.status, LiuRewardStatus.ACTION_INFEASIBLE)
        self.assertEqual(result.reward_result.reward, 0.0)
        self.assertIn("des_finality", result.constraint_result.failed_constraints)

    def test_results_are_frozen_and_hashes_deterministic(self):
        first = EvaluationFixture().evaluate()
        second = EvaluationFixture().evaluate()
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())
        with self.assertRaises(FrozenInstanceError):
            first.reward_result.reward = 0

    def test_half_open_boundary_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "half-open"):
            LiuRuntimeExecutionMeasurement(
                7,
                LiuConsensusProtocol.PBFT,
                10.0,
                20.0,
                (RuntimeFinalizedBlockMeasurement(7, 2, digest(1), 19.0, 20.0, (1,)),),
            )

    def test_multiple_heights_use_unique_transactions_and_worst_height_latency(self):
        fixture = EvaluationFixture()
        measurement = LiuRuntimeExecutionMeasurement(
            7,
            LiuConsensusProtocol.PBFT,
            10.0,
            20.0,
            (
                RuntimeFinalizedBlockMeasurement(7, 2, digest(1), 11.0, 11.5, (1, 2)),
                RuntimeFinalizedBlockMeasurement(7, 3, digest(2), 12.0, 13.0, (2, 3)),
            ),
        )
        result = fixture.evaluator.evaluate(
            LiuRuntimeConstraintInput(
                7,
                fixture.state,
                fixture.action,
                fixture.threat,
                fixture.geo,
                measurement,
                fixture.analytical,
                1.0,
                1.0,
                10.0,
            )
        )
        self.assertEqual(result.constraint_result.des_consensus_latency_s, 1.0)
        self.assertEqual(result.reward_result.observed_des_tps, 0.3)


class CrossProtocolEvaluationTests(unittest.TestCase):
    def test_same_state_action_context_cross_protocol_table(self):
        rows = []
        for protocol in LiuConsensusProtocol:
            result = EvaluationFixture(protocol=protocol).evaluate()
            constraints = result.constraint_result
            reward = result.reward_result
            rows.append(
                (
                    protocol.value,
                    constraints.stake_gini,
                    constraints.geographic_gini,
                    constraints.tolerated_fault_count,
                    constraints.malicious_validator_count,
                    constraints.des_consensus_latency_s,
                    constraints.des_finality_latency_s,
                    reward.analytical_consensus_latency_s,
                    reward.analytical_finality_latency_s,
                    reward.observed_des_tps,
                    reward.liu_throughput_tps,
                    reward.feasible,
                    reward.reward,
                )
            )
        self.assertEqual([row[0] for row in rows], [item.value for item in LiuConsensusProtocol])
        self.assertEqual([row[3] for row in rows], [1, 1, 0])
        self.assertTrue(all(row[-2] for row in rows))
        self.assertTrue(all(row[-1] == 1000.0 for row in rows))


class RuntimeStateFixture:
    def __init__(self, seed=17):
        self.runtime = DynamicEpochFixture()
        profiles = NodeProfileSet(
            tuple(NodeProfile(node_id, float(node_id + 1), float(10 + node_id)) for node_id in range(5))
        )
        old = self.runtime.applicator.current_epoch
        context = LiuRuntimeEpochContext(old.epoch_configuration, profiles, old.link_state_matrix, old.threat_scenario)
        self.runtime.applicator.current_epoch = context
        self.runtime.applicator._contexts[context.epoch_id] = context
        for node in self.runtime.nodes:
            node._profile = profiles.profile_for(node.id)

        self.spatial = SpatialProfileSet.from_coordinates(
            ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0)),
            region_width_km=4,
            region_height_km=1,
        )
        model = GridSpatialIntensityModel(self.spatial, 1, 4)
        self.builder = LiuRuntimeStateBuilder(self.spatial, 0.002, model)
        levels = LinkRateLevels((10.0, 20.0))
        matrix = LinkTransitionMatrix(((0.45, 0.55), (0.35, 0.65)))
        self.fsmc = LinkFSMCState(levels, LinkTransitionTensor.shared(5, matrix), context.link_state_matrix)
        self.evolution = LiuRuntimeStateEvolution(
            self.runtime.applicator,
            self.builder,
            self.fsmc,
            random.Random(seed),
            initial_epoch_start_time=0.0,
        )

    def record_transaction(self, tx_id, creation_time, size_mb):
        InstrumentationCollector.record_transaction_creation(
            TransactionCreationRecord(tx_id, tx_id % 5, creation_time, size_mb)
        )

    def begin(self, protocol=LiuConsensusProtocol.PBFT, validators=(1, 2, 3, 4), boundary=20.0, height=1):
        action = self.runtime.action(validators, protocol, 2.0, 1.0)
        return self.evolution.begin_epoch(action, current_finalized_height=height, boundary_time=boundary)

    def complete(self, height=2, boundary=80.0):
        return self.evolution.complete_epoch(current_finalized_height=height, boundary_time=boundary)

    def execute_one_epoch(self, protocol=LiuConsensusProtocol.PBFT, validators=(1, 2, 3, 4)):
        self.begin(protocol, validators)
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        self.runtime.run_to_height(2)
        return self.complete()


class RuntimeStateBuilderTests(unittest.TestCase):
    def test_initial_state_has_all_n_components_in_node_order(self):
        fixture = RuntimeStateFixture()
        result = fixture.evolution.observe_initial_state()
        state = result.state
        self.assertEqual(state.node_count, 5)
        self.assertEqual(state.transaction_size_bytes, 2000.0)
        self.assertEqual(state.stakes_tokens, (1.0, 2.0, 3.0, 4.0, 5.0))
        self.assertEqual(state.computational_capabilities_ghz, (10.0, 11.0, 12.0, 13.0, 14.0))
        self.assertEqual(tuple(profile.node_id for profile in state.spatial_profiles.profiles), tuple(range(5)))
        self.assertEqual(state.link_state_matrix.node_count, 5)
        self.assertEqual(result.chi_source, "configured_workload_mean")
        self.assertEqual(result.chi_sample_count, 0)
        self.assertIsNotNone(result.geographic_gini)
        self.assertIsNotNone(result.stake_gini)

    def test_previous_epoch_empirical_chi_excludes_future_records(self):
        fixture = RuntimeStateFixture()
        initial = fixture.evolution.observe_initial_state()
        fixture.begin()
        fixture.record_transaction(1, 25.0, 0.001)
        fixture.record_transaction(2, 79.0, 0.003)
        fixture.record_transaction(3, 85.0, 1.0)
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        fixture.runtime.run_to_height(2)
        result = fixture.complete()
        self.assertEqual(result.state.transaction_size_bytes, 2000.0)
        self.assertEqual(result.chi_sample_count, 2)
        self.assertEqual(result.previous_state_hash, initial.state_hash)
        self.assertEqual(result.chi_source, "previous_completed_epoch_empirical_mean")

    def test_missing_previous_epoch_samples_is_explicitly_rejected(self):
        fixture = RuntimeStateFixture()
        fixture.evolution.observe_initial_state()
        fixture.begin()
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        fixture.runtime.run_to_height(2)
        before = fixture.evolution.link_fsmc_state
        with self.assertRaisesRegex(ValueError, "no transaction samples"):
            fixture.complete()
        self.assertIs(fixture.evolution.link_fsmc_state, before)

    def test_state_hash_and_instrumentation_chain_are_deterministic(self):
        fixture = RuntimeStateFixture()
        initial = fixture.evolution.observe_initial_state()
        fixture.begin()
        fixture.record_transaction(1, 25.0, 0.002)
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        fixture.runtime.run_to_height(2)
        result = fixture.complete()
        records = LiuRuntimeInstrumentationCollector.runtime_states
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].state_hash, initial.state.deterministic_hash())
        self.assertEqual(records[1].state_hash, result.state.deterministic_hash())
        self.assertEqual(records[1].previous_state_hash, records[0].state_hash)
        self.assertEqual(records[1].action_hash, result.action_hash)
        self.assertEqual(records[1].link_state_hash, result.state.link_state_matrix.deterministic_hash())

    def test_missing_all_node_stake_or_spatial_shape_is_rejected(self):
        fixture = RuntimeStateFixture()
        context = fixture.runtime.applicator.current_epoch
        missing = NodeProfileSet(tuple(NodeProfile(i, None, 20.0) for i in range(5)))
        invalid_context = LiuRuntimeEpochContext(
            context.epoch_configuration,
            missing,
            context.link_state_matrix,
            context.threat_scenario,
        )
        with self.assertRaisesRegex(ValueError, "explicit stake"):
            fixture.builder.build(invalid_context, observation_time=0.0, is_initial_state=True)


class RuntimeFSMCEvolutionTests(unittest.TestCase):
    def test_fsmc_changes_only_at_successful_epoch_boundary(self):
        fixture = RuntimeStateFixture()
        initial_links = fixture.evolution.link_fsmc_state.current_links
        fixture.evolution.observe_initial_state()
        self.assertIs(fixture.evolution.link_fsmc_state.current_links, initial_links)
        fixture.begin()
        self.assertIs(fixture.evolution.link_fsmc_state.current_links, initial_links)
        fixture.record_transaction(1, 25.0, 0.002)
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        fixture.runtime.run_to_height(2)
        result = fixture.complete()
        self.assertEqual(result.state.link_state_matrix, fixture.evolution.link_fsmc_state.current_links)
        self.assertNotEqual(result.state.link_state_matrix, initial_links)

    def test_in_flight_old_epoch_event_keeps_original_arrival(self):
        fixture = RuntimeStateFixture()
        fixture.evolution.observe_initial_state()
        old_event = next(event for _, event in fixture.runtime.queue.prio_queue.pq if hasattr(event, "liu_context"))
        old_arrival = old_event.time
        old_hash = old_event.liu_context.epoch_hash
        fixture.begin()
        self.assertEqual(old_event.time, old_arrival)
        self.assertEqual(old_event.liu_context.epoch_hash, old_hash)
        self.assertNotEqual(old_hash, fixture.runtime.applicator.current_epoch.epoch_hash)

    def test_dedicated_seed_controls_trajectory_without_global_rng(self):
        def run(seed):
            fixture = RuntimeStateFixture(seed)
            fixture.evolution.observe_initial_state()
            fixture.begin()
            fixture.record_transaction(1, 25.0, 0.002)
            TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
            fixture.runtime.run_to_height(2)
            global_before = random.getstate()
            numpy_before = repr(np.random.get_state())
            result = fixture.complete()
            return (
                global_before,
                random.getstate(),
                numpy_before,
                repr(np.random.get_state()),
                result.state.link_state_matrix.deterministic_hash(),
            )

        first = run(17)
        same = run(17)
        different = run(99)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        self.assertEqual(first[4], same[4])
        self.assertNotEqual(first[4], different[4])

    def test_validator_and_protocol_changes_preserve_all_n_static_vectors(self):
        fixture = RuntimeStateFixture()
        initial = fixture.evolution.observe_initial_state()
        fixture.begin(LiuConsensusProtocol.ZYZZYVA, (1, 2, 3, 4))
        fixture.record_transaction(1, 25.0, 0.002)
        TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
        fixture.runtime.run_to_height(2)
        changed = fixture.complete()
        self.assertEqual(changed.state.stakes_tokens, initial.state.stakes_tokens)
        self.assertEqual(changed.state.computational_capabilities_ghz, initial.state.computational_capabilities_ghz)
        self.assertEqual(changed.state.spatial_profiles, initial.state.spatial_profiles)
        self.assertEqual(changed.state.node_count, 5)
        self.assertEqual(changed.geographic_gini.selected_validator_ids, (1, 2, 3, 4))

    def test_same_state_action_schedule_produces_identical_state_sequence(self):
        def run_once():
            fixture = RuntimeStateFixture(1837413)
            states = [fixture.evolution.observe_initial_state().state_hash]
            fixture.begin()
            fixture.record_transaction(1, 25.0, 0.001)
            fixture.record_transaction(2, 28.0, 0.003)
            TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
            fixture.runtime.run_to_height(2)
            states.append(fixture.complete().state_hash)
            action = fixture.runtime.action((0, 1, 2, 3), LiuConsensusProtocol.ZYZZYVA, 4.0, 1.5)
            fixture.evolution.begin_epoch(action, current_finalized_height=2, boundary_time=80.0)
            fixture.record_transaction(3, 85.0, 0.004)
            TransactionFactory.global_mempool.append(Transaction(0, 13, 81.0, 0.001))
            fixture.runtime.run_to_height(3)
            states.append(fixture.evolution.complete_epoch(current_finalized_height=3, boundary_time=160.0).state_hash)
            return tuple(states), tuple(
                record.link_state_hash for record in LiuRuntimeInstrumentationCollector.runtime_states
            )

        self.assertEqual(run_once(), run_once())

    def test_epoch_evaluation_completes_state_action_reward_hash_chain(self):
        def run_once():
            fixture = RuntimeStateFixture(1837413)
            state_t = fixture.evolution.observe_initial_state()
            activation = fixture.begin()
            fixture.record_transaction(1, 25.0, 0.002)
            TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
            fixture.runtime.run_to_height(2)
            analytical = AnalyticalConsensusResult(
                LiuConsensusProtocol.PBFT,
                0.25,
                0.25,
                0.5,
                1.0,
                1.5,
                1,
                (("fixture", True),),
            )
            evaluation = fixture.evolution.evaluate_epoch(
                LiuRuntimeConstraintEvaluator(),
                analytical,
                boundary_time=80.0,
                stake_gini_threshold=1.0,
                geographic_gini_threshold=1.0,
                finality_multiplier_omega=100.0,
            )
            state_next = fixture.evolution.complete_epoch(
                current_finalized_height=2,
                boundary_time=80.0,
                evaluation=evaluation,
            )
            record = LiuRuntimeInstrumentationCollector.epoch_evaluations[-1]
            self.assertEqual(record.state_hash, state_t.state_hash)
            self.assertEqual(record.action_hash, activation.action_hash)
            self.assertEqual(record.execution_measurement_hash, evaluation.execution_measurement.deterministic_hash())
            self.assertEqual(record.constraint_result_hash, evaluation.constraint_result.deterministic_hash())
            self.assertEqual(record.reward_result_hash, evaluation.reward_result.deterministic_hash())
            self.assertEqual(record.next_state_hash, state_next.state_hash)
            self.assertEqual(evaluation.reward_result.observed_des_tps, 1 / 60)
            return record, evaluation.deterministic_hash(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        self.assertEqual(run_once(), run_once())

    def test_all_three_runtime_protocols_produce_evaluable_des_outcomes(self):
        rows = []
        for protocol in LiuConsensusProtocol:
            fixture = RuntimeStateFixture(1837413)
            fixture.evolution.observe_initial_state()
            fixture.begin(protocol=protocol)
            fixture.record_transaction(1, 25.0, 0.002)
            TransactionFactory.global_mempool.append(Transaction(0, 12, 21.0, 0.001))
            fixture.runtime.run_to_height(2)
            tolerated = 0 if protocol is LiuConsensusProtocol.LIU_QUORUM else 1
            analytical = AnalyticalConsensusResult(
                protocol, 0.25, 0.25, 0.5, 1.0, 1.5, tolerated, (("fixture", True),)
            )
            evaluation = fixture.evolution.evaluate_epoch(
                LiuRuntimeConstraintEvaluator(),
                analytical,
                boundary_time=80.0,
                stake_gini_threshold=1.0,
                geographic_gini_threshold=1.0,
                finality_multiplier_omega=100.0,
            )
            rows.append(
                (
                    protocol.value,
                    evaluation.constraint_result.des_consensus_latency_s,
                    evaluation.constraint_result.des_finality_latency_s,
                    evaluation.reward_result.observed_des_tps,
                    evaluation.reward_result.reward,
                )
            )
        self.assertEqual(tuple(row[0] for row in rows), tuple(protocol.value for protocol in LiuConsensusProtocol))
        self.assertTrue(all(row[1] is not None and row[1] >= 0 for row in rows))
        self.assertTrue(all(row[2] == 1.0 + row[1] for row in rows))
        self.assertTrue(all(row[3] == 1 / 60 for row in rows))
        self.assertTrue(all(row[4] == 1000.0 for row in rows))


class PhaseRelativeTimeoutTests(unittest.TestCase):
    def test_pbft_block_interval_boundaries_never_timeout_before_dispatch(self):
        for interval in (2.5, 5.0, 8.0, 50.0):
            with self.subTest(block_interval_s=interval):
                fixture = LiuPBFTRuntimeFixture(
                    request_timeout_s=5.0, block_interval_s=interval, alpha=0.0, beta=0.0
                )
                self.assertFalse(any(item[1].payload["type"] in (PBFTMessages.TIMEOUT, PBFTMessages.VIEW_TIMEOUT)
                                     for item in fixture.queue.prio_queue.pq))
                dispatched, _ = fixture.step()
                self.assertEqual(dispatched.payload["type"], PBFTMessages.START_REQUEST)
                timers = [item[1] for item in fixture.queue.prio_queue.pq
                          if item[1].payload["type"] in (PBFTMessages.TIMEOUT, PBFTMessages.VIEW_TIMEOUT)]
                self.assertTrue(timers)
                self.assertTrue(all(timer.time >= interval + 5.0 for timer in timers))
                fixture.run_to_all_observed()
                self.assertFalse(any(record.status == "fired" for record in LiuRuntimeInstrumentationCollector.view_changes))

    def test_zyzzyva_block_interval_boundaries_never_timeout_before_dispatch(self):
        for interval in (2.5, 5.0, 8.0, 50.0):
            with self.subTest(block_interval_s=interval):
                fixture = LiuZyzzyvaRuntimeFixture(
                    timeout_s=5.0, block_interval_s=interval, alpha=0.0, beta=0.0
                )
                self.assertFalse(any(item[1].payload["type"] in (ZyzzyvaMessages.TIMEOUT, ZyzzyvaMessages.VIEW_TIMEOUT)
                                     for item in fixture.queue.prio_queue.pq))
                dispatched, _ = fixture.step()
                self.assertEqual(dispatched.payload["type"], ZyzzyvaMessages.START_REQUEST)
                timers = [item[1] for item in fixture.queue.prio_queue.pq
                          if item[1].payload["type"] in (ZyzzyvaMessages.TIMEOUT, ZyzzyvaMessages.VIEW_TIMEOUT)]
                self.assertTrue(timers)
                self.assertTrue(all(timer.time > interval for timer in timers))
                fixture.run_to_all_observed()
                self.assertFalse(any(record.status == "fired" for record in LiuRuntimeInstrumentationCollector.view_changes))

    def test_old_epoch_relative_pbft_timer_reproduces_premature_view_change(self):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=5.0, block_interval_s=8.0, alpha=0.0, beta=0.0)
        replica = fixture.nodes[1].cp
        PBFTMessages.schedule_view_timeout(replica, 5.0, replica.timeout_identity(), LiuPBFTPhase.WAITING_PREPREPARE)
        timeout, _ = fixture.step()
        self.assertEqual(timeout.payload["type"], PBFTMessages.VIEW_TIMEOUT)
        self.assertEqual(timeout.time, 5.0)
        self.assertTrue(any(record.status == "fired" for record in LiuRuntimeInstrumentationCollector.view_changes))

    def test_stale_phase_timers_are_ignored(self):
        pbft = LiuPBFTRuntimeFixture(request_timeout_s=5.0)
        pbft.step()
        timer = next(item[1] for item in pbft.queue.prio_queue.pq
                     if item[1].payload["type"] == PBFTMessages.VIEW_TIMEOUT)
        timer.actor.cp.state.phase = LiuPBFTPhase.PREPARE_COLLECTING
        self.assertEqual(PBFTTransitions.view_timeout(timer.actor.cp, timer), "handled")
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].detail, "stale_phase")

        zyzzyva = LiuZyzzyvaRuntimeFixture(timeout_s=5.0)
        zyzzyva.step()
        timer = next(item[1] for item in zyzzyva.queue.prio_queue.pq
                     if item[1].payload["type"] == ZyzzyvaMessages.VIEW_TIMEOUT)
        timer.actor.cp.state.phase = LiuZyzzyvaPhase.PROCESSING_SPECULATIVE
        self.assertEqual(ZyzzyvaTransitions.view_timeout(timer.actor.cp, timer), "handled")
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].detail, "stale_phase")

    def test_policy_version_is_frozen(self):
        self.assertEqual(TIMEOUT_POLICY_VERSION, "liu_runtime_phase_relative_timeout_v2")


if __name__ == "__main__":
    unittest.main()
