import math
import random
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch


from enum import Enum
from types import SimpleNamespace
import json
import numpy as np
import tempfile

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Liu import (
    GridSpatialIntensityModel,
    LinkStateMatrix,
    LiuAction,
    LiuActionCandidateDomain,
    LiuActionCandidateGenerator,
    LiuActionTensorEncoder,
    LiuBellmanTargetConfig,
    LiuCandidateActionMasker,
    LiuEpsilonSchedule,
    LiuCandidateGenerationContext,
    LiuCandidateQNetwork,
    LiuCandidateQNetworkConfig,
    LiuCandidateSetConfig,
    LiuConsensusProtocol,
    LiuReplayBuffer,
    LiuReplayTransition,
    LiuState,
    LiuStateNormalizationConfig,
    LiuStateTensorEncoder,
    SpatialProfileSet,
    ThreatScenario,
    compute_bellman_targets,
    generate_candidate_actions,
    reconstruct_candidate_actions,
    select_greedy,
    select_uniform_candidate,
    LiuDQNCheckpointEnvironment,
    LiuDQNTrainer,
    LiuDQNTrainingConfig,
)


def fixture_state(node_count=3, *, stakes=(1.0, 2.0, 4.0)):
    spatial = SpatialProfileSet.from_coordinates(
        tuple((float(index), float(index + 1)) for index in range(node_count)),
        region_width_km=4.0,
        region_height_km=4.0,
    )
    links = LinkStateMatrix.from_rows(
        tuple(
            tuple(None if sender == receiver else float(10 * (sender + receiver + 1)) for receiver in range(node_count))
            for sender in range(node_count)
        )
    )
    return LiuState(2_000.0, stakes, spatial, tuple(10.0 * (index + 1) for index in range(node_count)), links)


def normalization(node_count=3):
    return LiuStateNormalizationConfig(node_count, 4_000.0, 4.0, 4.0, 4.0, 30.0, 50.0)


def actions():
    return (
        LiuAction(3, 2, (0, 1), LiuConsensusProtocol.PBFT, 1.0, 1.0),
        LiuAction(3, 2, (0, 2), LiuConsensusProtocol.ZYZZYVA, 2.0, 1.5),
        LiuAction(3, 2, (1, 2), LiuConsensusProtocol.LIU_QUORUM, 4.0, 2.0),
    )

class EncodingTests(unittest.TestCase):
    def test_state_shape_ordering_and_hand_normalization(self):
        state = fixture_state()
        encoder = LiuStateTensorEncoder(normalization())
        encoded = encoder.encode(state)
        self.assertEqual(encoded.dimension, 1 + 4 * 3 + 3 * 3)
        self.assertEqual(
            encoded.values,
            (
                0.5,
                0.25, 0.5, 1.0,
                0.0, 0.25, 0.5,
                0.25, 0.5, 0.75,
                1 / 3, 2 / 3, 1.0,
                0.0, 0.4, 0.6,
                0.4, 0.0, 0.8,
                0.6, 0.8, 0.0,
            ),
        )
        self.assertTrue(all(math.isfinite(value) for value in encoded.values))
        self.assertEqual(encoded, encoder.encode(state))
        self.assertEqual(encoded.deterministic_hash(), encoder.encode(state).deterministic_hash())

    def test_state_n100_has_fixed_dimension(self):
        n = 100
        spatial = SpatialProfileSet.from_coordinates(((0.0, 0.0),) * n, region_width_km=1.0, region_height_km=1.0)
        links = LinkStateMatrix.from_rows(tuple(tuple(None if i == j else 10.0 for j in range(n)) for i in range(n)))
        state = LiuState(1.0, (1.0,) * n, spatial, (20.0,) * n, links)
        encoded = LiuStateTensorEncoder(LiuStateNormalizationConfig(n, 1.0, 1.0, 1.0, 1.0, 20.0, 10.0)).encode(state)
        self.assertEqual(encoded.dimension, 10_401)

    def test_configured_bounds_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "transaction size"):
            LiuStateTensorEncoder(LiuStateNormalizationConfig(3, 1_000.0, 4.0, 4.0, 4.0, 30.0, 50.0)).encode(fixture_state())

    def test_action_shape_mask_one_hot_and_normalization(self):
        encoder = LiuActionTensorEncoder(3, max_block_size_mb=4.0, max_block_interval_s=2.0)
        pbft, zyzzyva, quorum = tuple(encoder.encode(action) for action in actions())
        self.assertEqual(pbft.dimension, 8)
        self.assertEqual(pbft.values, (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.25, 0.5))
        self.assertEqual(zyzzyva.values[3:6], (0.0, 1.0, 0.0))
        self.assertEqual(quorum.values[3:6], (0.0, 0.0, 1.0))
        self.assertEqual(quorum.values[-2:], (1.0, 1.0))
        self.assertEqual(pbft, encoder.encode(actions()[0]))

    def test_action_n100_has_105_features(self):
        action = LiuAction(100, 21, tuple(range(21)), LiuConsensusProtocol.PBFT, 4.0, 2.0)
        encoded = LiuActionTensorEncoder(100, max_block_size_mb=4.0, max_block_interval_s=2.0).encode(action)
        self.assertEqual(encoded.dimension, 105)
        self.assertEqual(sum(encoded.values[:100]), 21.0)

    def test_encodings_are_immutable(self):
        encoder = LiuStateTensorEncoder(normalization())
        encoded = encoder.encode(fixture_state())
        with self.assertRaises(FrozenInstanceError):
            encoded.values = ()
        with self.assertRaises(FrozenInstanceError):
            encoder.normalization = normalization()


class CandidateAndNetworkTests(unittest.TestCase):
    def setUp(self):
        self.state = fixture_state()
        self.domain = LiuActionCandidateDomain(3, 2, tuple(LiuConsensusProtocol), (1.0, 2.0, 4.0), (1.0, 1.5, 2.0))
        self.generator = LiuActionCandidateGenerator(
            self.domain,
            ThreatScenario(3, (0,)),
            GridSpatialIntensityModel(self.state.spatial_profiles, 1, 3),
            stake_gini_threshold=0.2,
            geographic_gini_threshold=1.0,
        )

    def test_masking_uses_c1_c2_only_and_retains_diagnostics(self):
        candidate_list = actions()
        result = LiuCandidateActionMasker().screen(self.generator, self.state, candidate_list)
        self.assertEqual(result.generated_count, 3)
        self.assertEqual(result.screened_out_count, 2)
        self.assertEqual(result.evaluated_count, 1)
        self.assertEqual(result.pre_execution_mask, (False, False, True))
        self.assertTrue(all(item.c3_status == "runtime_required" for item in result.screening_results))
        q = torch.tensor((1.0, 100.0, 200.0))
        masked = LiuCandidateActionMasker.apply_to_q_values(q, result.pre_execution_mask)
        self.assertTrue(torch.isneginf(masked[:2]).all())
        self.assertEqual(float(masked[2]), 200.0)

    def test_network_scalar_and_batched_scoring_are_deterministic(self):
        state_encoder = LiuStateTensorEncoder(normalization())
        action_encoder = LiuActionTensorEncoder(3, max_block_size_mb=4.0, max_block_interval_s=2.0)
        state = state_encoder.encode(self.state).as_tensor()
        encoded_actions = torch.stack(tuple(action_encoder.encode(action).as_tensor() for action in actions()))
        config = LiuCandidateQNetworkConfig(state.shape[0], encoded_actions.shape[1], (16, 8), initialization_seed=91)
        global_before = torch.random.get_rng_state().clone()
        first = LiuCandidateQNetwork(config)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        second = LiuCandidateQNetwork(config)
        scores = first.score_candidates(state, encoded_actions)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(first(torch.cat((state, encoded_actions[0]))).shape, torch.Size([]))
        self.assertTrue(torch.equal(scores, second.score_candidates(state, encoded_actions)))

    def test_greedy_and_uniform_selection(self):
        candidates = actions()
        greedy = select_greedy(candidates, torch.tensor((1.0, 20.0, 3.0)), (True, False, True))
        self.assertEqual(greedy.candidate_index, 2)
        first = select_uniform_candidate(candidates, (True, False, True), random.Random(17))
        second = select_uniform_candidate(candidates, (True, False, True), random.Random(17))
        self.assertEqual(first, second)

    def test_generator_remains_lazy_for_combinatorial_space(self):
        n = 100
        spatial = SpatialProfileSet.from_coordinates(((0.0, 0.0),) * n, region_width_km=1.0, region_height_km=1.0)
        links = LinkStateMatrix.from_rows(tuple(tuple(None if i == j else 10.0 for j in range(n)) for i in range(n)))
        state = LiuState(1.0, (1.0,) * n, spatial, (20.0,) * n, links)
        domain = LiuActionCandidateDomain(n, 21, tuple(LiuConsensusProtocol), (1.0,), (1.0,))
        generator = LiuActionCandidateGenerator(
            domain, ThreatScenario(n, ()), GridSpatialIntensityModel(spatial, 1, 1),
            stake_gini_threshold=1.0, geographic_gini_threshold=1.0,
        )
        iterator = generator.iter_candidates(state, random.Random(1), 2)
        self.assertFalse(isinstance(iterator, (tuple, list)))
        self.assertEqual(len(tuple(iterator)), 2)

    def test_candidate_context_exactly_reconstructs_step_local_set(self):
        config = LiuCandidateSetConfig(5)
        global_before = random.getstate()
        first, context = generate_candidate_actions(self.generator, self.state, config, candidate_seed=1837413)
        reconstructed = reconstruct_candidate_actions(context, self.generator, self.state)
        self.assertEqual(first, reconstructed)
        self.assertEqual(context.candidates_per_step, 5)
        self.assertEqual(global_before, random.getstate())


class ReplayAndBellmanTests(unittest.TestCase):
    def transition(self, reward=1.0, *, terminated=False, truncated=False):
        state = fixture_state()
        state_encoding = LiuStateTensorEncoder(normalization()).encode(state)
        action_encoding = LiuActionTensorEncoder(3, max_block_size_mb=4.0, max_block_interval_s=2.0).encode(actions()[0])
        next_candidates = () if terminated else actions()[:2]
        next_mask = () if terminated else (True, False)
        context = None if terminated else LiuCandidateGenerationContext(
            state_encoding.source_state_hash, 17, 2, "domain-hash", "generator-v1"
        )
        return LiuReplayTransition(
            state_encoding,
            action_encoding,
            reward,
            None if terminated else state_encoding,
            terminated,
            truncated,
            next_candidates,
            next_mask,
            context,
        )

    def test_replay_transition_immutable_and_hashable(self):
        transition = self.transition()
        with self.assertRaises(FrozenInstanceError):
            transition.reward = 3.0
        self.assertEqual(transition.deterministic_hash(), self.transition().deterministic_hash())

    def test_replay_sampling_is_deterministic(self):
        buffer = LiuReplayBuffer(3)
        for reward in (1.0, 2.0, 3.0):
            buffer.append(self.transition(reward))
        first = buffer.sample(2, random.Random(41))
        second = buffer.sample(2, random.Random(41))
        self.assertEqual(first, second)
        snapshot = buffer.snapshot()
        buffer.append(self.transition(4.0))
        self.assertEqual(tuple(item.reward for item in snapshot), (1.0, 2.0, 3.0))
        self.assertEqual(tuple(item.reward for item in buffer.snapshot()), (2.0, 3.0, 4.0))

    def test_terminal_target_has_no_bootstrap(self):
        target = compute_bellman_targets(
            torch.tensor((5.0,)), torch.tensor((True,)), torch.tensor((False,)),
            torch.tensor(((100.0, 200.0),)), torch.tensor(((True, True),)), LiuBellmanTargetConfig(0.9),
        )
        self.assertEqual(float(target[0]), 5.0)

    def test_truncation_bootstraps_under_versioned_policy(self):
        args = (
            torch.tensor((1.0,)), torch.tensor((False,)), torch.tensor((True,)),
            torch.tensor(((2.0, 9.0),)), torch.tensor(((True, False),)),
        )
        bootstrap = compute_bellman_targets(*args, LiuBellmanTargetConfig(0.5, True))
        no_bootstrap = compute_bellman_targets(*args, LiuBellmanTargetConfig(0.5, False))
        self.assertEqual(float(bootstrap[0]), 2.0)
        self.assertEqual(float(no_bootstrap[0]), 1.0)


class FixtureStatus(str, Enum):
    FEASIBLE = "feasible"


class TinyTrainingEnvironment:
    def __init__(self, runtime_seed=101):
        self.runtime_seed = runtime_seed
        self.step_index = 0
        self.action_hashes = []
        self.spatial = SpatialProfileSet.from_coordinates(
            ((0.25, 0.25), (1.25, 0.25), (2.25, 0.25), (3.25, 0.25)),
            region_width_km=4.0,
            region_height_km=1.0,
        )
        self.links = LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 20.0 for j in range(4)) for i in range(4))
        )

    def state(self):
        return LiuState(
            1_000.0 + 10.0 * self.step_index,
            (1.0, 1.0, 1.0, 1.0),
            self.spatial,
            (20.0, 20.0, 20.0, 20.0),
            self.links,
        )

    def reset(self):
        self.step_index = 0
        self.action_hashes = []
        state = self.state()
        return SimpleNamespace(observation=state, state_hash=state.deterministic_hash())

    def step(self, action):
        self.action_hashes.append(action.deterministic_hash())
        self.step_index += 1
        protocol_reward = {
            LiuConsensusProtocol.PBFT: 1.0,
            LiuConsensusProtocol.ZYZZYVA: 2.0,
            LiuConsensusProtocol.LIU_QUORUM: 3.0,
        }[action.consensus_protocol]
        reward = protocol_reward + action.block_size_mb * 0.1 + sum(action.validator_ids) * 0.01
        return SimpleNamespace(
            observation=self.state(),
            reward=reward,
            terminated=False,
            truncated=False,
            status=FixtureStatus.FEASIBLE,
        )

    def snapshot(self):
        return {"action_hashes": list(self.action_hashes), "step_index": self.step_index}

    def restore(self, snapshot):
        self.step_index = snapshot["step_index"]
        self.action_hashes = list(snapshot["action_hashes"])
        return self.state()

    def metadata(self):
        return {"runtime_seed": self.runtime_seed, "fixture": "tiny_dqn_training_v1"}


def training_config(**overrides):
    values = dict(
        gamma=0.9,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip_norm=0.25,
        loss="mse",
        replay_capacity=64,
        replay_warmup_steps=1,
        batch_size=1,
        update_every_steps=1,
        target_sync_steps=2,
        candidates_per_step=4,
        max_steps_per_episode=6,
        epsilon_schedule=LiuEpsilonSchedule(0.8, 0.2, 10),
        exploration_seed=11,
        replay_seed=13,
        candidate_seed=17,
        torch_seed=19,
    )
    values.update(overrides)
    return LiuDQNTrainingConfig(**values)


def make_trainer(config=None, *, exploration_seed=None, threat=()):
    config = config or training_config()
    if exploration_seed is not None:
        config = training_config(exploration_seed=exploration_seed)
    environment = TinyTrainingEnvironment()
    bridge = LiuDQNCheckpointEnvironment(
        environment,
        environment.snapshot,
        environment.restore,
        environment.metadata,
    )
    domain = LiuActionCandidateDomain(
        4,
        2,
        tuple(LiuConsensusProtocol),
        (1.0, 2.0),
        (1.0, 2.0),
    )
    generator = LiuActionCandidateGenerator(
        domain,
        ThreatScenario(4, threat),
        GridSpatialIntensityModel(environment.spatial, 1, 4),
        stake_gini_threshold=1.0,
        geographic_gini_threshold=1.0,
    )
    state_encoder = LiuStateTensorEncoder(
        LiuStateNormalizationConfig(4, 2_000.0, 1.0, 4.0, 1.0, 20.0, 20.0)
    )
    action_encoder = LiuActionTensorEncoder(4, 2.0, 2.0)
    network_config = LiuCandidateQNetworkConfig(
        state_encoder.dimension,
        action_encoder.dimension,
        (12, 8),
        initialization_seed=23,
    )
    return LiuDQNTrainer(config, bridge, generator, state_encoder, action_encoder, network_config)


def parameters(network):
    return tuple(parameter.detach().clone() for parameter in network.parameters())


class ScheduleAndInitializationTests(unittest.TestCase):
    def test_epsilon_schedule_exact(self):
        schedule = LiuEpsilonSchedule(1.0, 0.1, 10)
        self.assertEqual(schedule.value(0), 1.0)
        self.assertAlmostEqual(schedule.value(5), 0.55)
        self.assertEqual(schedule.value(10), 0.1)
        self.assertEqual(schedule.value(100), 0.1)

    def test_target_initially_identical_and_has_no_gradients(self):
        trainer = make_trainer()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters(trainer.online_network), parameters(trainer.target_network))))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.target_network.parameters()))

    def test_epsilon_one_is_seeded_exploration_and_zero_is_greedy(self):
        exploratory = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(1.0, 1.0, 1)))
        greedy = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(0.0, 0.0, 1)))
        self.assertEqual(exploratory.train_step().selection_mode, "exploratory")
        self.assertEqual(greedy.train_step().selection_mode, "greedy")
        repeated = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(1.0, 1.0, 1)))
        self.assertEqual(exploratory.step_records[0].selected_action_hash, repeated.train_step().selected_action_hash)


class UpdateLifecycleTests(unittest.TestCase):
    def test_nonsemantic_wall_clock_and_screening_diagnostics_are_exposed(self):
        trainer = make_trainer()
        trainer.train_step()
        profile = trainer.wall_clock_profiles[-1]
        screening = trainer.selection_diagnostics[-1]
        self.assertGreaterEqual(profile.environment_execution_s, 0.0)
        self.assertGreaterEqual(profile.q_scoring_s, 0.0)
        self.assertEqual(screening.feasible_candidate_count, trainer.config.candidates_per_step)
        self.assertEqual(screening.c1_failure_count, 0)
        self.assertEqual(screening.c2_failure_count, 0)

    def test_optimizer_changes_online_but_not_target_before_sync(self):
        trainer = make_trainer(training_config(target_sync_steps=10))
        online_before = parameters(trainer.online_network)
        target_before = parameters(trainer.target_network)
        record = trainer.train_step()
        self.assertIsNotNone(record.loss)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(online_before, parameters(trainer.online_network))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(target_before, parameters(trainer.target_network))))

    def test_replay_warmup_and_update_cadence(self):
        trainer = make_trainer(training_config(replay_warmup_steps=3, update_every_steps=2))
        for _ in range(3):
            trainer.train_step()
        self.assertEqual(trainer.training_state.optimizer_step, 0)
        trainer.train_step()
        self.assertEqual(trainer.training_state.optimizer_step, 1)

    def test_hard_target_sync_is_exact(self):
        trainer = make_trainer(training_config(target_sync_steps=2))
        trainer.train_step()
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(parameters(trainer.online_network), parameters(trainer.target_network))))
        second = trainer.train_step()
        self.assertTrue(second.target_synchronized)
        self.assertEqual(trainer.training_state.target_sync_count, 1)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters(trainer.online_network), parameters(trainer.target_network))))

    def test_gradient_clipping_loss_and_q_are_finite(self):
        trainer = make_trainer(training_config(gradient_clip_norm=0.01))
        record = trainer.train_step()
        update = trainer.update_results[-1]
        self.assertTrue(np.isfinite(record.loss))
        self.assertTrue(np.isfinite(record.selected_q))
        self.assertLessEqual(update.gradient_norm_after_clip, 0.010001)
        self.assertGreaterEqual(update.gradient_norm_before_clip, update.gradient_norm_after_clip)

    def test_complete_episode_and_counters(self):
        trainer = make_trainer()
        result = trainer.train_episode()
        self.assertEqual(result.episode_length, 6)
        self.assertEqual(result.termination_reason, "trainer_episode_horizon")
        self.assertEqual(result.feasible_steps, 6)
        self.assertEqual(trainer.training_state.environment_step, 6)
        self.assertEqual(trainer.training_state.episode_index, 1)
        self.assertEqual(trainer.training_state.optimizer_step, 6)
        self.assertEqual(trainer.training_state.target_sync_count, 3)
        self.assertTrue(np.isfinite(result.mean_loss))


class DeterminismAndCheckpointTests(unittest.TestCase):
    def test_same_seeds_same_trace_and_different_exploration_seed_can_change_path(self):
        first = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(1.0, 1.0, 1)))
        second = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(1.0, 1.0, 1)))
        different = make_trainer(training_config(epsilon_schedule=LiuEpsilonSchedule(1.0, 1.0, 1), exploration_seed=99))
        first_result = first.train_episode()
        second_result = second.train_episode()
        different_result = different.train_episode()
        self.assertEqual(first_result, second_result)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters(first.online_network), parameters(second.online_network))))
        self.assertNotEqual(
            tuple(record.selected_action_hash for record in first_result.step_records),
            tuple(record.selected_action_hash for record in different_result.step_records),
        )

    def test_checkpoint_round_trip_and_exact_continuation(self):
        uninterrupted = make_trainer(training_config(max_steps_per_episode=10))
        resumed = make_trainer(training_config(max_steps_per_episode=10))
        for _ in range(3):
            uninterrupted.train_step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            uninterrupted.save_checkpoint(path)
            identical_path = Path(directory) / "checkpoint-identical.json"
            uninterrupted.save_checkpoint(identical_path)
            self.assertEqual(path.read_bytes(), identical_path.read_bytes())
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["checkpoint_schema_version"], "liu_dqn_checkpoint_v1")
            resumed.load_checkpoint(path)
            before = len(uninterrupted.step_records)
            for _ in range(3):
                uninterrupted.train_step()
                resumed.train_step()
        self.assertEqual(uninterrupted.step_records[before:], resumed.step_records[before:])
        self.assertEqual(uninterrupted.training_state, resumed.training_state)
        self.assertEqual(uninterrupted.replay_buffer.snapshot(), resumed.replay_buffer.snapshot())
        self.assertEqual(uninterrupted.update_results, resumed.update_results)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters(uninterrupted.online_network), parameters(resumed.online_network))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters(uninterrupted.target_network), parameters(resumed.target_network))))

    def test_no_global_python_numpy_or_torch_rng_consumption(self):
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        external_fsmc_rng = random.Random(8675309)
        fsmc_before = external_fsmc_rng.getstate()
        trainer = make_trainer()
        trainer.train_episode()
        self.assertEqual(python_before, random.getstate())
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state())))
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertEqual(fsmc_before, external_fsmc_rng.getstate())


class FailurePolicyTests(unittest.TestCase):
    def test_no_feasible_candidate_is_explicit_and_does_not_invent_action(self):
        config = training_config(candidates_per_step=2)
        environment = TinyTrainingEnvironment()
        bridge = LiuDQNCheckpointEnvironment(environment, environment.snapshot, environment.restore, environment.metadata)
        domain = LiuActionCandidateDomain(4, 2, (LiuConsensusProtocol.LIU_QUORUM,), (1.0,), (1.0,))
        generator = LiuActionCandidateGenerator(
            domain,
            ThreatScenario(4, (0, 1, 2, 3)),
            GridSpatialIntensityModel(environment.spatial, 1, 4),
            stake_gini_threshold=1.0,
            geographic_gini_threshold=1.0,
        )
        state_encoder = LiuStateTensorEncoder(LiuStateNormalizationConfig(4, 2_000.0, 1.0, 4.0, 1.0, 20.0, 20.0))
        action_encoder = LiuActionTensorEncoder(4, 1.0, 1.0)
        trainer = LiuDQNTrainer(
            config,
            bridge,
            generator,
            state_encoder,
            action_encoder,
            LiuCandidateQNetworkConfig(state_encoder.dimension, action_encoder.dimension, (8,), initialization_seed=1),
        )
        result = trainer.train_episode()
        self.assertEqual(result.termination_reason, "liu_dqn_no_feasible_candidate_terminate_episode_v1")
        self.assertEqual(result.failed_steps, 1)
        self.assertEqual(result.episode_length, 0)
        self.assertEqual(len(trainer.replay_buffer), 0)
        self.assertEqual(len(trainer.wall_clock_profiles), 1)
        self.assertEqual(trainer.wall_clock_profiles[0].environment_execution_s, 0.0)


if __name__ == "__main__":
    unittest.main()
