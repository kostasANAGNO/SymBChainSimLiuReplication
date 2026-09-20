import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))
from Chain.Consensus.BigFoot.BigFoot_state import BigFoot
from Chain.Consensus.PBFT import PBFT_transition
from Chain.Consensus.PBFT.PBFT_state import PBFT
from Chain.Consensus.Tendermint.TM_state import Tendermint
from Chain.Network import Network
from Chain.Reconfiguration.ConfigurationBlock import ConfigurationBlock
from Engine.Event import Event
from Utils.Instrumentation import InstrumentationCollector
from Utils.Metrics import Metrics


from Chain.Block import Block
from Chain.Consensus import HighLevelSync
from Chain.Node import Node
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Chain.ValidatorSet import ValidatorSet
from Engine.EventQueue import Queue
from Parameters import Parameters
from Utils.ComputationalDelay import ComputationalDelay
from Utils.DecentralizationMetrics import DecentralizationMetrics, canonical_pairwise_gini
from Utils.Instrumentation import InstrumentationCollector, RunProfileContext

class NodeProfileTests(unittest.TestCase):
    def test_missing_configuration_is_explicitly_unavailable_and_immutable(self):
        profiles = NodeProfileSet.from_config(3)

        self.assertEqual(tuple(profile.node_id for profile in profiles.profiles), (0, 1, 2))
        self.assertTrue(all(profile.stake_tokens is None for profile in profiles.profiles))
        self.assertTrue(all(profile.computational_capability_ghz is None for profile in profiles.profiles))
        self.assertFalse(profiles.capability_scaling_enabled)
        with self.assertRaises(FrozenInstanceError):
            profiles.profile_for(0).stake_tokens = 5

    def test_explicit_ordered_configuration_maps_by_node_id(self):
        profiles = NodeProfileSet.from_config(
            3,
            {"stake_tokens": [3, 1, 2], "computational_capability_ghz": [10, 20, 30]},
        )

        self.assertEqual(profiles.profile_for(1), NodeProfile(1, 1.0, 20.0))
        self.assertTrue(profiles.capability_scaling_enabled)

    def test_profile_lists_must_be_complete_and_capabilities_positive(self):
        with self.assertRaisesRegex(ValueError, "exactly Nn=3"):
            NodeProfileSet.from_config(3, {"stake_tokens": [1, 2]})
        with self.assertRaisesRegex(ValueError, "positive"):
            NodeProfileSet.from_config(1, {"computational_capability_ghz": [0]})

    def test_node_exposes_read_only_profile_values(self):
        profile = NodeProfile(0, 7.0, 30.0)
        node = Node(0, Queue(), ValidatorSet((0,)), profile)

        self.assertIs(node.profile, profile)
        self.assertEqual(node.stake_tokens, 7.0)
        self.assertEqual(node.computational_capability_ghz, 30.0)
        with self.assertRaises(AttributeError):
            node.stake_tokens = 10.0


class ComputationalDelayTests(unittest.TestCase):
    def test_inverse_reference_scaling(self):
        self.assertEqual(ComputationalDelay.scale(0.01, None), 0.01)
        self.assertEqual(ComputationalDelay.scale(0.01, 10), 0.02)
        self.assertEqual(ComputationalDelay.scale(0.01, 20), 0.01)
        self.assertAlmostEqual(ComputationalDelay.scale(0.01, 30), 0.01 * 2 / 3)

    def test_stake_does_not_enter_delay_scaling(self):
        low_stake = Node(0, Queue(), ValidatorSet((0,)), NodeProfile(0, 1.0, 20.0))
        high_stake = Node(0, Queue(), ValidatorSet((0,)), NodeProfile(0, 50.0, 20.0))

        self.assertEqual(ComputationalDelay.for_node(0.25, low_stake), ComputationalDelay.for_node(0.25, high_stake))

    def test_sync_scales_only_block_validation_portion(self):
        Parameters.application = {"transaction_model": "local"}
        Parameters.execution = {"block_val_delay": 0.01, "sync_message_request_delay": 0.1}
        Parameters.simulation = {"event_id": 0, "events": {}}
        validators = ValidatorSet((0, 1))
        slow = Node(0, Queue(), validators, NodeProfile(0, None, 10.0))
        source = Node(1, Queue(), validators, NodeProfile(1, None, 20.0))
        genesis = Block(depth=0, id=1)
        slow.blockchain = [genesis]
        source.blockchain = [genesis.copy(), Block(depth=1, id=2, size=1.0)]

        with patch("Chain.Consensus.HighLevelSync.Network.calculate_message_propagation_delay", return_value=0.5):
            HighLevelSync.create_local_sync_event(slow, source, 0.0)

        # Initial request (0.1) + network (0.5) + scaled validation (0.02)
        # + per-block request delay (0.1). Network and request costs are intact.
        self.assertAlmostEqual(slow.queue.pop_next_event().time, 0.72)


class DecentralizationMetricTests(unittest.TestCase):
    def test_canonical_pairwise_gini_known_populations(self):
        self.assertEqual(canonical_pairwise_gini([1, 1, 1]), 0.0)
        self.assertAlmostEqual(canonical_pairwise_gini([0, 0, 3]), 2 / 3)
        self.assertAlmostEqual(canonical_pairwise_gini([1, 2, 3]), 2 / 9)

    def test_validator_population_drives_producer_and_stake_gini(self):
        profiles = NodeProfileSet.from_config(
            4,
            {"stake_tokens": [1, 1, 4, 50], "computational_capability_ghz": [10, 20, 30, 25]},
        )
        result = DecentralizationMetrics.calculate(profiles, ValidatorSet((0, 1, 2)), [0, 0, 1])

        self.assertEqual(result.producer_gini.population_node_ids, (0, 1, 2))
        self.assertAlmostEqual(result.producer_gini.value, 4 / 9)
        self.assertAlmostEqual(result.stake_gini.value, 1 / 3)
        self.assertIsNone(result.geographic_gini.value)
        self.assertIn("spatial intensity", result.geographic_gini.unavailable_reason)
        self.assertEqual(result.validator_capability_summary.mean_ghz, 20.0)
        self.assertEqual(result.all_node_capability_summary.configured_count, 4)

    def test_missing_stake_and_capability_are_not_imputed(self):
        result = DecentralizationMetrics.calculate(NodeProfileSet.from_config(3), ValidatorSet((0, 1)), [0])

        self.assertIsNone(result.stake_gini.value)
        self.assertIn("unavailable", result.stake_gini.unavailable_reason)
        self.assertIsNone(result.validator_capability_summary.mean_ghz)


class RunContextTests(unittest.TestCase):
    def setUp(self):
        InstrumentationCollector.reset()

    def test_profile_context_exports_separately_from_raw_records(self):
        profiles = NodeProfileSet.from_config(2, {"stake_tokens": [1, 2], "computational_capability_ghz": [10, 20]})
        context = RunProfileContext(
            validator_ids=(0,),
            node_profiles=profiles.profiles,
            computational_capability_reference_ghz=20.0,
            capability_scaling_enabled=True,
            capability_scaling_formula=ComputationalDelay.FORMULA,
            capability_scaling_scope=("consensus_message_validation",),
        )
        InstrumentationCollector.configure_run_context(context)

        exported = InstrumentationCollector.export_run_context()
        self.assertEqual(exported["validator_ids"], (0,))
        self.assertEqual(exported["node_profiles"][1]["computational_capability_ghz"], 20.0)
        self.assertEqual(InstrumentationCollector.transaction_creations, [])


class ValidatorSetTests(unittest.TestCase):
    def test_omitted_count_means_all_nodes_without_rng(self):
        validator_set = ValidatorSet.first_nodes(4)

        self.assertEqual(validator_set.ids, (0, 1, 2, 3))
        self.assertEqual(validator_set.count, 4)

    def test_static_count_selects_first_nodes_and_is_immutable(self):
        validator_set = ValidatorSet.first_nodes(100, 21)

        self.assertEqual(validator_set.ids, tuple(range(21)))
        self.assertEqual(validator_set.proposer_for(22), 1)
        with self.assertRaises(FrozenInstanceError):
            validator_set.ids = (4,)

    def test_invalid_validator_count_is_rejected(self):
        for count in (0, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                ValidatorSet.first_nodes(3, count)

    def test_current_bft_arithmetic_uses_k_without_correcting_bound(self):
        Parameters.application = {"Nn": 100, "validator_count": 21}

        Parameters.configure_validator_set()
        Parameters.calculate_fault_tolerance()

        self.assertEqual(Parameters.validator_set.count, 21)
        self.assertEqual(Parameters.application["f"], 7)
        self.assertEqual(Parameters.application["required_messages"], 15)


class PassiveObserverTests(unittest.TestCase):
    def setUp(self):
        InstrumentationCollector.reset()
        Parameters.application = {"Nn": 3, "f": 0, "required_messages": 1, "transaction_model": "global"}
        Parameters.execution = {
            "proposer_selection": "round_robin",
            "msg_val_delay": 0.001,
            "block_val_delay": 0.01,
            "sync_message_request_delay": 0.1,
        }
        Parameters.simulation = {"event_id": 0, "events": {}, "debugging_mode": False}
        Parameters.data = {"Bsize": 1.0, "block_time": 0.2}
        Parameters.PBFT = {"timeout": 10}
        Parameters.Tendermint = {"timeout": 10}
        Parameters.BigFoot = {"timeout": 10, "fast_path_timeout": 5}
        Parameters.CPs = {PBFT.NAME: PBFT, Tendermint.NAME: Tendermint, BigFoot.NAME: BigFoot}

    def make_observer(self, protocol):
        queue = Queue()
        node = Node(2, queue, ValidatorSet((0, 1)))
        genesis = Block(depth=0, id=10)
        genesis.extra_data["round"] = -1
        node.blockchain.append(genesis)
        configuration = ConfigurationBlock(depth=0)
        configuration.configuration = {"CP": protocol.NAME, "block_size": 1.0, "block_time": 0.2}
        node.reconfiguration_state.confchain.append(configuration)
        node.cp = protocol(node)
        node.cp.init(0.0, 0)
        return node, queue

    def test_observers_keep_protocol_state_without_consensus_events(self):
        for protocol in (PBFT, Tendermint, BigFoot):
            with self.subTest(protocol=protocol.NAME):
                node, queue = self.make_observer(protocol)
                self.assertFalse(node.is_validator)
                self.assertEqual(node.cp.NAME, protocol.NAME)
                self.assertEqual(node.cp.rounds.round, 0)
                self.assertEqual(queue.size(), 0)

    def test_observer_rejects_consensus_event(self):
        node, _ = self.make_observer(PBFT)
        event = Event(PBFT.handle_event, node, 1.0, {"type": "timeout", "round": 0, "CP": "PBFT"})

        self.assertEqual(PBFT.handle_event(event), "invalid")

    def test_observer_adds_finalized_block_without_local_decision(self):
        observer, queue = self.make_observer(PBFT)
        validator = Node(0, Queue(), observer.active_validator_set)
        validator.cp = PBFT(validator)
        block = Block(depth=1, id=11, previous=10, miner=0, consensus="PBFT")
        block.extra_data = {"round": 0, "configuration_depth": 0}
        event = Event(PBFT.handle_event, validator, 1.0, {"type": "new_block", "block": block, "round": 0, "CP": "PBFT"})
        event.actor = observer

        self.assertEqual(PBFT_transition.new_block(observer.cp, event), "new_state")
        self.assertEqual(observer.blockchain_length(), 1)
        self.assertEqual(queue.size(), 0)
        self.assertEqual(InstrumentationCollector.local_consensus_decisions, [])
        self.assertEqual(InstrumentationCollector.block_observations[0].cause, "finalized_block_announcement")

    def test_observer_sync_rejoins_as_passive_protocol_state(self):
        observer, queue = self.make_observer(PBFT)
        validator = Node(0, Queue(), observer.active_validator_set)
        validator.cp = PBFT(validator)
        validator.blockchain = [observer.blockchain[0].copy()]
        block = Block(depth=1, id=11, previous=10, miner=0, consensus="PBFT")
        block.extra_data = {"round": 0, "configuration_depth": 0}
        validator.blockchain.append(block)

        with patch("Chain.Consensus.HighLevelSync.Network.calculate_message_propagation_delay", return_value=0.0):
            HighLevelSync.create_local_sync_event(observer, validator, 0.0)
        event = queue.pop_next_event()

        self.assertEqual(HighLevelSync.handle_local_sync_event(event), "successfully_synced")
        self.assertEqual(observer.blockchain_length(), 1)
        self.assertEqual(observer.cp.NAME, "PBFT")
        self.assertFalse(observer.is_validator)
        self.assertEqual(queue.size(), 0)
        self.assertEqual(InstrumentationCollector.local_consensus_decisions, [])

    def test_observer_applies_received_protocol_configuration_without_timeout(self):
        observer, queue = self.make_observer(PBFT)
        validator = Node(0, Queue(), observer.active_validator_set)
        validator.cp = PBFT(validator)
        configuration = ConfigurationBlock(depth=1, id=20, previous=0)
        configuration.configuration = {"CP": "Tendermint", "block_size": 2.0, "block_time": 0.5}
        event = Event(
            observer.reconfiguration_state.handle_event,
            validator,
            2.0,
            {"type": "prop_conf_block", "block": configuration},
        )
        event.actor = observer

        with patch("Chain.Reconfiguration.ReconfigurationState.Scheduler.schedule_broadcast_message"):
            result = observer.reconfiguration_state.handle_receive_configuration_block(event)

        self.assertEqual(result, "new_state")
        self.assertEqual(observer.cp.NAME, "Tendermint")
        self.assertEqual(observer.reconfiguration_state.current_configuration_depth, 1)
        self.assertEqual(queue.size(), 0)


class ScopedNetworkTests(unittest.TestCase):
    def setUp(self):
        Parameters.network = {"gossip": True}

    def test_separated_consensus_delivery_targets_only_validators(self):
        validator_set = ValidatorSet((0, 1))
        nodes = [Node(index, Queue(), validator_set) for index in range(4)]
        Network.nodes = nodes
        Network.received = {node: set() for node in nodes}
        event = Event(lambda _: "handled", nodes[0], 0.0, {"type": "prepare"})
        event.recipient_scope = "validators"

        with patch.object(Network, "_message") as send:
            Network.send_message(nodes[0], event)

        self.assertEqual([call.args[1].id for call in send.call_args_list], [1])
        self.assertEqual(event.payload, {"type": "prepare"})

    def test_all_validator_scope_uses_legacy_gossip_path(self):
        validator_set = ValidatorSet((0, 1, 2))
        nodes = [Node(index, Queue(), validator_set) for index in range(3)]
        nodes[0].neighbours = [nodes[2], nodes[1]]
        Network.nodes = nodes
        Network.received = {node: set() for node in nodes}
        event = Event(lambda _: "handled", nodes[0], 0.0, {"type": "prepare"})
        event.recipient_scope = "validators"

        with patch.object(Network, "_message") as send:
            Network.send_message(nodes[0], event)

        self.assertEqual([call.args[1].id for call in send.call_args_list], [2, 1])
        self.assertEqual(event.recipient_scope, "all")


class ValidatorScopedMetricsTests(unittest.TestCase):
    def test_confirmation_ignores_observer_lag(self):
        validator_set = ValidatorSet((0, 1))
        nodes = [Node(index, Queue(), validator_set) for index in range(3)]
        for node, block_count in zip(nodes, (3, 3, 1)):
            node.blockchain = [Block(depth=depth) for depth in range(block_count + 1)]
        simulation = type("SimulationStub", (), {"nodes": nodes})()

        self.assertEqual(Metrics.confirmed_blocks(simulation), 3)

    def test_block_production_decentralization_uses_validator_population(self):
        validator_set = ValidatorSet((0, 1))
        nodes = [Node(index, Queue(), validator_set) for index in range(3)]
        simulation = type("SimulationStub", (), {"nodes": nodes})()
        blocks = [Block(depth=1, miner=0), Block(depth=2, miner=0)]

        self.assertAlmostEqual(Metrics.measure_decentralisation_nodes(simulation, blocks), 1 / 3)


if __name__ == "__main__":
    unittest.main()
