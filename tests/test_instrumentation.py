import random
import sys
import unittest
from collections import deque
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))
from Utils.Instrumentation import BlockProposalRecord, LocalConsensusDecisionRecord, TransactionCreationRecord
from Utils.InstrumentationMetrics import InstrumentationMetrics, LogicalBlockKey


from Chain.Block import Block
from Chain.Node import Node
from Chain.TransactionFactory import Transaction, TransactionFactory
from Engine.EventQueue import Queue
from Parameters import Parameters
from Utils.Instrumentation import InstrumentationCollector, TransactionCreationRecord

class TransactionTimingTests(unittest.TestCase):
    def setUp(self):
        InstrumentationCollector.reset()

    def test_timestamp_aliases_availability_without_changing_creation_time(self):
        transaction = Transaction(creator=2, id=7, timestamp=10.0, size=0.5)

        self.assertEqual(transaction.original_creation_time, 10.0)
        self.assertEqual(transaction.available_at, 10.0)
        self.assertEqual(transaction.timestamp, 10.0)

        transaction.timestamp = 12.5

        self.assertEqual(transaction.available_at, 12.5)
        self.assertEqual(transaction.timestamp, 12.5)
        self.assertEqual(transaction.original_creation_time, 10.0)
        with self.assertRaises(AttributeError):
            transaction.original_creation_time = 99.0

    def test_local_propagation_preserves_original_time_and_legacy_availability(self):
        nodes = [SimpleNamespace(id=index, pool=deque()) for index in range(3)]
        TransactionFactory.nodes = nodes
        Parameters.application = {"transaction_model": "local"}
        transaction = Transaction(creator=0, id=3, timestamp=10.0, size=0.2)

        with patch("Chain.TransactionFactory.Network.calculate_message_propagation_delay", side_effect=[1.0, 2.0]):
            TransactionFactory.transaction_prop(transaction)

        self.assertEqual(nodes[0].pool, deque())
        self.assertEqual(nodes[1].pool[0].available_at, 11.0)
        self.assertEqual(nodes[2].pool[0].available_at, 13.0)
        self.assertEqual(nodes[1].pool[0].original_creation_time, 10.0)
        self.assertEqual(nodes[2].pool[0].original_creation_time, 10.0)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        InstrumentationCollector.reset()

    def test_records_are_immutable_and_do_not_consume_randomness(self):
        state_before = random.getstate()
        record = TransactionCreationRecord(1, 2, 3.0, 0.1)
        InstrumentationCollector.record_transaction_creation(record)

        self.assertEqual(random.getstate(), state_before)
        self.assertEqual(InstrumentationCollector.transaction_creations, [record])
        with self.assertRaises(FrozenInstanceError):
            record.transaction_size = 9.0

    def test_reset_starts_a_fresh_run_scope(self):
        record = TransactionCreationRecord(1, 2, 3.0, 0.1)
        InstrumentationCollector.record_transaction_creation(record)

        InstrumentationCollector.reset()

        self.assertEqual(InstrumentationCollector.transaction_creations, [])
        self.assertEqual(InstrumentationCollector.block_proposals, [])
        self.assertEqual(InstrumentationCollector.local_consensus_decisions, [])
        self.assertEqual(InstrumentationCollector.block_observations, [])
        self.assertIsNone(InstrumentationCollector.run_profile_context)

    def test_block_observation_records_cause_without_mutating_block_metadata(self):
        Parameters.application = {"transaction_model": "local"}
        node = Node(4, Queue())
        block = Block(depth=1, id=8, time_created=2.0, miner=4, consensus="PBFT")
        block.extra_data = {"round": 3, "configuration_depth": 0}
        original_metadata = dict(block.extra_data)

        node.add_block(block, time=5.0, cause="finalized_block_announcement")

        self.assertEqual(block.time_added, 5.0)
        self.assertEqual(block.extra_data, original_metadata)
        observation = InstrumentationCollector.block_observations[0]
        self.assertEqual(observation.observation_time, 5.0)
        self.assertEqual(observation.cause, "finalized_block_announcement")

    def test_sync_observation_uses_precomputed_block_time(self):
        Parameters.application = {"transaction_model": "local"}
        node = Node(4, Queue())
        block = Block(depth=1, id=8, time_created=2.0, miner=4, consensus="PBFT")
        block.time_added = 7.5
        block.extra_data = {"round": 3, "configuration_depth": 0}

        node.add_block(block, time=-1, update_time_added=False, cause="synchronization")

        self.assertEqual(InstrumentationCollector.block_observations[0].observation_time, 7.5)


def proposal(block_id=10, depth=1, transaction_ids=(1, 2), proposal_time=5.0):
    return BlockProposalRecord(
        block_id=block_id,
        block_depth=depth,
        proposer=0,
        consensus_protocol="PBFT",
        round=depth,
        configuration_depth=0,
        proposal_time=proposal_time,
        block_size=2.2,
        transaction_count=len(transaction_ids),
        transaction_ids=transaction_ids,
        configured_block_size=2.0,
        configured_block_time=2.0,
    )


def decision(block_id=10, depth=1, node_id=0, decision_time=7.0):
    return LocalConsensusDecisionRecord(
        block_id=block_id,
        node_id=node_id,
        decision_time=decision_time,
        consensus_protocol="PBFT",
        round=depth,
        configuration_depth=0,
        decision_path="commit_quorum",
        quorum_size=2,
        validator_count=3,
    )


class InstrumentationMetricsTests(unittest.TestCase):
    def calculate(self, creations, proposals, decisions, start=None, end=None):
        return InstrumentationMetrics.calculate(creations, proposals, decisions, [], start, end)

    def test_logical_block_grouping_and_first_quorum_evidence(self):
        creations = [TransactionCreationRecord(1, 0, 1.0, 1.0), TransactionCreationRecord(2, 0, 2.0, 1.0)]
        decisions = [decision(node_id=0, decision_time=8.0), decision(node_id=1, decision_time=7.0), decision(node_id=1, decision_time=7.0)]

        result = self.calculate(creations, [proposal()], decisions, 5.0, 8.0)

        block = result.finalized_blocks[0]
        self.assertEqual(block.logical_block_key, LogicalBlockKey(0, "PBFT", 1, 10))
        self.assertEqual(block.first_consensus_decision_time, 7.0)
        self.assertEqual(block.first_quorum_evidence.node_id, 1)
        self.assertEqual(block.quorum_adoption_time, 1.0)

    def test_consensus_latency_and_distribution(self):
        creations = [TransactionCreationRecord(1, 0, 1.0, 1.0)]
        result = self.calculate(creations, [proposal(transaction_ids=(1,), proposal_time=5.0)], [decision(decision_time=7.5)], 5.0, 7.5)

        self.assertEqual(result.finalized_blocks[0].consensus_latency, 2.5)
        summary = result.consensus_latency_by_protocol["PBFT"]
        self.assertEqual((summary.mean, summary.median, summary.minimum, summary.maximum, summary.standard_deviation), (2.5, 2.5, 2.5, 2.5, 0.0))

    def test_first_evidence_can_precede_a_different_valid_decision_path(self):
        creations = [TransactionCreationRecord(1, 0, 1.0, 1.0)]
        fast = decision(node_id=0, decision_time=8.0)
        fast = LocalConsensusDecisionRecord(
            fast.block_id, fast.node_id, fast.decision_time, fast.consensus_protocol, fast.round, fast.configuration_depth, "bigfoot_fast", 3, 3
        )
        slow = LocalConsensusDecisionRecord(
            fast.block_id, 1, 7.0, fast.consensus_protocol, fast.round, fast.configuration_depth, "bigfoot_slow", 2, 3
        )

        result = self.calculate(creations, [proposal(transaction_ids=(1,))], [fast, slow], 5.0, 8.0)

        block = result.finalized_blocks[0]
        self.assertEqual(block.first_quorum_evidence.decision_path, "bigfoot_slow")
        self.assertEqual(block.first_quorum_evidence.quorum_size, 2)
        self.assertEqual(block.quorum_adoption_time, 1.0)

    def test_transaction_quorum_ttf_joins_creation_membership_and_decision(self):
        creations = [TransactionCreationRecord(1, 0, 1.0, 1.0), TransactionCreationRecord(2, 0, 3.0, 1.0)]
        result = self.calculate(creations, [proposal()], [decision(decision_time=7.0)], 5.0, 7.0)

        by_id = {record.transaction_id: record.quorum_ttf for record in result.transaction_quorum_ttf}
        self.assertEqual(by_id, {1: 6.0, 2: 4.0})

    def test_system_throughput_uses_unique_transactions_and_first_to_last_window(self):
        creations = [TransactionCreationRecord(i, 0, float(i), 1.0) for i in range(1, 5)]
        proposals = [proposal(10, 1, (1, 2), 5.0), proposal(11, 2, (2, 3), 9.0), proposal(12, 3, (4,), 13.0)]
        decisions = [decision(10, 1, 0, 7.0), decision(11, 2, 0, 11.0), decision(12, 3, 0, 15.0)]

        result = self.calculate(creations, proposals, decisions)

        throughput = result.simulation_throughput
        self.assertEqual((throughput.measurement_start, throughput.measurement_end, throughput.measurement_duration), (7.0, 15.0, 8.0))
        self.assertEqual(throughput.unique_transaction_count, 3)
        self.assertEqual(throughput.transactions_per_second, 3 / 8)
        self.assertEqual(result.duplicate_transaction_ids, (2,))
        self.assertEqual(len([item for item in result.transaction_quorum_ttf if item.transaction_id == 2]), 1)
        self.assertEqual(next(item for item in result.transaction_quorum_ttf if item.transaction_id == 2).first_consensus_decision_time, 7.0)

    def test_analytical_capacity_is_explicit_empirical_estimate(self):
        creations = [TransactionCreationRecord(1, 0, 1.0, 0.5), TransactionCreationRecord(2, 0, 2.0, 1.5)]
        proposals = [proposal(10, 1, (1,), 5.0), proposal(11, 2, (2,), 9.0)]
        decisions = [decision(10, 1, 0, 7.0), decision(11, 2, 0, 11.0)]

        result = self.calculate(creations, proposals, decisions)

        estimate = result.analytical_capacity_estimate
        self.assertEqual(estimate.empirical_mean_transaction_size, 1.5)
        self.assertAlmostEqual(estimate.per_protocol["PBFT"].mean, 2.0 / 1.5 / 2.0)

    def test_conflicting_proposals_expose_non_fork_proof_key(self):
        original = proposal()
        collision = BlockProposalRecord(
            block_id=original.block_id,
            block_depth=99,
            proposer=2,
            consensus_protocol=original.consensus_protocol,
            round=original.round,
            configuration_depth=original.configuration_depth,
            proposal_time=6.0,
            block_size=1.0,
            transaction_count=0,
            transaction_ids=(),
            configured_block_size=2.0,
            configured_block_time=2.0,
        )

        with self.assertRaisesRegex(ValueError, "not fork-proof"):
            self.calculate([], [original, collision], [])

    def test_missing_creation_record_is_not_inferred(self):
        with self.assertRaisesRegex(ValueError, "Missing creation record"):
            self.calculate([], [proposal(transaction_ids=(9,))], [decision()], 5.0, 7.0)


if __name__ == "__main__":
    unittest.main()
