import copy
from collections import deque
from dataclasses import FrozenInstanceError
import random
import sys
import unittest
from pathlib import Path


SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Chain.Consensus.LiuRuntime.Common.ActionApplicator import LiuRuntimeActionApplicator
from Chain.Consensus.LiuRuntime.Common.ProtocolFactory import (
    LiuRuntimeProtocolFactory,
    LiuRuntimeProtocolSettings,
)
from Chain.Consensus.LiuRuntime.Common.Identity import parent_digest
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase
from Chain.Consensus.LiuRuntime.LiuQuorum.State import LiuQuorumPhase
from Chain.Consensus.LiuRuntime.LiuZyzzyva.State import LiuZyzzyvaPhase
from Chain.TransactionFactory import Transaction, TransactionFactory
from Engine.Handler import handle_event
from Liu import LiuAction, LiuConsensusProtocol
from Parameters import Parameters
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector
from tests.test_liu_quorum_runtime import LiuQuorumRuntimeFixture


class DynamicEpochFixture:
    def __init__(self):
        self.base = LiuQuorumRuntimeFixture()
        Parameters.simulation["debugging_mode"] = False
        settings = LiuRuntimeProtocolSettings(
            signature_cycles_alpha=1_000_000_000.0,
            mac_cycles_beta=1_000_000_000.0,
            request_timeout_s=100.0,
            propagation_delay_s=0.0,
        )
        self.factories = {
            protocol: LiuRuntimeProtocolFactory(settings)
            for protocol in LiuConsensusProtocol
        }
        self.applicator = LiuRuntimeActionApplicator(
            self.base.nodes,
            self.base.epoch_context,
            self.factories,
        )
        self.base.run_to_all_observed()

    @property
    def nodes(self):
        return self.base.nodes

    @property
    def queue(self):
        return self.base.queue

    def action(self, validators=(1, 2, 3, 4), protocol=LiuConsensusProtocol.PBFT, size=2.0, interval=1.0):
        return LiuAction(5, 4, validators, protocol, size, interval)

    def add_transaction(self, tx_id, size=1.0):
        TransactionFactory.global_mempool.append(Transaction(0, tx_id, 0.0, size))

    def run_to_height(self, height, maximum=2000):
        events = []
        for _ in range(maximum):
            if all(node.last_block.depth == height for node in self.nodes):
                return events
            event = self.queue.pop_next_event()
            events.append((event, handle_event(event)))
        raise AssertionError(f"runtime did not converge at height {height}")


class DynamicEpochAcceptanceTests(unittest.TestCase):
    def test_epoch_metadata_is_immutable_hashed_and_historically_resolvable(self):
        fixture = DynamicEpochFixture()
        result = fixture.applicator.apply(fixture.action(), 1, 20.0)
        epoch = result.epoch_configuration
        self.assertEqual(epoch.epoch_hash, result.epoch_context.epoch_hash)
        self.assertEqual(fixture.applicator.context_for_epoch(0), fixture.base.epoch_context)
        self.assertEqual(fixture.applicator.context_for_epoch(1, epoch.epoch_hash), result.epoch_context)
        with self.assertRaises(ValueError):
            fixture.applicator.context_for_epoch(1, "0" * 64)
        with self.assertRaises(FrozenInstanceError):
            epoch.activation_height = 3

    def test_validator_rotation_sync_gating_and_stale_old_epoch_event(self):
        fixture = DynamicEpochFixture()
        old_event = next(
            event for _, event in fixture.queue.prio_queue.pq
            if hasattr(event, "liu_context") and event.liu_context.epoch_id == 0
        )
        # Model an entering observer that missed the last finalized announcement.
        fixture.nodes[4].blockchain.pop()
        fixture.nodes[4].state.synced = False

        result = fixture.applicator.apply(fixture.action(), 1, 20.0)
        self.assertEqual(result.epoch_configuration.epoch_id, 1)
        self.assertEqual(result.epoch_configuration.activation_height, 2)
        self.assertEqual(result.epoch_configuration.previous_epoch_hash, fixture.base.epoch_context.epoch_hash)
        self.assertEqual(result.validator_added, (4,))
        self.assertEqual(result.validator_removed, (0,))
        self.assertEqual(result.pending_validator_sync, (4,))
        self.assertFalse(fixture.nodes[0].is_validator)
        self.assertFalse(fixture.nodes[4].liu_epoch_ready)
        self.assertIs(fixture.nodes[0].cp.state.phase, LiuPBFTPhase.OBSERVER)
        self.assertEqual(handle_event(old_event), "invalid")
        self.assertTrue(any(record.event_type == "stale_old_epoch_event_rejected" for record in LiuRuntimeInstrumentationCollector.epoch_lifecycle))

        fixture.applicator.complete_validator_sync(4, 1, 20.1)
        self.assertTrue(fixture.nodes[4].liu_epoch_ready)
        self.assertEqual(fixture.nodes[4].last_block.depth, 1)
        fixture.add_transaction(12)
        fixture.run_to_height(2)
        epoch_one_certificates = [record for record in LiuRuntimeInstrumentationCollector.certificates if record.epoch_id == 1]
        self.assertTrue(epoch_one_certificates)
        self.assertTrue(all(set(record.signer_ids).issubset({1, 2, 3, 4}) for record in epoch_one_certificates))
        self.assertFalse(any(0 in record.signer_ids for record in epoch_one_certificates))

    def test_protocol_switch_sequence_preserves_parent_continuity(self):
        fixture = DynamicEpochFixture()
        parents = [parent_digest(fixture.nodes[0].last_block)]
        schedule = (
            (LiuConsensusProtocol.PBFT, (0, 1, 2, 3)),
            (LiuConsensusProtocol.ZYZZYVA, (0, 1, 2, 3)),
        )
        expected_types = (LiuPBFTPhase, LiuZyzzyvaPhase)
        for height, ((protocol, validators), phase_type) in enumerate(zip(schedule, expected_types), start=2):
            result = fixture.applicator.apply(fixture.action(validators, protocol), height - 1, 20.0 * height)
            self.assertEqual(result.epoch_configuration.consensus_protocol, protocol)
            self.assertTrue(all(isinstance(node.cp.state.phase, phase_type) for node in fixture.nodes))
            fixture.add_transaction(10 + height)
            fixture.run_to_height(height)
            block = fixture.nodes[0].last_block
            self.assertEqual(block.previous, fixture.nodes[0].blockchain[-2].id)
            self.assertEqual(block.extra_data["liu_identity"]["parent_digest"], parents[-1])
            parents.append(parent_digest(block))
        self.assertEqual([context.epoch_id for context in fixture.applicator.epoch_history], [0, 1, 2])

    def test_block_size_interval_and_full_action_start_at_activation_height(self):
        fixture = DynamicEpochFixture()
        action = fixture.action((1, 2, 3, 4), LiuConsensusProtocol.ZYZZYVA, 4.0, 1.5)
        result = fixture.applicator.apply(action, 1, 50.0)
        self.assertEqual(result.epoch_configuration.activation_height, 2)
        self.assertEqual(result.epoch_configuration.block_size_mb, 4.0)
        self.assertEqual(result.epoch_configuration.block_interval_s, 1.5)
        self.assertTrue(all(node.reconfiguration_state.configuration.block_size == 4.0 for node in fixture.nodes))
        self.assertTrue(all(node.reconfiguration_state.configuration.block_time == 1.5 for node in fixture.nodes))
        new_start = min(
            event.time for _, event in fixture.queue.prio_queue.pq
            if hasattr(event, "liu_context") and event.liu_context.epoch_id == 1 and event.payload["type"] == "lz_start_request"
        )
        self.assertEqual(new_start, 51.5)
        applied = next(record for record in LiuRuntimeInstrumentationCollector.epoch_lifecycle if record.event_type == "action_applied")
        self.assertEqual(
            (applied.validator_ids, applied.consensus_protocol, applied.block_size_mb, applied.block_interval_s),
            ((1, 2, 3, 4), "ZYZZYVA", 4.0, 1.5),
        )

    def test_invalid_boundary_or_k_change_is_rejected_without_mutation(self):
        fixture = DynamicEpochFixture()
        before = tuple((node.active_validator_set, node.cp, node.last_block.id) for node in fixture.nodes)
        with self.assertRaisesRegex(ValueError, "configured K"):
            fixture.applicator.apply(LiuAction(5, 3, (0, 1, 2), LiuConsensusProtocol.PBFT, 2.0, 1.0), 1, 20.0)
        with self.assertRaisesRegex(ValueError, "activation parent"):
            fixture.applicator.apply(fixture.action(), 2, 20.0)
        self.assertEqual(before, tuple((node.active_validator_set, node.cp, node.last_block.id) for node in fixture.nodes))
        self.assertEqual(fixture.applicator.current_epoch.epoch_id, 0)

    def test_same_action_schedule_and_seed_is_deterministic(self):
        def run_once():
            random.seed(1837413)
            before_rng = random.getstate()
            fixture = DynamicEpochFixture()
            fixture.applicator.apply(fixture.action((1, 2, 3, 4), LiuConsensusProtocol.PBFT, 2.0, 1.0), 1, 20.0)
            fixture.add_transaction(12)
            fixture.run_to_height(2)
            projection = tuple(
                tuple(
                    (block.depth, block.id, block.previous, block.consensus, copy.deepcopy(block.extra_data))
                    for block in node.blockchain
                )
                for node in fixture.nodes
            )
            return before_rng, random.getstate(), projection, LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
