import copy
from collections import deque
from dataclasses import FrozenInstanceError, replace
import heapq
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


from collections import Counter
from dataclasses import FrozenInstanceError

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))
from Chain.Consensus.LiuRuntime.LiuZyzzyva import Messages, Transitions
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters


from Chain.Block import Block
from Chain.Network import Network
from Chain.Node import Node
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.ValidatorSet import ValidatorSet
from Chain.Consensus.LiuRuntime.Common.Certificates import (
    ZyzzyvaCommitCertificate,
    ZyzzyvaFastReplyCertificate,
    ZyzzyvaNewViewCertificate,
    ZyzzyvaRecoveryFinalityCertificate,
    ZyzzyvaSpeculativeEvidence,
    ZyzzyvaViewChangeEvidence,
    select_zyzzyva_safe_value,
)
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity
from Chain.Consensus.LiuRuntime.Common.Roles import LiuZyzzyvaRoles
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot, run_single_action,
)
from Chain.Consensus.LiuRuntime.LiuZyzzyva import Messages
from Chain.Consensus.LiuRuntime.LiuZyzzyva.Configuration import LiuZyzzyvaRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuZyzzyva.Protocol import LiuZyzzyva
from Chain.Consensus.LiuRuntime.LiuZyzzyva.State import LiuZyzzyvaPhase
from Engine.EventQueue import Queue
from Engine.Handler import handle_event
from Liu import (
    AnalyticalConsensusInput,
    EpochConfiguration,
    LinkStateMatrix,
    LiuConsensusProtocol,
    LiuTransmissionUnitPolicy,
    ThreatScenario,
    ZyzzyvaAnalyticalModel,
    ZyzzyvaAnalyticalPath,
)
from Parameters import Parameters
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector

K = 21
N = 100
VALIDATORS = tuple(range(K))
_CHI_ZYZ = 200
_LAMBDA_ZYZ = 0.6
_STAKES_ZYZ = tuple(10.0 for _ in range(N))
_CAPS_ZYZ = tuple(float(10 + (i % 21)) for i in range(N))
_POS_ZYZ = tuple((0.1 * (i % 10), 0.1 * (i // 10)) for i in range(N))
_LINKS_ZYZ = tuple(
    tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N))
    for i in range(N)
)


def epoch1_message_counts():
    return Counter(
        m.message_type
        for m in LiuRuntimeInstrumentationCollector.protocol_messages
        if m.epoch_id == 1
    )


def run(faulty_ids=(), link_scale=1.0, recovery_delay_s=0.05):
    link_rows = tuple(
        tuple(None if i == j else float(10 + ((i + j) % 91)) * link_scale for j in range(N))
        for i in range(N)
    )
    snapshot = LiuSingleActionSnapshot(
        node_count=N,
        transaction_size_bytes=float(_CHI_ZYZ),
        stakes_tokens=_STAKES_ZYZ,
        capabilities_ghz=_CAPS_ZYZ,
        positions_km=_POS_ZYZ,
        link_rows_mbps=link_rows,
        faulty_node_ids=faulty_ids,
        epoch0_validator_ids=VALIDATORS,
    )
    action = LiuAction(N, K, VALIDATORS, LiuConsensusProtocol.ZYZZYVA, 0.2, 1.0)
    geo = ContinuousSpatialIntensityModel(
        planar_gradient_intensity(K, _LAMBDA_ZYZ), K,
        lambda_form=f"planar_gradient s={_LAMBDA_ZYZ}",
    )
    params = LiuReferenceParameters(
        signature_verification_cycles_alpha=2_000_000.0,
        mac_operation_cycles_beta=1_000_000.0,
        network_timeout_s=100.0,
        finality_multiplier_omega=6.0,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=_CHI_ZYZ,
        recovery_delay_s=recovery_delay_s,
    )
    return run_single_action(
        snapshot, action, geo,
        reference_params=params,
        stake_gini_threshold=0.2,
        geographic_gini_threshold=0.3,
        finality_multiplier_omega=6.0,
        signature_cycles_alpha=2_000_000.0,
        mac_cycles_beta=1_000_000.0,
        offered_workload_tx=100_000,
        workload_tx_size_mb=_CHI_ZYZ / 1_000_000.0,
        max_events=2_000_000,
    )


def speculative(signer_ids, block_digest, originating_view=0):
    ids = tuple(signer_ids)
    return ZyzzyvaSpeculativeEvidence(
        epoch_id=0,
        height=1,
        originating_view=originating_view,
        block_digest=block_digest,
        parent_digest="0" * 64,
        signer_ids=ids,
        threshold=len(ids),
        creation_time=0.0,
    )


def evidence(sender_id, target_view=1, speculative_evidence=None, *, commit=None, locked=None):
    return ZyzzyvaViewChangeEvidence(
        epoch_id=0,
        height=1,
        target_view=target_view,
        sender_id=sender_id,
        highest_speculative_evidence=speculative_evidence,
        commit_certificate=commit,
        local_commit_digest=None,
        locked_digest=locked,
    )


class LiuZyzzyvaRuntimeFixture:
    def __init__(self, links=None, timeout_s=100.0, validator_count=4, block_interval_s=0.5, alpha=1_000_000_000.0, beta=1_000_000_000.0):
        node_count = validator_count + 1
        Parameters.simulation = {"debugging_mode": False, "event_id": 0, "events": {}}
        Parameters.application = {"Nn": node_count, "transaction_model": "global"}
        Parameters.data = {"base_block_size": 0.0}
        Parameters.network = {"gossip": True}
        InstrumentationCollector.reset()
        LiuRuntimeInstrumentationCollector.reset()
        TransactionFactory.global_mempool = deque([Transaction(0, 11, 0.0, 1.0)])
        TransactionFactory.depth_removed = -1
        TransactionFactory.produced_tx = 0

        self.queue = Queue()
        self.validator_set = ValidatorSet(tuple(range(validator_count)))
        self.profiles = NodeProfileSet(tuple(NodeProfile(node_id, None, 1.0) for node_id in range(node_count)))
        self.links = links or LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 10.0 for j in range(node_count)) for i in range(node_count))
        )
        epoch = EpochConfiguration(0, self.validator_set, LiuConsensusProtocol.ZYZZYVA, 1.0, block_interval_s)
        self.epoch_context = LiuRuntimeEpochContext(epoch, self.profiles, self.links, ThreatScenario(node_count, ()))
        self.configuration = LiuZyzzyvaRuntimeConfiguration(
            self.epoch_context,
            signature_cycles_alpha=alpha,
            mac_cycles_beta=beta,
            request_timeout_s=timeout_s,
            propagation_delay_s=0.0,
            faulty_replica_count=0,
        )
        self.nodes = [
            Node(node_id, self.queue, self.validator_set, self.profiles.profile_for(node_id))
            for node_id in range(node_count)
        ]
        Network.nodes = self.nodes
        Network.received = {node: set() for node in self.nodes}
        genesis = Block(depth=0, id=7, previous=-1, size=0.0)
        genesis.extra_data = {"configuration_depth": 0, "round": -1}
        for node in self.nodes:
            node.blockchain = [genesis.copy()]
            node.reconfiguration_state.configuration = SimpleNamespace(block_size=1.0, block_time=block_interval_s)
            node.cp = LiuZyzzyva(node, self.configuration)
        TransactionFactory.nodes = self.nodes
        for node in self.nodes:
            node.cp.init(0.0, 0)

    def step(self):
        event = self.queue.pop_next_event()
        return event, handle_event(event)

    def run_until(self, predicate, maximum=200):
        events = []
        for _ in range(maximum):
            if predicate():
                return events
            event, result = self.step()
            events.append((event, result))
        raise AssertionError("Zyzzyva fixture did not reach the requested condition")

    def run_to_all_observed(self):
        return self.run_until(lambda: all(node.last_block.depth == 1 for node in self.nodes))

    def projection(self):
        return tuple(
            tuple(
                (
                    block.depth,
                    block.id,
                    block.previous,
                    block.time_created,
                    block.time_added,
                    block.miner,
                    block.size,
                    tuple(tx.id for tx in block.transactions),
                    copy.deepcopy(block.extra_data),
                )
                for block in node.blockchain
            )
            for node in self.nodes
        )


class LiuZyzzyvaConfigurationTests(unittest.TestCase):
    def test_k21_roles_and_full_fast_reply_threshold(self):
        validators = ValidatorSet(tuple(range(21)))
        roles = LiuZyzzyvaRoles.for_height_view(validators, 1, 0)
        self.assertEqual(roles.client_id, 1)
        self.assertEqual(roles.primary_id, 0)
        self.assertEqual(len(roles.replica_ids), 20)
        self.assertNotIn(roles.client_id, roles.replica_ids)

        profiles = NodeProfileSet(tuple(NodeProfile(node_id, None, 20.0) for node_id in range(21)))
        links = LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 100.0 for j in range(21)) for i in range(21))
        )
        epoch = EpochConfiguration(0, validators, LiuConsensusProtocol.ZYZZYVA, 1.0, 0.5)
        configuration = LiuZyzzyvaRuntimeConfiguration(
            LiuRuntimeEpochContext(epoch, profiles, links, ThreatScenario(21, ())),
            1.0,
            1.0,
            10.0,
        )
        self.assertEqual(configuration.required_fast_replies, 20)
        self.assertEqual(configuration.tolerated_faults, 6)
        self.assertEqual(configuration.speculative_recovery_quorum, 13)
        self.assertEqual(configuration.local_commit_quorum, 13)
        self.assertEqual(configuration.RECOVERY_QUORUM_POLICY_VERSION, "liu_zyzzyva_recovery_quorum_v1")
        self.assertEqual(configuration.RECOVERY_SAFE_DIGEST_POLICY_VERSION, "liu_zyzzyva_recovery_safe_digest_v1")

        fixture = LiuZyzzyvaRuntimeFixture()
        self.assertEqual(fixture.configuration.required_fast_replies, 3)
        self.assertEqual(fixture.configuration.PROCESSING_DECOMPOSITION_VERSION, "liu_zyzzyva_fast_processing_decomposition_v1")
        self.assertEqual(fixture.configuration.FAST_REPLY_POLICY_VERSION, "liu_zyzzyva_all_replica_replies_v1")

    def test_fault_count_must_equal_selected_malicious_and_must_not_exceed_F1(self):
        # Zyzzyva now permits the Liu Eq.(17) recovery path. Configuration cross-checks
        # that faulty_replica_count equals |malicious_node_ids ∩ selected validators|,
        # and rejects any selected-set faulty count exceeding F^1 = floor((K-1)/3).
        fixture = LiuZyzzyvaRuntimeFixture()
        # f>0 declared but no malicious in selected -> mismatch.
        with self.assertRaisesRegex(ValueError, "faulty_replica_count must equal"):
            LiuZyzzyvaRuntimeConfiguration(
                fixture.epoch_context,
                1.0,
                1.0,
                10.0,
                faulty_replica_count=1,
            )
        # Malicious selected but declared count wrong (should be 1, not 0).
        malicious_context = LiuRuntimeEpochContext(
            fixture.epoch_context.epoch_configuration,
            fixture.profiles,
            fixture.links,
            ThreatScenario(5, (2,)),
        )
        with self.assertRaisesRegex(ValueError, "faulty_replica_count must equal"):
            LiuZyzzyvaRuntimeConfiguration(malicious_context, 1.0, 1.0, 10.0)
        # Matched (count==1) succeeds under K=4, f=1 <= F^1 = 1.
        cfg = LiuZyzzyvaRuntimeConfiguration(
            malicious_context, 1.0, 1.0, 10.0, faulty_replica_count=1
        )
        self.assertEqual(cfg.faulty_replica_count, 1)
        self.assertEqual(cfg.faulty_replica_ids, frozenset({2}))


class LiuZyzzyvaFastPathTests(unittest.TestCase):
    def test_roles_traffic_full_certificate_finality_and_observer_exclusion(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_to_all_observed()
        messages = LiuRuntimeInstrumentationCollector.protocol_messages
        by_type = {
            kind: [record for record in messages if record.message_type == kind]
            for kind in (Messages.REQUEST, Messages.ORDER_REQUEST, Messages.SPECULATIVE_REPLY, Messages.FINALIZED_BLOCK)
        }
        self.assertEqual([(r.sender_id, r.receiver_id) for r in by_type[Messages.REQUEST]], [(1, 0)])
        self.assertEqual({(r.sender_id, r.receiver_id) for r in by_type[Messages.ORDER_REQUEST]}, {(0, 2), (0, 3)})
        self.assertEqual({(r.sender_id, r.receiver_id) for r in by_type[Messages.SPECULATIVE_REPLY]}, {(0, 1), (2, 1), (3, 1)})
        self.assertEqual(len(by_type[Messages.FINALIZED_BLOCK]), 4)
        self.assertFalse(
            any(
                record.message_type != Messages.FINALIZED_BLOCK
                and (record.sender_id == 4 or record.receiver_id == 4)
                for record in messages
            )
        )

        certificates = [
            record
            for record in LiuRuntimeInstrumentationCollector.certificates
            if record.certificate_type == "zyzzyva_fast_all_replica_replies"
        ]
        self.assertEqual(len(certificates), 1)
        self.assertEqual(certificates[0].signer_ids, (0, 2, 3))
        self.assertEqual(certificates[0].threshold, 3)
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities[0].finality_path, "zyzzyva_fast_all_replica_replies")
        self.assertEqual({node.last_block.id for node in fixture.nodes}, {fixture.nodes[1].last_block.id})
        self.assertEqual(fixture.nodes[4].cp.state.phase, LiuZyzzyvaPhase.OBSERVER)
        observations = [
            record
            for record in InstrumentationCollector.block_observations
            if record.consensus_protocol == "LiuZyzzyva" and record.block_depth == 1
        ]
        self.assertEqual(len(observations), 5)
        self.assertEqual(
            {record.cause for record in observations},
            {"protocol_finality", "certified_finalized_block_announcement"},
        )

    def test_speculative_processing_is_instrumented_for_every_replica(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_to_all_observed()
        starts = [
            record
            for record in LiuRuntimeInstrumentationCollector.phase_transitions
            if record.new_phase == LiuZyzzyvaPhase.PROCESSING_SPECULATIVE.value
        ]
        ends = [
            record
            for record in LiuRuntimeInstrumentationCollector.phase_transitions
            if record.new_phase == LiuZyzzyvaPhase.SPECULATIVELY_EXECUTED.value
        ]
        self.assertEqual({record.node_id for record in starts}, {0, 2, 3})
        self.assertEqual({record.node_id for record in ends}, {0, 2, 3})
        self.assertTrue(all(record.processing_work_name == "zyzzyva_speculative_execution" for record in starts))

    def test_duplicate_reply_is_ignored(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.SPECULATIVE_REPLY for item in fixture.queue.prio_queue.pq)
        )
        reply = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.SPECULATIVE_REPLY)
        self.assertIn(handle_event(reply), ("handled", "new_state"))
        state = fixture.nodes[1].cp.state
        count = len(state.speculative_replies[state.block_identity.block_digest])
        self.assertEqual(handle_event(reply), "handled")
        self.assertEqual(len(state.speculative_replies[state.block_identity.block_digest]), count)

    def test_missing_reply_times_out_to_explicit_pending_recovery(self):
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=20.0)
        fixture.run_until(
            lambda: len(
                [
                    record
                    for record in LiuRuntimeInstrumentationCollector.protocol_messages
                    if record.message_type == Messages.SPECULATIVE_REPLY
                ]
            )
            == 3
        )
        omitted = next(
            item[1]
            for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == Messages.SPECULATIVE_REPLY and item[1].creator.id == 3
        )
        fixture.queue.remove_event(omitted)
        fixture.run_until(lambda: fixture.nodes[1].cp.state.phase is LiuZyzzyvaPhase.PENDING_RECOVERY)
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities, [])
        self.assertIn("missing_fast_replies", fixture.nodes[1].cp.state.failure_reason)

    def test_conflicting_digest_enters_pending_recovery_without_finality(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.SPECULATIVE_REPLY for item in fixture.queue.prio_queue.pq)
        )
        reply = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.SPECULATIVE_REPLY)
        reply.payload["reply_digest"] = "0" * 64
        self.assertEqual(handle_event(reply), "new_state")
        self.assertEqual(fixture.nodes[1].cp.state.phase, LiuZyzzyvaPhase.PENDING_RECOVERY)
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities, [])

    def test_stale_epoch_view_and_wrong_parent_are_rejected(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.ORDER_REQUEST for item in fixture.queue.prio_queue.pq)
        )
        order = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.ORDER_REQUEST)
        stale_epoch = copy.copy(order)
        stale_epoch.liu_context = replace(order.liu_context, epoch_id=99)
        stale_view = copy.copy(order)
        stale_view.liu_context = replace(order.liu_context, view=1)
        stale_height = copy.copy(order)
        stale_height.liu_context = replace(order.liu_context, height=0)
        wrong_parent = copy.copy(order)
        wrong_parent.payload = dict(order.payload)
        wrong_parent.payload["block"] = order.payload["block"].copy()
        wrong_parent.payload["block"].previous = 999
        self.assertEqual(handle_event(stale_epoch), "invalid")
        self.assertEqual(handle_event(stale_view), "invalid")
        self.assertEqual(handle_event(stale_height), "invalid")
        self.assertEqual(handle_event(wrong_parent), "invalid")

    def test_fast_certificate_is_immutable_and_requires_exact_expected_replicas(self):
        identity = LiuBlockIdentity(0, 1, "p", "d")
        certificate = ZyzzyvaFastReplyCertificate(0, 1, 0, "d", "p", (3, 0, 2), 3, 8.9)
        reordered = ZyzzyvaFastReplyCertificate(0, 1, 0, "d", "p", (2, 3, 0), 3, 8.9)
        self.assertEqual(certificate.signer_ids, (0, 2, 3))
        self.assertEqual(certificate.deterministic_hash(), reordered.deterministic_hash())
        self.assertTrue(certificate.validate(identity, 0, (0, 2, 3)))
        self.assertFalse(certificate.validate(identity, 0, (0, 2)))
        with self.assertRaises(ValueError):
            ZyzzyvaFastReplyCertificate(0, 1, 0, "d", "p", (0, 2), 3, 8.9)
        with self.assertRaises(FrozenInstanceError):
            certificate.threshold = 2

    def test_certified_announcement_with_parent_mismatch_is_rejected(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        announcement = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.FINALIZED_BLOCK)
        receiver = announcement.actor
        announcement.payload["block"].previous = 999
        self.assertEqual(handle_event(announcement), "invalid")
        self.assertEqual(receiver.last_block.depth, 0)

    def test_same_seed_is_deterministic_and_consumes_no_global_rng(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            fixture = LiuZyzzyvaRuntimeFixture()
            fixture.run_to_all_observed()
            return before, random.getstate(), fixture.projection(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2:], second[2:])

    def test_fault_free_des_matches_analytical_fast_path_oracle(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        des_consensus = LiuRuntimeInstrumentationCollector.protocol_finalities[0].consensus_latency_s
        messages = LiuRuntimeInstrumentationCollector.protocol_messages
        des_delivery = sum(
            max(record.serialization_delay_s for record in messages if record.message_type == message_type)
            for message_type in (Messages.REQUEST, Messages.ORDER_REQUEST, Messages.SPECULATIVE_REPLY)
        )
        des_validation = des_consensus - des_delivery
        analytical = ZyzzyvaAnalyticalModel().evaluate(
            AnalyticalConsensusInput(
                validator_ids=(0, 1, 2, 3),
                validator_count_k=4,
                protocol=LiuConsensusProtocol.ZYZZYVA,
                client_validator_id=1,
                primary_validator_id=0,
                block_size_mb=1.0,
                transaction_size_bytes=1_000_000,
                batch_size_m=1,
                block_interval_s=0.5,
                computational_capabilities_ghz=(1.0, 1.0, 1.0, 1.0),
                directed_link_rates_mbps=LinkStateMatrix.from_rows(
                    tuple(tuple(None if i == j else 10.0 for j in range(4)) for i in range(4))
                ),
                signature_verification_cycles_alpha=1_000_000_000.0,
                mac_operation_cycles_beta=1_000_000_000.0,
                network_timeout_s=100.0,
                faulty_replica_count=0,
                zyzzyva_path=ZyzzyvaAnalyticalPath.FAST,
                recovery_delay_s=0.0,
                transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
            )
        )
        self.assertAlmostEqual(des_delivery, 2.4)
        self.assertAlmostEqual(des_validation, 6.0)
        self.assertAlmostEqual(des_consensus, 8.4)
        self.assertAlmostEqual(des_delivery, analytical.delivery_delay_s)
        self.assertAlmostEqual(des_validation, analytical.validation_delay_s)
        self.assertAlmostEqual(des_consensus, analytical.consensus_delay_s)


class LiuZyzzyvaRecoveryTests(unittest.TestCase):
    @staticmethod
    def recovery_fixture(missing_replica_ids=(4,)):
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=20.0, validator_count=5)
        fixture.run_until(
            lambda: all(
                fixture.nodes[node_id].cp.state.phase is LiuZyzzyvaPhase.PROCESSING_SPECULATIVE
                for node_id in fixture.nodes[1].cp.state.replica_ids
            )
        )
        completions = [
            item[1]
            for item in tuple(fixture.queue.prio_queue.pq)
            if item[1].payload["type"] == Messages.COMPLETE_SPECULATIVE
            and item[1].actor.id in missing_replica_ids
        ]
        for event in completions:
            queued_index = next(index for index, item in enumerate(fixture.queue.prio_queue.pq) if item[1] is event)
            del fixture.queue.prio_queue.pq[queued_index]
            heapq.heapify(fixture.queue.prio_queue.pq)
        return fixture

    def test_timeout_with_2f_plus_1_replies_executes_t4_t5_and_recovery_finality(self):
        fixture = self.recovery_fixture()
        fixture.run_to_all_observed()
        certificates = {record.certificate_type: record for record in LiuRuntimeInstrumentationCollector.certificates}
        commit = certificates["zyzzyva_recovery_commit_certificate"]
        finality = certificates["zyzzyva_recovery_finality_certificate"]
        self.assertEqual((commit.threshold, len(commit.signer_ids)), (3, 3))
        self.assertEqual((finality.threshold, len(finality.signer_ids)), (3, 3))
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities[0].finality_path, "zyzzyva_recovery_local_commit_quorum")
        messages = LiuRuntimeInstrumentationCollector.protocol_messages
        self.assertEqual(
            {(record.sender_id, record.receiver_id) for record in messages if record.message_type == Messages.COMMIT_CERTIFICATE},
            {(1, 0), (1, 2), (1, 3), (1, 4)},
        )
        self.assertEqual(
            {record.sender_id for record in messages if record.message_type == Messages.LOCAL_COMMIT},
            {0, 2, 3, 4},
        )
        self.assertFalse(
            any(
                record.message_type in (Messages.COMMIT_CERTIFICATE, Messages.LOCAL_COMMIT)
                and (record.sender_id == 5 or record.receiver_id == 5)
                for record in messages
            )
        )
        self.assertEqual({node.last_block.id for node in fixture.nodes}, {fixture.nodes[1].last_block.id})

    def test_fewer_than_2f_plus_1_replies_remains_pending_without_certificate(self):
        fixture = self.recovery_fixture(missing_replica_ids=(3, 4))
        fixture.run_until(lambda: fixture.nodes[1].cp.state.phase is LiuZyzzyvaPhase.PENDING_RECOVERY)
        self.assertEqual(len(fixture.nodes[1].cp.state.speculative_replies[fixture.nodes[1].cp.state.block_identity.block_digest]), 2)
        self.assertIsNone(fixture.nodes[1].cp.state.recovery_commit_certificate)
        self.assertEqual(LiuRuntimeInstrumentationCollector.certificates, [])
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities, [])

    def test_conflicting_reply_digest_is_never_combined_with_matching_replies(self):
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=20.0, validator_count=5)
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.SPECULATIVE_REPLY for item in fixture.queue.prio_queue.pq)
        )
        conflicting = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.SPECULATIVE_REPLY
        )
        signer = conflicting.creator.id
        conflicting.payload["reply_digest"] = "f" * 64
        self.assertEqual(handle_event(conflicting), "new_state")
        fixture.run_until(lambda: fixture.nodes[1].cp.state.recovery_commit_certificate is not None)
        certificate = fixture.nodes[1].cp.state.recovery_commit_certificate
        self.assertNotIn(signer, certificate.signer_ids)
        self.assertEqual(certificate.block_digest, fixture.nodes[1].cp.state.block_identity.block_digest)
        self.assertIn(signer, fixture.nodes[1].cp.state.conflicting_speculative_replies["f" * 64])

    def test_commit_and_recovery_finality_certificates_are_immutable_and_exact(self):
        identity = LiuBlockIdentity(0, 1, "parent", "digest")
        commit = ZyzzyvaCommitCertificate(0, 1, 0, "digest", "parent", (4, 0, 2), 3, 10.0)
        reordered = ZyzzyvaCommitCertificate(0, 1, 0, "digest", "parent", (2, 4, 0), 3, 10.0)
        self.assertEqual(commit.deterministic_hash(), reordered.deterministic_hash())
        self.assertEqual(commit.QUORUM_POLICY_VERSION, "liu_zyzzyva_recovery_quorum_v1")
        self.assertEqual(commit.SAFE_DIGEST_POLICY_VERSION, "liu_zyzzyva_recovery_safe_digest_v1")
        self.assertEqual(commit.creation_time, 10.0)
        self.assertTrue(commit.validate(identity, 0, (0, 2, 3, 4), 3))
        with self.assertRaises(ValueError):
            ZyzzyvaCommitCertificate(0, 1, 0, "digest", "parent", (0, 2), 3, 10.0)
        finality = ZyzzyvaRecoveryFinalityCertificate(
            0, 1, 0, "digest", "parent", 1, (2, 3, 4), 3, commit.deterministic_hash(), 14.0
        )
        self.assertTrue(finality.validate(identity, 0, 1, (0, 2, 3, 4), 3, commit))
        with self.assertRaises(FrozenInstanceError):
            commit.threshold = 2

    def test_local_commit_requires_valid_certificate_and_cannot_change_digest(self):
        fixture = self.recovery_fixture()
        fixture.run_until(lambda: fixture.nodes[2].cp.state.local_committed_height == 1)
        replica = fixture.nodes[2].cp
        committed_digest = replica.state.local_commit_digest
        commit_message = next(
            record
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == Messages.COMMIT_CERTIFICATE and record.receiver_id == 2
        )
        self.assertIsNotNone(commit_message)
        forged_identity = replace(replica.state.block_identity, block_digest="0" * 64)
        forged = ZyzzyvaCommitCertificate(
            0,
            1,
            0,
            "0" * 64,
            forged_identity.parent_digest,
            (0, 2, 3),
            3,
            30.0,
        )
        event = Messages.send_commit_certificate(fixture.nodes[1].cp, 2, 30.0, forged_identity, forged)
        self.assertEqual(handle_event(event), "invalid")
        self.assertEqual(replica.state.local_commit_digest, committed_digest)
        self.assertEqual(replica.state.local_committed_height, 1)

    def test_duplicate_local_commit_does_not_increase_quorum(self):
        fixture = self.recovery_fixture()
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.LOCAL_COMMIT for item in fixture.queue.prio_queue.pq)
        )
        acknowledgement = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.LOCAL_COMMIT
        )
        self.assertIn(handle_event(acknowledgement), ("handled", "new_state"))
        state = fixture.nodes[1].cp.state
        count = len(state.recovery_local_commits[state.block_identity.block_digest])
        self.assertEqual(handle_event(acknowledgement), "handled")
        self.assertEqual(len(state.recovery_local_commits[state.block_identity.block_digest]), count)

    def test_stale_recovery_epoch_view_and_height_are_rejected(self):
        fixture = self.recovery_fixture()
        fixture.run_until(
            lambda: any(item[1].payload["type"] == Messages.COMMIT_CERTIFICATE for item in fixture.queue.prio_queue.pq)
        )
        message = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.COMMIT_CERTIFICATE
        )
        stale_epoch = copy.copy(message)
        stale_epoch.liu_context = replace(message.liu_context, epoch_id=99)
        stale_view = copy.copy(message)
        stale_view.liu_context = replace(message.liu_context, view=1)
        stale_height = copy.copy(message)
        stale_height.liu_context = replace(message.liu_context, height=2)
        self.assertEqual(handle_event(stale_epoch), "invalid")
        self.assertEqual(handle_event(stale_view), "invalid")
        self.assertEqual(handle_event(stale_height), "invalid")
        self.assertIsNone(message.actor.cp.state.recovery_commit_certificate)

    def test_recovery_processing_matches_equation17_with_causal_difference_documented(self):
        fixture = self.recovery_fixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        commit_record = next(
            record
            for record in LiuRuntimeInstrumentationCollector.certificates
            if record.certificate_type == "zyzzyva_recovery_commit_certificate"
        )
        reply_arrivals = {
            record.sender_id: record.arrival_at
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == Messages.SPECULATIVE_REPLY
        }
        recovery_wait = commit_record.creation_time - max(reply_arrivals[signer] for signer in commit_record.signer_ids)
        network_delivery = 5 * 0.8
        des_delivery = network_delivery + recovery_wait
        des_consensus = finality.consensus_latency_s
        des_validation = des_consensus - des_delivery
        analytical = ZyzzyvaAnalyticalModel().evaluate(
            AnalyticalConsensusInput(
                validator_ids=(0, 1, 2, 3, 4),
                validator_count_k=5,
                protocol=LiuConsensusProtocol.ZYZZYVA,
                client_validator_id=1,
                primary_validator_id=0,
                block_size_mb=1.0,
                transaction_size_bytes=1_000_000,
                batch_size_m=1,
                block_interval_s=0.5,
                computational_capabilities_ghz=(1.0,) * 5,
                directed_link_rates_mbps=LinkStateMatrix.from_rows(
                    tuple(tuple(None if i == j else 10.0 for j in range(5)) for i in range(5))
                ),
                signature_verification_cycles_alpha=1_000_000_000.0,
                mac_operation_cycles_beta=1_000_000_000.0,
                network_timeout_s=100.0,
                faulty_replica_count=1,
                zyzzyva_path=ZyzzyvaAnalyticalPath.RECOVERY,
                recovery_delay_s=recovery_wait,
                transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
            )
        )
        self.assertAlmostEqual(recovery_wait, 10.6)
        self.assertAlmostEqual(des_delivery, analytical.delivery_delay_s)
        self.assertAlmostEqual(des_validation, 9.0)
        self.assertAlmostEqual(analytical.validation_delay_s, 10.0)
        self.assertAlmostEqual(des_consensus, 23.6)
        self.assertAlmostEqual(analytical.consensus_delay_s, 24.6)

    def test_recovery_same_seed_is_deterministic_and_consumes_no_global_rng(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            fixture = self.recovery_fixture()
            fixture.run_to_all_observed()
            return before, random.getstate(), fixture.projection(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2:], second[2:])


class RoleFidelityTests(unittest.TestCase):
    def test_client_and_primary_distinct_correct(self):
        roles = LiuZyzzyvaRoles.for_height_view(ValidatorSet(VALIDATORS), height=2, view=0)
        self.assertNotEqual(roles.client_id, roles.primary_id)
        self.assertIn(roles.primary_id, roles.replica_ids)
        self.assertNotIn(roles.client_id, roles.replica_ids)
        self.assertEqual(len(roles.replica_ids), K - 1)


class FastPathOperationCountTests(unittest.TestCase):
    def test_primary_and_replica_counts_match_eq16_m1_f0(self):
        prim_ord = LiuProcessingWork.zyzzyva_primary_order_work(K - 1)   # (0, K-1)
        spec = LiuProcessingWork.zyzzyva_speculative_replica_work()      # (1, 2)
        prim_sig = prim_ord.signature_operations + spec.signature_operations
        prim_mac = prim_ord.mac_operations + spec.mac_operations
        rep_sig = spec.signature_operations
        rep_mac = spec.mac_operations
        # Eq.16 at M=1, f=0: primary = (M, 2M+K-1), replica = (M, M+1).
        self.assertEqual((prim_sig, prim_mac), (1, 2 * 1 + (K - 1)))     # (1, 22)
        self.assertEqual((rep_sig, rep_mac), (1, 1 + 1))                 # (1, 2)


class FastPathGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(faulty_ids=())
        cls.d = cls.result.to_dict()

    def test_fast_path_finalizes_1000_transactions(self):
        self.assertEqual(self.d["des_observed"]["finalized_transactions"], 1000)

    def test_fast_path_three_leg_message_pattern(self):
        by = epoch1_message_counts()
        self.assertEqual(by["lz_request"], 1)                 # t1 client -> primary
        self.assertEqual(by["lz_order_request"], K - 2)       # t2 primary -> other backups
        self.assertEqual(by["lz_speculative_reply"], K - 1)   # t3 replicas -> client
        self.assertNotIn("lz_commit_certificate", by)         # no recovery legs
        self.assertNotIn("lz_local_commit", by)

    def test_fast_path_c1_c2_c3_all_pass_and_reward_is_finalized_tps(self):
        des = self.d["des_observed"]
        self.assertTrue(des["c1_decentralization_passed"] and des["c2_finality_passed"] and des["c3_security_passed"])
        self.assertEqual(des["reward_des_realized"], des["throughput_des_finalized"])
        self.assertAlmostEqual(
            des["throughput_des_finalized"],
            des["finalized_transactions"] / des["measurement_duration_s"],
            places=6,
        )

    def test_paper_oracle_available_beside_des(self):
        paper = self.d["paper_reference"]
        self.assertEqual(paper["throughput_paper_nominal"], 1000.0)
        self.assertEqual(paper["reward_paper_reference"], 1000.0)


class RecoveryPathGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pick a non-primary, non-client replica as the single faulty node.
        roles = LiuZyzzyvaRoles.for_height_view(ValidatorSet(VALIDATORS), height=2, view=0)
        cls.faulty = (next(i for i in roles.replica_ids if i != roles.primary_id),)
        cls.result = run(faulty_ids=cls.faulty, recovery_delay_s=0.05)
        cls.d = cls.result.to_dict()

    def test_recovery_finalizes_1000_transactions(self):
        self.assertEqual(self.d["des_observed"]["finalized_transactions"], 1000)

    def test_recovery_c3_uses_selected_faulty_count(self):
        des = self.d["des_observed"]
        self.assertEqual(des["malicious_validator_count"], 1)
        self.assertEqual(des["tolerated_fault_count"], (K - 1) // 3)
        self.assertTrue(des["c3_security_passed"])

    def test_recovery_five_leg_message_pattern_present(self):
        by = epoch1_message_counts()
        # Eq.17: t1 request, t2 primary->backups, t3 replicas->client,
        #        t4 client->replicas (COMMIT_CERTIFICATE), t5 replicas->client (LOCAL_COMMIT).
        self.assertEqual(by["lz_request"], 1)
        self.assertEqual(by["lz_order_request"], K - 2)
        self.assertEqual(by["lz_speculative_reply"], K - 1)
        self.assertEqual(by["lz_commit_certificate"], K - 1)     # t4 (client -> replicas)
        self.assertEqual(by["lz_local_commit"], K - 1)           # t5 (replicas -> client)

    def test_recovery_latency_strictly_greater_than_fast(self):
        fast = run(faulty_ids=()).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreater(self.d["des_observed"]["t_c_des_s"], fast)


class ExactCapacityAndCausalityTests(unittest.TestCase):
    def test_zyzzyva_fast_capacity_uses_exact_byte_accounting(self):
        # Regression: block-capacity fix must apply on the Zyzzyva path too.
        self.assertEqual(run(faulty_ids=()).to_dict()["des_observed"]["finalized_transactions"], 1000)

    def test_slower_links_do_not_decrease_t_c_des(self):
        fast = run().to_dict()["des_observed"]["t_c_des_s"]
        slow = run(link_scale=0.5).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreater(slow, fast)


class DeterminismTests(unittest.TestCase):
    def test_same_snapshot_action_reproduces_identical_fast_result(self):
        self.assertEqual(run(faulty_ids=()).to_dict(), run(faulty_ids=()).to_dict())


class ZyzzyvaViewChangeCertificateTests(unittest.TestCase):
    def test_quorum_formula_requires_strict_honest_intersection(self):
        k21 = LiuZyzzyvaRuntimeFixture(validator_count=21).configuration
        self.assertEqual((k21.replica_count, k21.tolerated_faults, k21.view_change_quorum), (20, 6, 14))
        self.assertGreater(2 * k21.view_change_quorum - k21.replica_count, k21.tolerated_faults)
        k4 = LiuZyzzyvaRuntimeFixture(validator_count=4).configuration
        self.assertEqual((k4.replica_count, k4.tolerated_faults, k4.view_change_quorum), (3, 1, 3))
        self.assertEqual(k21.VIEW_CHANGE_QUORUM_POLICY_VERSION, "liu_zyzzyva_view_change_quorum_v1")

    def test_commit_evidence_has_priority_over_speculative_evidence(self):
        commit = ZyzzyvaCommitCertificate(0, 1, 0, "c" * 64, "p" * 64, (0, 2, 3), 3, 2.0)
        items = (
            evidence(0, speculative_evidence=speculative((0,), "d" * 64), commit=commit, locked=commit.block_digest),
            evidence(2, speculative_evidence=speculative((2,), "d" * 64)),
            evidence(3),
        )
        selected_commit, selected_speculative, error = select_zyzzyva_safe_value(items, 2)
        self.assertEqual(selected_commit, commit)
        self.assertIsNone(selected_speculative)
        self.assertIsNone(error)

    def test_conflicting_commit_certificates_are_explicit_safety_error(self):
        left = ZyzzyvaCommitCertificate(0, 1, 0, "a" * 64, "p" * 64, (0, 2, 3), 3, 2.0)
        right = ZyzzyvaCommitCertificate(0, 1, 0, "b" * 64, "p" * 64, (2, 3, 4), 3, 2.0)
        commit, speculative_value, error = select_zyzzyva_safe_value(
            (evidence(0, commit=left), evidence(2, commit=right), evidence(3)),
            2,
        )
        self.assertIsNone(commit)
        self.assertIsNone(speculative_value)
        self.assertEqual(error, "conflicting_commit_certificates")

    def test_highest_view_matching_speculative_evidence_is_selected(self):
        items = (
            evidence(0, 2, speculative((0,), "a" * 64, 0)),
            evidence(2, 2, speculative((2,), "a" * 64, 0)),
            evidence(3, 2, speculative((0, 3), "b" * 64, 1)),
        )
        commit, selected, error = select_zyzzyva_safe_value(items, 2)
        self.assertIsNone(commit)
        self.assertIsNone(error)
        self.assertEqual((selected.originating_view, selected.block_digest), (1, "b" * 64))

    def test_conflicting_strongest_evidence_is_explicit_safety_error(self):
        items = (
            evidence(0, 2, speculative((0, 2), "a" * 64, 1)),
            evidence(2, 2, speculative((2, 3), "b" * 64, 1)),
            evidence(3, 2),
        )
        commit, selected, error = select_zyzzyva_safe_value(items, 2)
        self.assertIsNone(commit)
        self.assertIsNone(selected)
        self.assertEqual(error, "conflicting_highest_view_speculative_evidence")

    def test_new_view_certificate_is_canonical_immutable_and_exact_quorum(self):
        items = (evidence(0), evidence(2), evidence(3))
        first = ZyzzyvaNewViewCertificate(0, 1, 1, 2, items, 3, 2, None, None, None, None, None, 4.0)
        second = ZyzzyvaNewViewCertificate(
            0, 1, 1, 2, tuple(reversed(items)), 3, 2, None, None, None, None, None, 4.0
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())
        self.assertTrue(first.validate((0, 2, 3, 4), 2, 3, 2, 3))
        with self.assertRaises(ValueError):
            ZyzzyvaNewViewCertificate(0, 1, 1, 2, items[:2], 3, 2, None, None, None, None, None, 4.0)
        with self.assertRaises(ValueError):
            ZyzzyvaNewViewCertificate(0, 1, 1, 2, (items[0], items[0], items[2]), 3, 2, None, None, None, None, None, 4.0)
        with self.assertRaises(FrozenInstanceError):
            first.target_view = 2


class ZyzzyvaViewChangeRuntimeTests(unittest.TestCase):
    @staticmethod
    def silent_primary_fixture():
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=10.0)
        start, _ = fixture.step()
        assert start.payload["type"] == Messages.START_REQUEST
        request = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.REQUEST)
        index = next(index for index, item in enumerate(fixture.queue.prio_queue.pq) if item[1] is request)
        del fixture.queue.prio_queue.pq[index]
        heapq.heapify(fixture.queue.prio_queue.pq)
        fixture.delayed_old_request = request
        return fixture

    @staticmethod
    def insufficient_reply_fixture():
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=10.0, validator_count=5)
        fixture.run_until(
            lambda: len(
                [item for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.SPECULATIVE_REPLY]
            )
            == 4
        )
        replies = [item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.SPECULATIVE_REPLY]
        retained_sender = min(reply.creator.id for reply in replies)
        fixture.queue.prio_queue.pq[:] = [
            item
            for item in fixture.queue.prio_queue.pq
            if not (
                item[1].payload["type"] == Messages.SPECULATIVE_REPLY
                and item[1].creator.id != retained_sender
            )
        ]
        heapq.heapify(fixture.queue.prio_queue.pq)
        return fixture

    def test_silent_primary_reaches_view_change_assisted_finality(self):
        fixture = self.silent_primary_fixture()
        fixture.run_to_all_observed()
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        new_view = next(
            record for record in LiuRuntimeInstrumentationCollector.certificates if record.certificate_type == "zyzzyva_new_view"
        )
        self.assertEqual((new_view.threshold, new_view.signer_ids), (3, (0, 2, 3)))
        self.assertEqual(finality.finality_path, "zyzzyva_view_change_fast_all_replica_replies")
        self.assertAlmostEqual(finality.finality_time, 30.5)
        self.assertEqual({node.last_block.id for node in fixture.nodes}, {fixture.nodes[1].last_block.id})

    def test_view_timeout_payload_is_epoch_height_view_scoped(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        fixture.step()  # REQUEST dispatch arms replica phase timers.
        timeout = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.VIEW_TIMEOUT
        )
        self.assertEqual(
            (timeout.payload["epoch_id"], timeout.payload["height"], timeout.payload["view"]),
            (timeout.liu_context.epoch_id, timeout.liu_context.height, timeout.liu_context.view),
        )
        self.assertIn(timeout.payload["phase"], ("primary_waiting_request", "backup_waiting_order"))
        self.assertEqual(timeout.payload["timeout_policy"], "liu_runtime_phase_relative_timeout_v2")
        self.assertEqual(fixture.configuration.VIEW_TIMEOUT_POLICY_VERSION, "liu_zyzzyva_view_timeout_v1")

    def test_insufficient_replies_carry_speculative_value_and_resume(self):
        fixture = self.insufficient_reply_fixture()
        fixture.run_to_all_observed()
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        self.assertEqual(finality.finality_path, "zyzzyva_view_change_fast_all_replica_replies")
        self.assertTrue(
            any(
                record.event_type == "resumed_path" and record.status == "speculative_reorder"
                for record in LiuRuntimeInstrumentationCollector.view_changes
            )
        )
        self.assertTrue(
            any(
                record.event_type == "recovery_timeout" and record.status == "fired"
                for record in LiuRuntimeInstrumentationCollector.view_changes
            )
        )

    def test_deterministic_primary_rotation_duplicate_and_stale_evidence(self):
        fixture = LiuZyzzyvaRuntimeFixture()
        protocol = fixture.nodes[2].cp
        self.assertEqual([protocol.roles_for(1, view).primary_id for view in range(5)], [0, 2, 3, 0, 2])
        item = evidence(0)
        sender = fixture.nodes[0].cp
        message = Messages.send_view_change(sender, 2, 1.0, sender.timeout_identity(), item, None)
        self.assertEqual(Transitions._register_view_change(protocol, item, None, message), "handled")
        self.assertEqual(Transitions._register_view_change(protocol, item, None, message), "handled")
        self.assertEqual(tuple(protocol.state.view_change_evidence[1]), (0,))
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].status, "duplicate_ignored")

    def test_stale_timeout_and_old_view_messages_are_ignored_after_transition(self):
        fixture = self.silent_primary_fixture()
        old_timeout = next(
            item[1]
            for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == Messages.VIEW_TIMEOUT and item[1].actor.id == 0
        )
        fixture.run_until(lambda: fixture.nodes[0].cp.state.current_view == 1)
        self.assertEqual(Transitions.view_timeout(old_timeout.actor.cp, old_timeout), "handled")
        self.assertEqual(handle_event(fixture.delayed_old_request), "invalid")
        sender = fixture.nodes[0].cp
        stale = evidence(0)
        stale_message = Messages.send_view_change(sender, 2, 25.0, sender.timeout_identity(), stale, None)
        self.assertEqual(Transitions.receive_view_change(stale_message.actor.cp, stale_message), "handled")
        self.assertEqual(fixture.nodes[0].cp.state.current_view, 1)
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].status, "ignored")

    def test_observer_never_participates_in_view_change(self):
        fixture = self.silent_primary_fixture()
        fixture.run_to_all_observed()
        self.assertFalse(any(record.node_id == 4 for record in LiuRuntimeInstrumentationCollector.view_changes))
        self.assertFalse(
            any(
                record.message_type in (Messages.VIEW_CHANGE, Messages.NEW_VIEW)
                and (record.sender_id == 4 or record.receiver_id == 4)
                for record in LiuRuntimeInstrumentationCollector.protocol_messages
            )
        )

    def test_commit_certificate_and_lock_survive_view_transition(self):
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=100.0, validator_count=5)
        fixture.run_until(
            lambda: all(
                fixture.nodes[node_id].cp.state.phase is LiuZyzzyvaPhase.SPECULATIVELY_EXECUTED
                for node_id in fixture.nodes[1].cp.state.replica_ids
            )
        )
        fixture.queue.prio_queue.pq[:] = [
            item for item in fixture.queue.prio_queue.pq if item[1].payload["type"] != Messages.SPECULATIVE_REPLY
        ]
        heapq.heapify(fixture.queue.prio_queue.pq)
        identity = fixture.nodes[0].cp.state.block_identity
        block = fixture.nodes[0].cp.state.block.copy()
        commit = ZyzzyvaCommitCertificate(0, 1, 0, identity.block_digest, identity.parent_digest, (0, 2, 3), 3, 10.0)
        old_local_commit = Messages.send_local_commit(fixture.nodes[0].cp, 1, 9.0, identity, commit)
        old_index = next(
            index for index, item in enumerate(fixture.queue.prio_queue.pq) if item[1] is old_local_commit
        )
        del fixture.queue.prio_queue.pq[old_index]
        heapq.heapify(fixture.queue.prio_queue.pq)
        for sender_id in (0, 3, 4):
            sender = fixture.nodes[sender_id].cp
            sender.state.recovery_commit_certificate = commit
            sender.state.locked_digest = identity.block_digest
            sender.state.locked_block = block.copy()
            item = evidence(sender_id, commit=commit, locked=identity.block_digest)
            Messages.send_view_change(sender, 2, 10.0, identity, item, block)
        fixture.run_until(lambda: fixture.nodes[0].cp.state.current_view == 1)
        self.assertEqual(fixture.nodes[0].cp.state.locked_digest, identity.block_digest)
        self.assertEqual(fixture.nodes[0].cp.state.recovery_commit_certificate, commit)
        self.assertEqual(handle_event(old_local_commit), "invalid")
        fixture.run_to_all_observed()
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        self.assertEqual(finality.finality_path, "zyzzyva_view_change_recovery_local_commit_quorum")
        self.assertTrue(
            any(
                record.event_type == "resumed_path" and record.status == "commit_certificate_recovery"
                for record in LiuRuntimeInstrumentationCollector.view_changes
            )
        )

    def test_locally_committed_replica_cannot_transition_or_double_commit(self):
        fixture = LiuZyzzyvaRuntimeFixture(timeout_s=20.0, validator_count=5)
        recovery = __import__("tests.test_liu_zyzzyva_runtime", fromlist=["LiuZyzzyvaRecoveryTests"])
        fixture = recovery.LiuZyzzyvaRecoveryTests.recovery_fixture()
        fixture.run_until(lambda: fixture.nodes[2].cp.state.local_committed_height == 1)
        protocol = fixture.nodes[2].cp
        digest = protocol.state.local_commit_digest
        items = (evidence(0), evidence(2), evidence(3))
        certificate = ZyzzyvaNewViewCertificate(0, 1, 1, 2, items, 3, 2, None, None, None, None, None, 30.0)
        self.assertFalse(protocol.transition_to_view(1, certificate, 30.0, "manual"))
        self.assertEqual(protocol.state.local_commit_digest, digest)
        self.assertEqual(protocol.state.local_committed_height, 1)

    def test_unsafe_new_view_certificate_is_rejected(self):
        fixture = LiuZyzzyvaRuntimeFixture(validator_count=5)
        items = (evidence(0), evidence(2), evidence(3))
        certificate = ZyzzyvaNewViewCertificate(0, 1, 1, 2, items, 3, 2, None, None, None, None, None, 1.0)
        object.__setattr__(certificate, "safe_block_digest", "f" * 64)
        sender = fixture.nodes[2].cp
        event = Messages.send_new_view(sender, 0, 1.0, sender.timeout_identity(), certificate, None)
        self.assertEqual(Transitions.receive_new_view(event.actor.cp, event), "invalid")
        self.assertEqual(event.actor.cp.state.current_view, 0)
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].status, "rejected")

    def test_view_change_execution_is_deterministic_and_consumes_no_global_rng(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            fixture = self.insufficient_reply_fixture()
            fixture.run_to_all_observed()
            return before, random.getstate(), fixture.projection(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2:], second[2:])


if __name__ == "__main__":
    unittest.main()
