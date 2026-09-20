import copy
from dataclasses import FrozenInstanceError, replace
from collections import deque
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from collections import Counter
from Liu import EpochConfiguration, LinkStateMatrix, LiuConsensusProtocol, ThreatScenario
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot, run_single_action,
)
from Chain.Consensus.LiuRuntime.LiuQuorum.AdmissibleTimeout import estimated_slowest_required_round_trip_s
from Chain.Consensus.LiuRuntime.LiuQuorum.Configuration import LiuQuorumRuntimeConfiguration
from Chain.Consensus.LiuRuntime.Processing.ProcessingCostModel import LiuProcessingCostModel
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Chain.Consensus.LiuRuntime.Transport.MessageSizePolicy import LiuPaperMessageSizePolicy
from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.DigitalTwinPaired import evaluate_des_observed
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters


from Chain.Block import Block
from Chain.Node import Node
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.ValidatorSet import ValidatorSet
from Chain.Network import Network
from Chain.Consensus.LiuRuntime.Common.Certificates import LiuQuorumReplyCertificate
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.Roles import LiuQuorumRoles
from Chain.Consensus.LiuRuntime.LiuQuorum.Configuration import (
    LiuQuorumRuntimeConfiguration,
    ReplicaFault,
    ReplicaFaultPolicy,
)
from Chain.Consensus.LiuRuntime.LiuQuorum.Protocol import LiuQuorum
from Chain.Consensus.LiuRuntime.LiuQuorum.State import LiuQuorumPhase
from Chain.Consensus.LiuRuntime.Processing.NodeComputeQueue import NodeComputeQueue
from Engine.EventQueue import Queue
from Engine.Handler import handle_event
from Liu import (
    AnalyticalConsensusInput,
    EpochConfiguration,
    LinkStateMatrix,
    LiuConsensusProtocol,
    LiuQuorumAnalyticalModel,
    LiuTransmissionUnitPolicy,
    ThreatScenario,
)
from Parameters import Parameters
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector

K = 4
ALPHA = 2_000_000.0
BETA = 1_000_000.0
QUORUM_TIMEOUT_POLICY_VERSION = "liu_quorum_state_action_timeout_v1"

_N_QUORUM = 5
_CHI_QUORUM = 200
_LAMBDA_QUORUM = 0.6
_STAKES_QUORUM = tuple(1.0 for _ in range(_N_QUORUM))
_CAPS_QUORUM = tuple(1.0 for _ in range(_N_QUORUM))
_POS_QUORUM = tuple((0.1 * i, 0.0) for i in range(_N_QUORUM))


def epoch1_message_counts():
    return Counter(
        m.message_type
        for m in LiuRuntimeInstrumentationCollector.protocol_messages
        if m.epoch_id == 1
    )


def run_golden(faulty_ids=(), link_scale=1.0, capability_scale=1.0):
    link_rows = tuple(
        tuple(None if i == j else 10.0 * link_scale for j in range(_N_QUORUM))
        for i in range(_N_QUORUM)
    )
    caps = tuple(c * capability_scale for c in _CAPS_QUORUM)
    snapshot = LiuSingleActionSnapshot(
        node_count=_N_QUORUM,
        transaction_size_bytes=float(_CHI_QUORUM),
        stakes_tokens=_STAKES_QUORUM,
        capabilities_ghz=caps,
        positions_km=_POS_QUORUM,
        link_rows_mbps=link_rows,
        faulty_node_ids=faulty_ids,
        epoch0_validator_ids=tuple(range(K)),
    )
    action = LiuAction(_N_QUORUM, K, tuple(range(K)), LiuConsensusProtocol.LIU_QUORUM, 0.2, 1.0)
    geo = ContinuousSpatialIntensityModel(
        planar_gradient_intensity(K, _LAMBDA_QUORUM), K,
        lambda_form=f"planar_gradient s={_LAMBDA_QUORUM}",
    )
    params = LiuReferenceParameters(
        signature_verification_cycles_alpha=2_000_000.0,
        mac_operation_cycles_beta=1_000_000.0,
        network_timeout_s=100.0,
        finality_multiplier_omega=6.0,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=_CHI_QUORUM,
        recovery_delay_s=0.05,
    )
    return run_single_action(
        snapshot, action, geo, reference_params=params,
        stake_gini_threshold=0.2, geographic_gini_threshold=0.3,
        finality_multiplier_omega=6.0, signature_cycles_alpha=2_000_000.0,
        mac_cycles_beta=1_000_000.0, offered_workload_tx=2000,
        workload_tx_size_mb=_CHI_QUORUM / 1_000_000.0, max_events=2_000_000,
    )


class LiuQuorumRuntimeFixture:
    def __init__(self, fault=None, links=None):
        Parameters.simulation = {"event_id": 0, "events": {}}
        Parameters.application = {"Nn": 5, "transaction_model": "global"}
        Parameters.data = {"base_block_size": 0.0}
        Parameters.network = {"gossip": True}
        InstrumentationCollector.reset()
        LiuRuntimeInstrumentationCollector.reset()
        TransactionFactory.global_mempool = deque([Transaction(0, 11, 0.0, 1.0)])
        TransactionFactory.depth_removed = -1
        TransactionFactory.produced_tx = 0

        self.queue = Queue()
        self.validator_set = ValidatorSet((0, 1, 2, 3))
        self.profiles = NodeProfileSet(tuple(NodeProfile(node_id, None, 1.0) for node_id in range(5)))
        self.links = links or LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 10.0 for j in range(5)) for i in range(5))
        )
        self.epoch = EpochConfiguration(
            0,
            self.validator_set,
            LiuConsensusProtocol.LIU_QUORUM,
            1.0,
            0.5,
        )
        self.epoch_context = LiuRuntimeEpochContext(
            self.epoch,
            self.profiles,
            self.links,
            ThreatScenario(5, ()),
        )
        faults = () if fault is None else (fault,)
        self.configuration = LiuQuorumRuntimeConfiguration(
            self.epoch_context,
            signature_cycles_alpha=1_000_000_000.0,
            mac_cycles_beta=1_000_000_000.0,
            request_timeout_s=10.0,
            propagation_delay_s=0.0,
            replica_faults=faults,
        )
        self.nodes = [Node(node_id, self.queue, self.validator_set, self.profiles.profile_for(node_id)) for node_id in range(5)]
        Network.nodes = self.nodes
        Network.received = {node: set() for node in self.nodes}
        genesis = Block(depth=0, id=7, previous=-1, size=0.0)
        genesis.extra_data = {"configuration_depth": 0, "round": -1}
        for node in self.nodes:
            node.blockchain = [genesis.copy()]
            node.reconfiguration_state.configuration = SimpleNamespace(block_size=1.0, block_time=0.5)
            node.cp = LiuQuorum(node, self.configuration)
        TransactionFactory.nodes = self.nodes
        for node in self.nodes:
            node.cp.init(0.0, 0)

    def step(self):
        event = self.queue.pop_next_event()
        return event, handle_event(event)

    def run_until(self, predicate, maximum=100):
        events = []
        for _ in range(maximum):
            if predicate():
                return events
            event, result = self.step()
            events.append((event, result))
        raise AssertionError("runtime fixture did not reach the requested condition")

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


class RuntimePrimitiveTests(unittest.TestCase):
    def test_k21_has_one_client_and_exactly_twenty_replicas(self):
        roles = LiuQuorumRoles.for_height(ValidatorSet(tuple(range(21))), 1)
        self.assertEqual(roles.client_id, 0)
        self.assertEqual(len(roles.replica_ids), 20)
        self.assertEqual(set(roles.replica_ids), set(range(1, 21)))

    def test_block_identity_is_deterministic_sha256_and_parent_bound(self):
        transactions = (Transaction(0, 1, 0.0, 0.25),)
        first = LiuBlockIdentity.create(0, 1, "a" * 64, 0, transactions)
        same = LiuBlockIdentity.create(0, 1, "a" * 64, 0, transactions)
        different_parent = LiuBlockIdentity.create(0, 1, "b" * 64, 0, transactions)
        self.assertEqual(first, same)
        self.assertEqual(len(first.block_digest), 64)
        self.assertNotEqual(first, different_parent)

    def test_cpu_queue_is_serial_and_non_mutating(self):
        queue = NodeComputeQueue()
        first = queue.reserve(2.0, 3.0)
        second = queue.reserve(3.0, 2.0)
        self.assertEqual((first.processing_start, first.processing_end), (2.0, 5.0))
        self.assertEqual((second.processing_start, second.processing_end), (5.0, 7.0))
        with self.assertRaises(FrozenInstanceError):
            first.processing_end = 9.0

    def test_certificate_requires_distinct_exact_replica_set(self):
        identity = LiuBlockIdentity(0, 1, "a" * 64, "b" * 64)
        certificate = LiuQuorumReplyCertificate(identity, 0, (3, 1, 2), 3, 1.0)
        self.assertEqual(certificate.signer_ids, (1, 2, 3))
        self.assertTrue(certificate.validate(0, (1, 2, 3), identity))
        with self.assertRaises(ValueError):
            LiuQuorumReplyCertificate(identity, 0, (1, 1, 2), 3, 1.0)

    def test_selected_malicious_validator_is_infeasible_for_f2_zero(self):
        validator_set = ValidatorSet((0, 1))
        profiles = NodeProfileSet((NodeProfile(0, None, 20.0), NodeProfile(1, None, 20.0)))
        links = LinkStateMatrix.from_rows(((None, 10.0), (10.0, None)))
        epoch = EpochConfiguration(0, validator_set, LiuConsensusProtocol.LIU_QUORUM, 1.0, 0.5)
        context = LiuRuntimeEpochContext(epoch, profiles, links, ThreatScenario(2, (1,)))
        with self.assertRaisesRegex(ValueError, "F\\^2=0"):
            LiuQuorumRuntimeConfiguration(context, 1.0, 1.0, 10.0)


class LiuQuorumRuntimeAcceptanceTests(unittest.TestCase):
    def test_fault_free_flow_counts_roles_finality_and_observation(self):
        fixture = LiuQuorumRuntimeFixture()
        fixture.run_to_all_observed()
        messages = LiuRuntimeInstrumentationCollector.protocol_messages
        requests = [record for record in messages if record.message_type == "lq_request"]
        replies = [record for record in messages if record.message_type == "lq_reply"]
        announcements = [record for record in messages if record.message_type == "lq_finalized_block"]

        self.assertEqual({record.sender_id for record in requests}, {0})
        self.assertEqual({record.receiver_id for record in requests}, {1, 2, 3})
        self.assertEqual({record.sender_id for record in replies}, {1, 2, 3})
        self.assertEqual({record.receiver_id for record in replies}, {0})
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(replies), 3)
        self.assertEqual(len(announcements), 4)
        self.assertEqual({record.receiver_id for record in announcements}, {1, 2, 3, 4})
        self.assertFalse(any(record.sender_id == 4 or record.receiver_id == 4 for record in requests + replies))

        finality = LiuRuntimeInstrumentationCollector.protocol_finalities
        certificate = LiuRuntimeInstrumentationCollector.certificates
        self.assertEqual(len(finality), 1)
        self.assertEqual(len(certificate), 1)
        self.assertEqual(certificate[0].signer_ids, (1, 2, 3))
        self.assertEqual(certificate[0].threshold, 3)
        self.assertEqual(finality[0].finality_time, max(record.arrival_at for record in replies))
        self.assertEqual([node.last_block.id for node in fixture.nodes], [fixture.nodes[0].last_block.id] * 5)
        self.assertEqual(fixture.nodes[4].cp.state.phase, LiuQuorumPhase.OBSERVER)

        for record in requests + replies:
            self.assertEqual(record.serialization_delay_s, 0.8)
            self.assertAlmostEqual(record.arrival_at - record.sent_at, 0.8)

    def test_duplicate_reply_does_not_increase_count(self):
        fixture = LiuQuorumRuntimeFixture()
        fixture.run_until(lambda: any(record.message_type == "lq_reply" for record in LiuRuntimeInstrumentationCollector.protocol_messages))
        reply_event = next(
            item[1]
            for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == "lq_reply"
        )
        handle_event(reply_event)
        self.assertEqual(len(fixture.nodes[0].cp.state.replies), 1)
        handle_event(reply_event)
        self.assertEqual(len(fixture.nodes[0].cp.state.replies), 1)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 0)

    def test_one_omitted_replica_prevents_finality(self):
        fixture = LiuQuorumRuntimeFixture(ReplicaFault(1, ReplicaFaultPolicy.OMIT))
        fixture.run_until(lambda: fixture.nodes[0].cp.state.phase is LiuQuorumPhase.FAILED)
        self.assertEqual(len(fixture.nodes[0].cp.state.replies), 2)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 0)

    def test_corrupt_reply_prevents_finality(self):
        fixture = LiuQuorumRuntimeFixture(ReplicaFault(2, ReplicaFaultPolicy.CORRUPT))
        fixture.run_until(lambda: fixture.nodes[0].cp.state.phase is LiuQuorumPhase.FAILED)
        self.assertEqual(len(fixture.nodes[0].cp.state.replies), 2)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 0)

    def test_delayed_required_reply_delays_finality(self):
        fixture = LiuQuorumRuntimeFixture(ReplicaFault(3, ReplicaFaultPolicy.DELAY, delay_s=2.0))
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        self.assertAlmostEqual(finality.consensus_latency_s, 6.6)

    def test_asymmetric_directed_rates_control_each_hop(self):
        rows = [[None if i == j else 10.0 for j in range(5)] for i in range(5)]
        rows[0][1] = 5.0
        rows[1][0] = 20.0
        fixture = LiuQuorumRuntimeFixture(links=LinkStateMatrix.from_rows(rows))
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        request = next(
            record
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == "lq_request" and record.receiver_id == 1
        )
        reply = next(
            record
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == "lq_reply" and record.sender_id == 1
        )
        self.assertEqual(request.rate_mbps, 5.0)
        self.assertEqual(request.serialization_delay_s, 1.6)
        self.assertEqual(reply.rate_mbps, 20.0)
        self.assertEqual(reply.serialization_delay_s, 0.4)

    def test_stale_epoch_reply_is_rejected(self):
        fixture = LiuQuorumRuntimeFixture()
        fixture.run_until(lambda: any(record.message_type == "lq_reply" for record in LiuRuntimeInstrumentationCollector.protocol_messages))
        reply_event = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lq_reply")
        reply_event.liu_context = replace(reply_event.liu_context, epoch_id=99)
        self.assertEqual(handle_event(reply_event), "invalid")
        self.assertEqual(len(fixture.nodes[0].cp.state.replies), 0)

    def test_parent_mismatch_announcement_is_rejected(self):
        fixture = LiuQuorumRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        announcement = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lq_finalized_block")
        receiver = announcement.actor
        announcement.payload["block"].previous = 999
        self.assertEqual(handle_event(announcement), "invalid")
        self.assertEqual(receiver.last_block.depth, 0)

    def test_no_global_rng_consumption_and_same_seed_determinism(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            fixture = LiuQuorumRuntimeFixture()
            fixture.run_to_all_observed()
            after = random.getstate()
            return before, after, fixture.projection(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2:], second[2:])

    def test_runtime_matches_si_analytical_oracle(self):
        fixture = LiuQuorumRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        measured = LiuRuntimeInstrumentationCollector.protocol_finalities[0].consensus_latency_s
        analytical = LiuQuorumAnalyticalModel().evaluate(
            AnalyticalConsensusInput(
                validator_ids=(0, 1, 2, 3),
                validator_count_k=4,
                protocol=LiuConsensusProtocol.LIU_QUORUM,
                client_validator_id=0,
                primary_validator_id=None,
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
                network_timeout_s=10.0,
                faulty_replica_count=0,
                zyzzyva_path=None,
                recovery_delay_s=0.0,
                transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
            )
        )
        self.assertAlmostEqual(measured, 4.6)
        self.assertAlmostEqual(measured, analytical.consensus_delay_s)


class OperationCountFidelityTests(unittest.TestCase):
    def test_per_replica_work_matches_eq_alpha_plus_2beta(self):
        w = LiuProcessingWork.quorum_replica_work()
        self.assertEqual((w.signature_operations, w.mac_operations), (1, 2))  # α + 2β


class QuorumGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_golden()
        cls.d = cls.result.to_dict()

    def test_quorum_finalizes_1000_transactions(self):
        self.assertEqual(self.d["des_observed"]["finalized_transactions"], 1000)

    def test_two_leg_message_pattern_no_primary_no_batching(self):
        by = epoch1_message_counts()
        self.assertEqual(by["lq_request"], K - 1)   # t1 client → each replica (no primary broadcast tier)
        self.assertEqual(by["lq_reply"], K - 1)     # t2 replicas → client
        # No PBFT/Zyzzyva message types leaked in.
        for absent in ("lp_pre_prepare", "lp_prepare", "lp_commit", "lz_order_request",
                       "lz_speculative_reply", "lz_commit_certificate", "lz_local_commit"):
            self.assertNotIn(absent, by)

    def test_c1_c2_c3_all_pass_at_f0_and_tolerated_faults_is_zero(self):
        des = self.d["des_observed"]
        self.assertTrue(des["c1_decentralization_passed"] and des["c2_finality_passed"] and des["c3_security_passed"])
        self.assertEqual(des["malicious_validator_count"], 0)
        self.assertEqual(des["tolerated_fault_count"], 0)   # F² = 0

    def test_reward_des_realized_uses_finalized_tps(self):
        des = self.d["des_observed"]
        self.assertEqual(des["reward_des_realized"], des["throughput_des_finalized"])
        self.assertAlmostEqual(
            des["throughput_des_finalized"],
            des["finalized_transactions"] / des["measurement_duration_s"],
            places=6,
        )

    def test_finality_semantics_t_f_equals_t_i_plus_t_c(self):
        des = self.d["des_observed"]
        self.assertAlmostEqual(des["t_f_des_s"], 1.0 + des["t_c_des_s"], places=6)


class F2ZeroRewardRegressionTests(unittest.TestCase):
    """F² = 0: any selected-faulty validator must zero the reward at eval time."""

    def _evaluate(self, *, malicious):
        return evaluate_des_observed(
            selected_validator_stakes=[25.0] * K,
            geographic_gini_lambda=0.1,
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            protocol=LiuConsensusProtocol.LIU_QUORUM,
            validator_count=K,
            malicious_validator_count=malicious,
            block_interval_s=1.0,
            finality_multiplier_omega=6.0,
            t_c_des_s=0.3,
            offered_transactions=1000,
            included_transactions=1000,
            finalized_transactions=1000,
            measurement_duration_s=1.3,
        )

    def test_f0_passes_c3(self):
        r = self._evaluate(malicious=0)
        self.assertTrue(r.c3_security_passed)
        self.assertEqual(r.tolerated_fault_count, 0)
        self.assertGreater(r.reward_des_realized, 0)

    def test_f1_fails_c3_and_reward_is_zero(self):
        r = self._evaluate(malicious=1)
        self.assertFalse(r.c3_security_passed)
        self.assertEqual(r.tolerated_fault_count, 0)     # F² = 0
        self.assertFalse(r.feasible)
        self.assertEqual(r.reward_des_realized, 0.0)

    def test_configuration_hard_rejects_malicious_selected_validator(self):
        # Belt-and-braces: the runtime Configuration itself refuses to construct
        # a Quorum epoch with any selected faulty validator (matches F² = 0).
        with self.assertRaisesRegex(ValueError, "F\\^2=0"):
            run_golden(faulty_ids=(0,))


class ExactCapacityAndCausalityTests(unittest.TestCase):
    def test_slower_links_do_not_decrease_t_c_des(self):
        fast = run_golden().to_dict()["des_observed"]["t_c_des_s"]
        slow = run_golden(link_scale=0.5).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreater(slow, fast)

    def test_lower_compute_does_not_decrease_t_c_des(self):
        base = run_golden().to_dict()["des_observed"]["t_c_des_s"]
        slower = run_golden(capability_scale=0.5).to_dict()["des_observed"]["t_c_des_s"]
        self.assertGreaterEqual(slower, base)

    def test_quorum_finalized_capacity_is_exactly_1000(self):
        self.assertEqual(run_golden().to_dict()["des_observed"]["finalized_transactions"], 1000)


class DeterminismTests(unittest.TestCase):
    def test_repeated_golden_run_reproduces_identical_paired_result(self):
        self.assertEqual(run_golden().to_dict(), run_golden().to_dict())


class Fixture:
    """A configurable N-node LiuQuorum height-1 harness."""

    def __init__(
        self,
        *,
        node_count=5,
        validator_ids=(0, 1, 2, 3),
        block_size_mb=8.0,
        block_interval_s=0.5,
        capabilities=None,
        links=None,
        request_timeout_s=5.0,
        timeout_safety_factor=2.0,
        timeout_maximum_s=320.0,
        faults=(),
    ):
        Parameters.simulation = {"event_id": 0, "events": {}}
        Parameters.application = {"Nn": node_count, "transaction_model": "global"}
        Parameters.data = {"base_block_size": 0.0}
        Parameters.network = {"gossip": True}
        InstrumentationCollector.reset()
        LiuRuntimeInstrumentationCollector.reset()
        # 200 B tx fits every tested block size; message sizes use the configured
        # action block size (S^B), so the tx payload never affects the timing.
        TransactionFactory.global_mempool = deque([Transaction(0, 11, 0.0, 0.0002)])
        TransactionFactory.depth_removed = -1
        TransactionFactory.produced_tx = 0

        self.queue = Queue()
        self.node_count = node_count
        self.validator_set = ValidatorSet(tuple(validator_ids))
        if capabilities is None:
            capabilities = [30.0] * node_count
        self.profiles = NodeProfileSet(
            tuple(NodeProfile(node_id, None, capabilities[node_id]) for node_id in range(node_count))
        )
        if links is None:
            links = LinkStateMatrix.from_rows(
                tuple(tuple(None if i == j else 100.0 for j in range(node_count)) for i in range(node_count))
            )
        self.links = links
        self.epoch = EpochConfiguration(
            0, self.validator_set, LiuConsensusProtocol.LIU_QUORUM, block_size_mb, block_interval_s
        )
        self.epoch_context = LiuRuntimeEpochContext(
            self.epoch, self.profiles, self.links, ThreatScenario(node_count, ())
        )
        self.configuration = LiuQuorumRuntimeConfiguration(
            self.epoch_context,
            signature_cycles_alpha=ALPHA,
            mac_cycles_beta=BETA,
            request_timeout_s=request_timeout_s,
            propagation_delay_s=0.0,
            replica_faults=tuple(faults),
            timeout_safety_factor=timeout_safety_factor,
            timeout_maximum_s=timeout_maximum_s,
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
            node.reconfiguration_state.configuration = SimpleNamespace(
                block_size=block_size_mb, block_time=block_interval_s
            )
            node.cp = LiuQuorum(node, self.configuration)
        TransactionFactory.nodes = self.nodes
        for node in self.nodes:
            node.cp.init(0.0, 0)

    @property
    def client(self):
        return self.nodes[self.validator_set.proposer_for(0)]

    @property
    def client_state(self):
        return self.client.cp.state

    def run_until(self, predicate, maximum=200):
        for _ in range(maximum):
            if predicate():
                return
            handle_event(self.queue.pop_next_event())
        raise AssertionError("fixture did not reach the requested condition")

    def _timeout_event(self):
        return next(
            (
                item[1]
                for item in self.queue.prio_queue.pq
                if item[1].payload.get("type") == "lq_request_timeout"
            ),
            None,
        )

    def run_to_request_dispatched(self):
        self.run_until(lambda: self._timeout_event() is not None)

    def finalities(self):
        return LiuRuntimeInstrumentationCollector.protocol_finalities

    def scheduled_timeout_time(self):
        self.run_to_request_dispatched()
        return self._timeout_event().time

    def request_sent_at(self):
        self.run_to_request_dispatched()
        return self.client_state.request_sent_at


def slow_pair_links(node_count, client_id, slow_replica_id, slow_rate, fast_rate=100.0):
    rows = [[None if i == j else fast_rate for j in range(node_count)] for i in range(node_count)]
    rows[client_id][slow_replica_id] = slow_rate
    rows[slow_replica_id][client_id] = slow_rate
    return LinkStateMatrix.from_rows(tuple(tuple(row) for row in rows))


class QuorumAdmissibleTimeoutTests(unittest.TestCase):
    def test_policy_version_label(self):
        self.assertEqual(QUORUM_TIMEOUT_POLICY_VERSION, "liu_quorum_state_action_timeout_v1")
        self.assertEqual(
            LiuQuorumRuntimeConfiguration.REQUEST_TIMEOUT_POLICY_VERSION,
            "liu_quorum_state_action_timeout_v1",
        )

    def test_estimate_matches_directed_round_trip_and_uses_slowest_replica(self):
        links = slow_pair_links(5, client_id=0, slow_replica_id=3, slow_rate=25.0)
        cost = LiuProcessingCostModel(ALPHA, BETA)
        policy = LiuPaperMessageSizePolicy()
        estimate = estimated_slowest_required_round_trip_s(
            link_state_matrix=links,
            capability_ghz=lambda node_id: 30.0,
            processing_cost=cost,
            message_size_policy=policy,
            propagation_delay_s=0.0,
            client_id=0,
            replica_ids=(1, 2, 3),
            block_size_mb=8.0,
        )
        proc = cost.duration_s(LiuProcessingWork.quorum_replica_work(), 30.0)
        expected = 8.0 * 8.0 / 25.0 + proc + 8.0 * 8.0 / 25.0
        self.assertAlmostEqual(estimate, expected)
        self.assertGreater(estimate, 5.0)  # the exact Stage-C-v2 boundary: 5.12 s > 5.0 s

    def test_exact_8mb_boundary_fails_under_fixed_5s_but_finalizes_under_policy(self):
        links = slow_pair_links(5, client_id=0, slow_replica_id=3, slow_rate=25.0)
        # Fixed 5 s reproduction: ceiling == floor freezes the timer at 5 s.
        fixed = Fixture(links=links, request_timeout_s=5.0, timeout_maximum_s=5.0)
        self.assertEqual(fixed.scheduled_timeout_time(), 5.0 + fixed.request_sent_at())
        fixed.run_until(lambda: fixed.client_state.phase is LiuQuorumPhase.FAILED)
        self.assertEqual(len(fixed.finalities()), 0)
        self.assertEqual(len(fixed.client_state.replies), 2)  # two fast replies arrived; the slow pair did not

        # State/action-aware policy: the same action now finalizes.
        policy = Fixture(links=links, request_timeout_s=5.0, timeout_safety_factor=2.0)
        estimate = estimated_slowest_required_round_trip_s(
            link_state_matrix=links,
            capability_ghz=policy.epoch_context.capability_ghz,
            processing_cost=policy.nodes[0].cp.processing_cost,
            message_size_policy=policy.nodes[0].cp.message_size_policy,
            propagation_delay_s=0.0,
            client_id=0,
            replica_ids=(1, 2, 3),
            block_size_mb=8.0,
        )
        expected_timeout = policy.request_sent_at() + min(320.0, max(5.0, 2.0 * estimate))
        self.assertAlmostEqual(policy.scheduled_timeout_time(), expected_timeout)
        policy.run_until(lambda: len(policy.finalities()) == 1)
        # All K-1 = 3 replies formed the certificate (state resets after finality).
        certificate = LiuRuntimeInstrumentationCollector.certificates[0]
        self.assertEqual(len(certificate.signer_ids), 3)
        self.assertAlmostEqual(policy.finalities()[0].consensus_latency_s, estimate)

    def test_small_block_keeps_frozen_floor_timeout(self):
        # 0.2 MB over 100 Mbps: estimate is tiny, so the 5 s floor is unchanged.
        fixture = Fixture(block_size_mb=0.2)
        self.assertAlmostEqual(
            fixture.scheduled_timeout_time(), fixture.request_sent_at() + 5.0
        )
        fixture.run_until(lambda: len(fixture.finalities()) == 1)

    def test_fast_link_large_block_finalizes(self):
        fixture = Fixture(block_size_mb=8.0)  # uniform 100 Mbps
        # 8 MB over 100 Mbps both ways = 1.28 s < 5 s floor -> timeout stays 5 s.
        self.assertAlmostEqual(
            fixture.scheduled_timeout_time(), fixture.request_sent_at() + 5.0
        )
        fixture.run_until(lambda: len(fixture.finalities()) == 1)

    def test_slow_paper_minimum_link_finalizes_with_scaled_timeout(self):
        # Paper-range worst case: 10 Mbps both ways, 8 MB -> 12.8 s round-trip.
        links = slow_pair_links(5, client_id=0, slow_replica_id=2, slow_rate=10.0)
        fixture = Fixture(links=links, request_timeout_s=5.0, timeout_safety_factor=2.0)
        self.assertGreater(
            fixture.scheduled_timeout_time() - fixture.request_sent_at(), 12.8
        )
        fixture.run_until(lambda: len(fixture.finalities()) == 1)
        self.assertAlmostEqual(fixture.finalities()[0].consensus_latency_s, 12.8, places=2)

    def test_capability_variation_increases_estimate(self):
        links = slow_pair_links(5, client_id=0, slow_replica_id=3, slow_rate=25.0)
        cost = LiuProcessingCostModel(ALPHA, BETA)
        policy = LiuPaperMessageSizePolicy()
        args = dict(
            link_state_matrix=links,
            processing_cost=cost,
            message_size_policy=policy,
            propagation_delay_s=0.0,
            client_id=0,
            replica_ids=(1, 2, 3),
            block_size_mb=8.0,
        )
        fast = estimated_slowest_required_round_trip_s(capability_ghz=lambda node_id: 30.0, **args)
        slow = estimated_slowest_required_round_trip_s(capability_ghz=lambda node_id: 10.0, **args)
        self.assertGreater(slow, fast)  # weaker CPUs add processing time to the estimate

    def test_all_replies_still_required_even_with_generous_timeout(self):
        # F=0: one omitted replica prevents finality; the client fails at the
        # computed timeout, which is strictly later than the honest round-trip.
        links = slow_pair_links(5, client_id=0, slow_replica_id=3, slow_rate=25.0)
        fixture = Fixture(links=links, faults=(ReplicaFault(1, ReplicaFaultPolicy.OMIT),))
        estimate = estimated_slowest_required_round_trip_s(
            link_state_matrix=links,
            capability_ghz=fixture.epoch_context.capability_ghz,
            processing_cost=fixture.nodes[0].cp.processing_cost,
            message_size_policy=fixture.nodes[0].cp.message_size_policy,
            propagation_delay_s=0.0,
            client_id=0,
            replica_ids=(1, 2, 3),
            block_size_mb=8.0,
        )
        # Capture the timer before it fires (after it fires it leaves the queue).
        timeout_delta = fixture.scheduled_timeout_time() - fixture.request_sent_at()
        self.assertGreater(timeout_delta, estimate)
        fixture.run_until(lambda: fixture.client_state.phase is LiuQuorumPhase.FAILED)
        self.assertEqual(len(fixture.finalities()), 0)
        self.assertEqual(len(fixture.client_state.replies), 2)

    def test_timeout_clamped_to_ceiling(self):
        config = _bare_config(timeout_safety_factor=2.0, timeout_maximum_s=8.0, request_timeout_s=5.0)
        self.assertEqual(config.request_timeout_for(100.0), 8.0)  # ceiling binds
        self.assertEqual(config.request_timeout_for(0.1), 5.0)  # floor binds
        self.assertAlmostEqual(config.request_timeout_for(3.0), 6.0)  # 2.0 * 3.0

    def test_determinism_and_no_global_rng(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            links = slow_pair_links(5, client_id=0, slow_replica_id=3, slow_rate=25.0)
            fixture = Fixture(links=links)
            fixture.run_until(lambda: len(fixture.finalities()) == 1)
            after = random.getstate()
            return before, after, fixture.finalities()[0].consensus_latency_s, fixture.scheduled_timeout_time()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])  # global RNG untouched
        self.assertEqual(first[2:], second[2:])  # identical timing across runs

    def test_configuration_validation(self):
        with self.assertRaisesRegex(ValueError, "safety factor"):
            _bare_config(timeout_safety_factor=0.5)
        with self.assertRaisesRegex(ValueError, "ceiling"):
            _bare_config(timeout_maximum_s=1.0, request_timeout_s=5.0)


def _bare_config(*, request_timeout_s=5.0, timeout_safety_factor=2.0, timeout_maximum_s=320.0):
    validator_set = ValidatorSet((0, 1, 2, 3))
    profiles = NodeProfileSet(tuple(NodeProfile(i, None, 30.0) for i in range(5)))
    links = LinkStateMatrix.from_rows(
        tuple(tuple(None if i == j else 100.0 for j in range(5)) for i in range(5))
    )
    epoch = EpochConfiguration(0, validator_set, LiuConsensusProtocol.LIU_QUORUM, 8.0, 0.5)
    context = LiuRuntimeEpochContext(epoch, profiles, links, ThreatScenario(5, ()))
    return LiuQuorumRuntimeConfiguration(
        context,
        ALPHA,
        BETA,
        request_timeout_s,
        0.0,
        (),
        timeout_safety_factor,
        timeout_maximum_s,
    )


if __name__ == "__main__":
    unittest.main()
