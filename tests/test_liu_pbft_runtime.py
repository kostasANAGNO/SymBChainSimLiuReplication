import copy
from collections import deque
from dataclasses import replace
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


from collections import Counter

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters


from Chain.Block import Block
from Chain.Node import Node
from Chain.NodeProfile import NodeProfile, NodeProfileSet
from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.ValidatorSet import ValidatorSet
from Chain.Network import Network
from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.Roles import LiuPBFTRoles
from Chain.Consensus.LiuRuntime.LiuPBFT.Configuration import LiuPBFTRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuPBFT.Protocol import LiuPBFT
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase
from Engine.EventQueue import Queue
from Engine.Handler import handle_event
from Liu import (
    AnalyticalConsensusInput,
    EpochConfiguration,
    LinkStateMatrix,
    LiuConsensusProtocol,
    LiuTransmissionUnitPolicy,
    PBFTAnalyticalModel,
    ThreatScenario,
)
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot, run_single_action, _assemble,
)
from Parameters import Parameters
from Utils.Instrumentation import InstrumentationCollector
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector

K = 21
_N_PBFT = 100
_CHI_PBFT = 200
_LAMBDA_PBFT = 0.6
_STAKES_PBFT = tuple(float(4 + (i % 10)) for i in range(_N_PBFT))
_CAPS_PBFT = tuple(float(10 + (i % 21)) for i in range(_N_PBFT))
_POS_PBFT = tuple((0.1 * (i % 10), 0.1 * (i // 10)) for i in range(_N_PBFT))
_LINKS_PBFT = tuple(
    tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(_N_PBFT))
    for i in range(_N_PBFT)
)


def golden_snapshot() -> LiuSingleActionSnapshot:
    return LiuSingleActionSnapshot(
        node_count=_N_PBFT,
        transaction_size_bytes=float(_CHI_PBFT),
        stakes_tokens=_STAKES_PBFT,
        capabilities_ghz=_CAPS_PBFT,
        positions_km=_POS_PBFT,
        link_rows_mbps=_LINKS_PBFT,
        faulty_node_ids=(),
        epoch0_validator_ids=tuple(range(K)),
    )


def golden_run():
    snapshot = golden_snapshot()
    action = LiuAction(_N_PBFT, K, tuple(range(K)), LiuConsensusProtocol.PBFT, 0.2, 1.0)
    geo = ContinuousSpatialIntensityModel(
        planar_gradient_intensity(K, _LAMBDA_PBFT), K,
        lambda_form=f"planar_gradient s={_LAMBDA_PBFT}",
    )
    params = LiuReferenceParameters(
        signature_verification_cycles_alpha=2_000_000.0,
        mac_operation_cycles_beta=1_000_000.0,
        network_timeout_s=100.0,
        finality_multiplier_omega=6.0,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=_CHI_PBFT,
        recovery_delay_s=0.05,
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
        workload_tx_size_mb=_CHI_PBFT / 1_000_000.0,
        max_events=2_000_000,
    )


class LiuPBFTRuntimeFixture:
    def __init__(self, request_timeout_s=100.0, block_interval_s=0.5, alpha=1_000_000_000.0, beta=1_000_000_000.0):
        Parameters.simulation = {"debugging_mode": False, "event_id": 0, "events": {}}
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
        self.links = LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 10.0 for j in range(5)) for i in range(5))
        )
        epoch = EpochConfiguration(0, self.validator_set, LiuConsensusProtocol.PBFT, 1.0, block_interval_s)
        self.epoch_context = LiuRuntimeEpochContext(epoch, self.profiles, self.links, ThreatScenario(5, ()))
        self.configuration = LiuPBFTRuntimeConfiguration(
            self.epoch_context,
            signature_cycles_alpha=alpha,
            mac_cycles_beta=beta,
            request_timeout_s=request_timeout_s,
            propagation_delay_s=0.0,
        )
        self.nodes = [Node(node_id, self.queue, self.validator_set, self.profiles.profile_for(node_id)) for node_id in range(5)]
        Network.nodes = self.nodes
        Network.received = {node: set() for node in self.nodes}
        genesis = Block(depth=0, id=7, previous=-1, size=0.0)
        genesis.extra_data = {"configuration_depth": 0, "round": -1}
        for node in self.nodes:
            node.blockchain = [genesis.copy()]
            node.reconfiguration_state.configuration = SimpleNamespace(block_size=1.0, block_time=block_interval_s)
            node.cp = LiuPBFT(node, self.configuration)
        TransactionFactory.nodes = self.nodes
        for node in self.nodes:
            node.cp.init(0.0, 0)

    def step(self):
        event = self.queue.pop_next_event()
        return event, handle_event(event)

    def run_until(self, predicate, maximum=500):
        events = []
        for _ in range(maximum):
            if predicate():
                return events
            event, result = self.step()
            events.append((event, result))
        raise AssertionError("PBFT fixture did not reach the requested condition")

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
                    tuple(tx.id for tx in block.transactions),
                    copy.deepcopy(block.extra_data),
                )
                for block in node.blockchain
            )
            for node in self.nodes
        )


class LiuPBFTQuorumPolicyTests(unittest.TestCase):
    def test_k21_reconstruction_is_f6_prepare14_commit14_reply7(self):
        validators = ValidatorSet(tuple(range(21)))
        profiles = NodeProfileSet(tuple(NodeProfile(node_id, None, 20.0) for node_id in range(21)))
        links = LinkStateMatrix.from_rows(
            tuple(tuple(None if i == j else 100.0 for j in range(21)) for i in range(21))
        )
        epoch = EpochConfiguration(0, validators, LiuConsensusProtocol.PBFT, 1.0, 0.5)
        config = LiuPBFTRuntimeConfiguration(
            LiuRuntimeEpochContext(epoch, profiles, links, ThreatScenario(21, ())),
            1.0,
            1.0,
            10.0,
        )
        roles = LiuPBFTRoles.for_height_view(validators, 1, 0)
        self.assertNotEqual(roles.client_id, roles.primary_id)
        self.assertEqual(len(roles.replica_ids), 20)
        self.assertEqual(config.tolerated_faults, 6)
        self.assertEqual(config.prepare_quorum, 14)
        self.assertEqual(config.commit_quorum, 14)
        self.assertEqual(config.reply_quorum, 7)
        self.assertEqual(config.view_change_quorum, 14)
        self.assertEqual(config.QUORUM_POLICY_VERSION, "liu_pbft_k_minus_one_replicas_v1")
        self.assertEqual(config.VIEW_CHANGE_QUORUM_POLICY_VERSION, "liu_pbft_view_change_quorum_v1")
        self.assertEqual(config.SAFE_VALUE_SELECTION_VERSION, "liu_pbft_safe_value_selection_v1")


class LiuPBFTRuntimeAcceptanceTests(unittest.TestCase):
    def test_normal_path_roles_traffic_quorums_and_finality(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_to_all_observed()
        messages = LiuRuntimeInstrumentationCollector.protocol_messages
        by_type = {
            kind: [record for record in messages if record.message_type == kind]
            for kind in ("lp_request", "lp_pre_prepare", "lp_prepare", "lp_commit", "lp_reply", "lp_finalized_block")
        }
        self.assertEqual(len(by_type["lp_request"]), 1)
        self.assertEqual((by_type["lp_request"][0].sender_id, by_type["lp_request"][0].receiver_id), (0, 1))
        self.assertEqual(len(by_type["lp_pre_prepare"]), 2)
        self.assertEqual(len(by_type["lp_prepare"]), 6)
        self.assertEqual(len(by_type["lp_commit"]), 6)
        self.assertEqual(len(by_type["lp_reply"]), 3)
        self.assertEqual(len(by_type["lp_finalized_block"]), 4)
        for kind in ("lp_prepare", "lp_commit"):
            participants = {record.sender_id for record in by_type[kind]} | {record.receiver_id for record in by_type[kind]}
            self.assertEqual(participants, {1, 2, 3})
        self.assertFalse(any(record.sender_id == 4 or record.receiver_id == 4 for kind in by_type for record in by_type[kind] if kind != "lp_finalized_block"))

        prepared = [record for record in LiuRuntimeInstrumentationCollector.certificates if record.certificate_type == "pbft_prepared"]
        commits = [record for record in LiuRuntimeInstrumentationCollector.certificates if record.certificate_type == "pbft_local_commit"]
        finality_certs = [
            record for record in LiuRuntimeInstrumentationCollector.certificates if record.certificate_type == "pbft_client_reply_quorum"
        ]
        self.assertEqual(len(prepared), 3)
        self.assertEqual(len(commits), 3)
        self.assertTrue(all(record.threshold == 2 and len(record.signer_ids) == 2 for record in prepared + commits))
        self.assertEqual(len(finality_certs), 1)
        self.assertEqual(finality_certs[0].threshold, 2)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 1)
        self.assertEqual([node.last_block.id for node in fixture.nodes], [fixture.nodes[0].last_block.id] * 5)
        self.assertEqual(fixture.nodes[4].cp.state.phase, LiuPBFTPhase.OBSERVER)

    def test_duplicate_and_wrong_digest_prepare_do_not_increase_votes(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: all(fixture.nodes[node_id].cp.state.sent_prepare for node_id in (1, 2, 3)))
        prepare = next(
            item[1]
            for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == "lp_prepare" and item[1].actor.id == 1
        )
        initial = len(fixture.nodes[1].cp.state.prepare_votes[prepare.payload["identity"].block_digest])
        self.assertIn(handle_event(prepare), ("handled", "new_state"))
        after_first = len(fixture.nodes[1].cp.state.prepare_votes[prepare.payload["identity"].block_digest])
        handle_event(prepare)
        self.assertEqual(len(fixture.nodes[1].cp.state.prepare_votes[prepare.payload["identity"].block_digest]), after_first)
        self.assertGreaterEqual(after_first, initial)

        wrong = copy.copy(prepare)
        wrong.payload = dict(prepare.payload)
        wrong_identity = replace(prepare.payload["identity"], block_digest="0" * 64)
        wrong.payload["identity"] = wrong_identity
        wrong.liu_context = replace(prepare.liu_context, block_identity=wrong_identity)
        self.assertEqual(handle_event(wrong), "invalid")
        self.assertNotIn("0" * 64, fixture.nodes[1].cp.state.prepare_votes)

    def test_duplicate_commit_and_one_local_commit_max(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: any(item[1].payload["type"] == "lp_commit" for item in fixture.queue.prio_queue.pq))
        commit = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lp_commit")
        target = commit.actor
        first = handle_event(commit)
        before = sum(len(votes) for votes in target.cp.state.commit_votes.values())
        second = handle_event(commit)
        self.assertIn(first, ("backlog", "handled", "new_state"))
        self.assertIn(second, ("backlog", "handled"))
        self.assertEqual(sum(len(votes) for votes in target.cp.state.commit_votes.values()), before)
        fixture.run_to_all_observed()
        local_commits = [
            record
            for record in LiuRuntimeInstrumentationCollector.phase_transitions
            if record.new_phase == "local_committed"
        ]
        self.assertEqual({record.node_id for record in local_commits}, {1, 2, 3})
        self.assertEqual(len(local_commits), 3)

    def test_commit_before_own_prepared_certificate_is_not_counted(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: any(item[1].payload["type"] == "lp_commit" for item in fixture.queue.prio_queue.pq))
        commit = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lp_commit")
        target = commit.actor
        if target.cp.state.prepared_certificate is None:
            self.assertEqual(handle_event(commit), "backlog")
            self.assertEqual(target.cp.state.commit_votes, {})
        else:
            other = next(node for node in fixture.nodes[1:4] if node.cp.state.prepared_certificate is None)
            commit.actor = other
            commit.receiver = other
            self.assertEqual(handle_event(commit), "backlog")
            self.assertEqual(other.cp.state.commit_votes, {})

    def test_stale_epoch_and_view_votes_are_rejected(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: all(fixture.nodes[node_id].cp.state.sent_prepare for node_id in (1, 2, 3)))
        prepare = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lp_prepare")
        stale_epoch = copy.copy(prepare)
        stale_epoch.liu_context = replace(prepare.liu_context, epoch_id=99)
        stale_view = copy.copy(prepare)
        stale_view.liu_context = replace(prepare.liu_context, view=1)
        self.assertEqual(handle_event(stale_epoch), "invalid")
        self.assertEqual(handle_event(stale_view), "invalid")

    def test_client_finality_occurs_at_f_plus_one_matching_reply(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        replies = sorted(
            record.arrival_at
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == "lp_reply"
        )
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities[0].finality_time, replies[1])
        self.assertEqual(len(fixture.nodes[0].cp.state.client_reply_votes), 0)  # next-height state is fresh

    def test_parent_mismatch_finalized_announcement_is_rejected(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        announcement = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == "lp_finalized_block")
        receiver = announcement.actor
        announcement.payload["block"].previous = 999
        self.assertEqual(handle_event(announcement), "invalid")
        self.assertEqual(receiver.last_block.depth, 0)

    def test_same_seed_is_deterministic_and_protocol_consumes_no_global_rng(self):
        def run_once():
            random.seed(1837413)
            before = random.getstate()
            fixture = LiuPBFTRuntimeFixture()
            fixture.run_to_all_observed()
            return before, random.getstate(), fixture.projection(), LiuRuntimeInstrumentationCollector.deterministic_hash()

        first = run_once()
        second = run_once()
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2:], second[2:])

    def test_controlled_analytical_comparison_has_causal_decomposition(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: len(LiuRuntimeInstrumentationCollector.protocol_finalities) == 1)
        des_tc = LiuRuntimeInstrumentationCollector.protocol_finalities[0].consensus_latency_s
        des_td = 5 * 0.8
        des_tv = des_tc - des_td
        analytical = PBFTAnalyticalModel().evaluate(
            AnalyticalConsensusInput(
                validator_ids=(0, 1, 2, 3),
                validator_count_k=4,
                protocol=LiuConsensusProtocol.PBFT,
                client_validator_id=0,
                primary_validator_id=1,
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
                zyzzyva_path=None,
                recovery_delay_s=0.0,
                transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
            )
        )
        self.assertAlmostEqual(analytical.delivery_delay_s, 4.0)
        self.assertAlmostEqual(analytical.validation_delay_s, 15.0)
        self.assertAlmostEqual(analytical.consensus_delay_s, 19.0)
        self.assertAlmostEqual(des_td, analytical.delivery_delay_s)
        self.assertGreater(des_tv, analytical.validation_delay_s)
        self.assertGreater(des_tc, analytical.consensus_delay_s)


class QuorumThresholdTests(unittest.TestCase):
    def test_pbft_quorum_thresholds_for_k21(self):
        snap = golden_snapshot()
        action = LiuAction(100, K, tuple(range(K)), LiuConsensusProtocol.PBFT, 0.2, 1.0)
        assembled = _assemble(
            snap, action, signature_cycles_alpha=2e6, mac_cycles_beta=1e6,
            request_timeout_s=100.0, propagation_delay_s=0.0,
        )
        cfg = LiuPBFTRuntimeConfiguration(assembled.epoch0_context, 2e6, 1e6, 100.0)
        # client = block producer; replicas = K-1; f = floor((K-1)/3).
        self.assertEqual(cfg.validator_count_k, K)
        self.assertEqual(cfg.replica_count, K - 1)          # 20
        self.assertEqual(cfg.tolerated_faults, (K - 1) // 3)  # 6
        self.assertEqual(cfg.prepare_quorum, (K - 1) - 6)     # 14 (= n-f, >= 2f+1=13, safe)
        self.assertEqual(cfg.commit_quorum, (K - 1) - 6)      # 14
        self.assertEqual(cfg.reply_quorum, 6 + 1)             # 7
        self.assertGreaterEqual(cfg.prepare_quorum, 2 * cfg.tolerated_faults + 1)  # BFT-safe


class RoleAssignmentTests(unittest.TestCase):
    def test_client_and_primary_are_distinct_and_correct(self):
        validators = ValidatorSet(tuple(range(K)))
        roles = LiuPBFTRoles.for_height_view(validators, height=2, view=0)
        self.assertNotEqual(roles.client_id, roles.primary_id)          # p != c (paper)
        self.assertIn(roles.primary_id, roles.replica_ids)
        self.assertNotIn(roles.client_id, roles.replica_ids)
        self.assertEqual(len(roles.replica_ids), K - 1)
        self.assertNotIn(roles.primary_id, roles.backup_ids)


class OperationCountFidelityTests(unittest.TestCase):
    def test_pbft_cpu_operation_counts_match_eq15_at_m1_f0(self):
        primary_sig = LiuProcessingWork.pbft_primary_request_work().signature_operations
        primary_mac = (
            LiuProcessingWork.pbft_primary_request_work().mac_operations
            + LiuProcessingWork.pbft_prepare_quorum_work(K - 1).mac_operations
            + LiuProcessingWork.pbft_commit_quorum_work(K - 1).mac_operations
        )
        replica_sig = LiuProcessingWork.pbft_backup_preprepare_work().signature_operations
        replica_mac = (
            LiuProcessingWork.pbft_backup_preprepare_work().mac_operations
            + LiuProcessingWork.pbft_prepare_quorum_work(K - 1).mac_operations
            + LiuProcessingWork.pbft_commit_quorum_work(K - 1).mac_operations
        )
        M, f = 1, 0
        self.assertEqual((primary_sig, primary_mac), (M, 2 * M + 4 * (K + f - 1)))  # (1, 82)
        self.assertEqual((replica_sig, replica_mac), (M, M + 4 * (K + f - 1)))       # (1, 81)

    def test_m3_batching_is_not_represented_at_event_level(self):
        # Documents outcome C: DES processes one block per instance (M=1), not M=3.
        primary_mac_m1 = (
            LiuProcessingWork.pbft_primary_request_work().mac_operations
            + LiuProcessingWork.pbft_prepare_quorum_work(K - 1).mac_operations
            + LiuProcessingWork.pbft_commit_quorum_work(K - 1).mac_operations
        )
        paper_primary_mac_m3 = 2 * 3 + 4 * (K - 1)
        self.assertNotEqual(primary_mac_m1, paper_primary_mac_m3)


class MessagePhaseProgressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = golden_run()
        cls.msgs = [
            m for m in LiuRuntimeInstrumentationCollector.protocol_messages
            if m.epoch_id == 1 and m.height == 2
        ]
        cls.by = Counter(m.message_type for m in cls.msgs)

    def test_all_five_phases_plus_finality_present(self):
        self.assertEqual(self.by["lp_request"], 1)          # single client request
        self.assertGreaterEqual(self.by["lp_pre_prepare"], 1)
        self.assertGreater(self.by["lp_prepare"], 0)
        self.assertGreater(self.by["lp_commit"], 0)
        self.assertGreater(self.by["lp_reply"], 0)
        finals = [
            f for f in LiuRuntimeInstrumentationCollector.protocol_finalities
            if f.epoch_id == 1 and f.height == 2
        ]
        self.assertEqual(len(finals), 1)

    def test_every_phase_message_carries_block_payload_s_b(self):
        for m in self.msgs:
            self.assertEqual(m.payload_size_mb, 0.2)  # S_B on every leg (matches Eq.15 M*S_B, M=1)


class BlockCapacityExactTests(unittest.TestCase):
    def test_pre_fix_float_accumulation_would_have_yielded_999(self):
        # Historical: the un-fixed greedy fill with float MB values under-filled by one.
        size, n = 0.0, 0
        while size + 0.0002 <= 0.2:
            size += 0.0002
            n += 1
        self.assertEqual(n, 999)  # pure-Python demonstration of the cause of the old bug
        # Exact integer-byte accounting must admit 1000.
        self.assertEqual(int(200_000 // 200), 1000)

    def test_golden_run_finalizes_1000_after_exact_byte_fix(self):
        self.assertEqual(golden_run().to_dict()["des_observed"]["finalized_transactions"], 1000)


class SingleEpochThroughputTests(unittest.TestCase):
    def test_duration_is_interval_plus_consensus_and_tps_formula(self):
        d = golden_run().to_dict()["des_observed"]
        # window = T_I (block-production wait) + T_C,DES (consensus)
        self.assertAlmostEqual(d["measurement_duration_s"], 1.0 + d["t_c_des_s"], places=6)
        self.assertAlmostEqual(
            d["throughput_des_finalized"],
            d["finalized_transactions"] / d["measurement_duration_s"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
