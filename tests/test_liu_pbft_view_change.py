import copy
from dataclasses import FrozenInstanceError
import sys
import unittest
from pathlib import Path


from dataclasses import replace

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Chain.Consensus.LiuRuntime.Common.Certificates import (
    PBFTNewViewCertificate,
    PBFTPreparedCertificate,
    PBFTViewChangeEvidence,
    select_pbft_safe_prepared,
)
from Chain.Consensus.LiuRuntime.LiuPBFT import Messages, Transitions
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase
from Engine.Handler import handle_event
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector
from tests.test_liu_pbft_runtime import LiuPBFTRuntimeFixture


def prepared(view: int, digest: str, parent: str = "parent") -> PBFTPreparedCertificate:
    return PBFTPreparedCertificate(0, 1, view, digest, parent, (1, 2), 2, float(view + 1))


def evidence(sender: int, target_view: int, certificate=None, commit=None) -> PBFTViewChangeEvidence:
    return PBFTViewChangeEvidence(0, 1, target_view, sender, certificate, commit)

class PBFTViewChangeCertificateTests(unittest.TestCase):
    def test_exact_distinct_quorum_and_immutable_evidence(self):
        first = evidence(1, 1)
        second = evidence(2, 1)
        certificate = PBFTNewViewCertificate(0, 1, 1, (second, first), 2, None, 3.0)
        self.assertEqual(certificate.signer_ids, (1, 2))
        self.assertEqual(certificate.QUORUM_POLICY_VERSION, "liu_pbft_view_change_quorum_v1")
        self.assertEqual(certificate.SAFE_VALUE_POLICY_VERSION, "liu_pbft_safe_value_selection_v1")
        with self.assertRaises(ValueError):
            PBFTNewViewCertificate(0, 1, 1, (first,), 2, None, 3.0)
        with self.assertRaises(ValueError):
            PBFTNewViewCertificate(0, 1, 1, (first, first), 2, None, 3.0)
        with self.assertRaises(FrozenInstanceError):
            first.target_view = 2

    def test_highest_originating_view_prepared_certificate_is_selected(self):
        low = prepared(0, "a" * 64)
        high = prepared(1, "b" * 64)
        selected, error = select_pbft_safe_prepared((evidence(1, 2, low), evidence(2, 2, high)))
        self.assertIsNone(error)
        self.assertEqual(selected, high)
        certificate = PBFTNewViewCertificate(
            0,
            1,
            2,
            (evidence(1, 2, low), evidence(2, 2, high)),
            2,
            high,
            4.0,
        )
        self.assertTrue(certificate.validate((1, 2, 3), 2, 2, 2))

    def test_conflicting_highest_view_prepared_certificates_are_a_safety_error(self):
        left = prepared(0, "a" * 64)
        right = prepared(0, "b" * 64)
        selected, error = select_pbft_safe_prepared((evidence(1, 1, left), evidence(2, 1, right)))
        self.assertIsNone(selected)
        self.assertEqual(error, "conflicting_highest_view_prepared_certificates")
        with self.assertRaisesRegex(ValueError, "conflicting_highest_view"):
            PBFTNewViewCertificate(
                0,
                1,
                1,
                (evidence(1, 1, left), evidence(2, 1, right)),
                2,
                left,
                3.0,
            )

    def test_certificate_serialization_and_hash_are_deterministic(self):
        items = (evidence(1, 1), evidence(2, 1))
        first = PBFTNewViewCertificate(0, 1, 1, items, 2, None, 3.0)
        second = PBFTNewViewCertificate(0, 1, 1, tuple(reversed(items)), 2, None, 3.0)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())


class PBFTViewChangeRuntimeTests(unittest.TestCase):
    @staticmethod
    def delayed_primary_fixture():
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=30.0)
        start_event, _ = fixture.step()
        assert start_event.payload["type"] == Messages.START_REQUEST
        request = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.REQUEST
        )
        fixture.queue.remove_event(request)
        return fixture

    def test_silent_primary_triggers_safe_view_change_and_recovery(self):
        fixture = self.delayed_primary_fixture()
        fixture.run_to_all_observed()
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        new_view = [
            record for record in LiuRuntimeInstrumentationCollector.certificates if record.certificate_type == "pbft_new_view"
        ]
        self.assertEqual(len(new_view), 1)
        self.assertEqual((new_view[0].threshold, new_view[0].signer_ids), (2, (1, 2)))
        self.assertEqual(finality.finality_path, "pbft_view_change_reply_quorum")
        self.assertAlmostEqual(finality.finality_time, 52.3)
        self.assertGreater(finality.consensus_latency_s, 20.2)
        self.assertEqual({node.last_block.id for node in fixture.nodes}, {fixture.nodes[0].last_block.id})
        self.assertTrue(
            any(record.event_type == "new_view_received" and record.status == "accepted" for record in LiuRuntimeInstrumentationCollector.view_changes)
        )

    def test_accepted_new_view_certificate_is_relayed_once_to_lagging_replica(self):
        fixture = self.delayed_primary_fixture()
        fixture.run_until(lambda: any(node.cp.state.new_view_certificate is not None for node in fixture.nodes[:4]))
        primary = next(node.cp for node in fixture.nodes[:4] if node.cp.state.new_view_certificate is not None)
        certificate = primary.state.new_view_certificate
        lagging_evidence = certificate.evidence[0]
        before = fixture.queue.size()
        self.assertEqual(
            Transitions._register_view_change(primary, lagging_evidence, None, 40.0, "catchup-test"),
            "handled",
        )
        self.assertEqual(fixture.queue.size(), before + 1)
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-2].event_type, "new_view_catchup_relay")
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-2].status, "sent")
        self.assertEqual(
            Transitions._register_view_change(primary, lagging_evidence, None, 40.1, "catchup-test-duplicate"),
            "handled",
        )
        self.assertEqual(fixture.queue.size(), before + 1)
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-2].status, "duplicate_ignored")

    def test_future_preprepare_is_rejected_then_exact_authorized_proposal_is_retransmitted(self):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=100.0)
        fixture.run_until(lambda: fixture.nodes[2].cp.state.block_identity is not None)
        primary = fixture.nodes[2].cp
        certificate = PBFTNewViewCertificate(
            0, 1, 1, (evidence(1, 1), evidence(2, 1)), 2, None, 5.0
        )
        self.assertTrue(primary.transition_to_view(1, certificate, 5.0, "manual-primary", None))
        client_event = Messages.send_new_view(primary, 0, 5.0, primary.timeout_identity(), certificate, None)
        fixture.queue.remove_event(client_event)
        self.assertEqual(Transitions.receive_new_view(client_event.actor.cp, client_event), "new_state")
        fixture.run_until(
            lambda: primary.state.current_view == 1 and primary.state.accepted_preprepare is not None
        )
        lagging = fixture.nodes[3].cp
        early = Messages.send_preprepare(primary, 3, 6.0, primary.state.block, primary.state.block_identity)
        fixture.queue.remove_event(early)
        self.assertEqual(Transitions.receive_preprepare(early.actor.cp, early), "invalid")
        self.assertEqual(lagging.state.current_view, 0)
        new_view = Messages.send_new_view(primary, 3, 6.1, primary.timeout_identity(), certificate, None)
        fixture.queue.remove_event(new_view)
        self.assertEqual(Transitions.receive_new_view(new_view.actor.cp, new_view), "new_state")
        before = fixture.queue.size()
        retransmitted = next(
            item[1] for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == Messages.PRE_PREPARE
            and item[1].creator.id == primary.node.id
            and item[1].actor.id == 3
            and item[1].liu_context.view == 1
        )
        fixture.queue.remove_event(retransmitted)
        self.assertEqual(Transitions.receive_preprepare(retransmitted.actor.cp, retransmitted), "new_state")
        self.assertEqual(fixture.queue.size(), before)
        self.assertFalse(primary.retransmit_authorized_proposal(3, 6.3))
        records = [
            record for record in LiuRuntimeInstrumentationCollector.view_changes
            if record.event_type == "new_view_proposal_retransmission"
        ]
        self.assertEqual([record.status for record in records[-2:]], ["sent", "duplicate_ignored"])

    def test_retransmission_requires_current_certificate_exact_value_and_replica_receiver(self):
        fixture = LiuPBFTRuntimeFixture()
        protocol = fixture.nodes[1].cp
        self.assertEqual(
            protocol.runtime_configuration.NEW_VIEW_PROPOSAL_DELIVERY_VERSION,
            "liu_pbft_new_view_proposal_delivery_v1",
        )
        self.assertFalse(protocol.retransmit_authorized_proposal(3, 1.0))
        self.assertFalse(protocol.retransmit_authorized_proposal(4, 1.0))

    def test_valid_higher_new_view_certificate_allows_direct_safe_catchup(self):
        fixture = LiuPBFTRuntimeFixture()
        certificate = PBFTNewViewCertificate(
            0, 1, 3, (evidence(1, 3), evidence(2, 3)), 2, None, 1.0
        )
        primary = fixture.nodes[1].cp  # replicas[3 % 3]
        event = Messages.send_new_view(primary, 2, 1.0, primary.timeout_identity(), certificate, None)
        self.assertEqual(Transitions.receive_new_view(event.actor.cp, event), "new_state")
        self.assertEqual(event.actor.cp.state.current_view, 3)
        self.assertEqual(event.actor.cp.state.highest_observed_view, 3)

    def test_two_missed_views_then_valid_view_three_certificate_reaches_finality(self):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=100.0)
        fixture.step()  # Populate the client's immutable pending request.
        old_request = next(item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.REQUEST)
        fixture.queue.remove_event(old_request)
        certificate = PBFTNewViewCertificate(
            0, 1, 3, (evidence(1, 3), evidence(2, 3)), 2, None, 1.0
        )
        primary = fixture.nodes[1].cp
        self.assertTrue(primary.transition_to_view(3, certificate, 1.0, "view-3-primary", None))
        for receiver_id in (0, 2, 3):
            event = Messages.send_new_view(primary, receiver_id, 1.0, primary.timeout_identity(), certificate, None)
            fixture.queue.remove_event(event)
            self.assertEqual(Transitions.receive_new_view(event.actor.cp, event), "new_state")
        fixture.run_to_all_observed()
        finalities = LiuRuntimeInstrumentationCollector.protocol_finalities
        self.assertEqual(len(finalities), 1)
        self.assertEqual(finalities[0].finality_path, "pbft_view_change_reply_quorum")
        accepted = [
            record for record in LiuRuntimeInstrumentationCollector.view_changes
            if record.event_type == "new_view_received" and record.status == "accepted"
        ]
        self.assertEqual({record.target_view for record in accepted}, {3})

    def test_observed_view_is_diagnostic_and_does_not_authorize_transition(self):
        fixture = LiuPBFTRuntimeFixture()
        target_primary = fixture.nodes[2].cp
        self.assertEqual(
            Transitions._register_view_change(target_primary, evidence(1, 1), None, 1.0, "single-evidence"),
            "handled",
        )
        self.assertEqual(target_primary.state.highest_observed_view, 1)
        self.assertEqual(target_primary.state.current_view, 0)
        self.assertEqual(
            Transitions._register_view_change(target_primary, evidence(1, 1), None, 1.1, "duplicate"),
            "handled",
        )
        self.assertEqual(target_primary.state.highest_observed_view, 1)
        self.assertEqual(target_primary.state.current_view, 0)

    def test_duplicate_evidence_does_not_increase_signer_count(self):
        fixture = LiuPBFTRuntimeFixture()
        primary = fixture.nodes[2].cp
        item = evidence(1, 1)
        identity = primary.timeout_identity()
        self.assertEqual(Transitions._register_view_change(primary, item, None, 1.0, "test"), "handled")
        self.assertEqual(Transitions._register_view_change(primary, item, None, 1.1, "test"), "handled")
        self.assertEqual(tuple(primary.state.view_change_evidence[1]), (1,))
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].status, "duplicate_ignored")
        self.assertEqual(identity.height, 1)

    def test_stale_view_change_and_stale_timeout_are_ignored_after_transition(self):
        fixture = self.delayed_primary_fixture()
        old_timeout = next(
            item[1]
            for item in fixture.queue.prio_queue.pq
            if item[1].payload["type"] == Messages.VIEW_TIMEOUT and item[1].actor.id == 1
        )
        fixture.run_until(lambda: fixture.nodes[1].cp.state.current_view == 1)
        sender = fixture.nodes[1].cp
        stale_view_change = Messages.send_view_change(
            sender,
            2,
            32.0,
            sender.timeout_identity(),
            PBFTViewChangeEvidence(0, 1, 1, 1, None, None),
            None,
        )
        self.assertEqual(Transitions.receive_view_change(stale_view_change.actor.cp, stale_view_change), "handled")
        self.assertEqual(Transitions.view_timeout(old_timeout.actor.cp, old_timeout), "handled")
        self.assertEqual(fixture.nodes[1].cp.state.current_view, 1)
        self.assertTrue(any(record.status == "ignored" for record in LiuRuntimeInstrumentationCollector.view_changes[-2:]))

    def test_primary_selection_is_replica_order_modulo_view(self):
        fixture = LiuPBFTRuntimeFixture()
        protocol = fixture.nodes[0].cp
        self.assertEqual(protocol.roles_for(1, 0).replica_ids, (1, 2, 3))
        self.assertEqual([protocol.roles_for(1, view).primary_id for view in range(6)], [1, 2, 3, 1, 2, 3])

    def test_replica_timeout_payload_is_explicitly_epoch_height_view_scoped(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.step()  # REQUEST dispatch arms replica phase timers.
        timeout = next(
            item[1] for item in fixture.queue.prio_queue.pq if item[1].payload["type"] == Messages.VIEW_TIMEOUT
        )
        self.assertEqual(
            (timeout.payload["epoch_id"], timeout.payload["height"], timeout.payload["view"]),
            (timeout.liu_context.epoch_id, timeout.liu_context.height, timeout.liu_context.view),
        )
        self.assertEqual(timeout.payload["phase"], LiuPBFTPhase.WAITING_PREPREPARE.value)
        self.assertEqual(timeout.payload["timeout_policy"], "liu_runtime_phase_relative_timeout_v2")
        timeout.payload["view"] = 99
        self.assertEqual(Transitions.view_timeout(timeout.actor.cp, timeout), "handled")
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].detail, "timeout_payload_context_mismatch")

    def test_prepared_certificate_is_carried_and_reproposed_with_same_digest(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: fixture.nodes[2].cp.state.highest_prepared_certificate is not None)
        protocol = fixture.nodes[2].cp
        locked = protocol.state.highest_prepared_certificate
        block = protocol.state.locked_block.copy()
        items = (evidence(1, 1, locked), evidence(2, 1, locked))
        certificate = PBFTNewViewCertificate(0, 1, 1, items, 2, locked, 15.0)
        self.assertTrue(protocol.transition_to_view(1, certificate, 15.0, "manual", block))
        protocol.propose_carried_value(15.0, "manual")
        carried = [
            record
            for record in LiuRuntimeInstrumentationCollector.protocol_messages
            if record.message_type == Messages.PRE_PREPARE and record.sent_at == 15.0
        ]
        self.assertEqual(len(carried), 4)
        self.assertTrue(all(record.block_digest == locked.block_digest for record in carried))
        self.assertEqual(protocol.state.locked_digest, locked.block_digest)
        self.assertTrue(
            any(record.event_type == "carried_prepared_certificate" and record.status == "applied" for record in LiuRuntimeInstrumentationCollector.view_changes)
        )

    def test_forged_unsafe_new_view_is_rejected(self):
        fixture = LiuPBFTRuntimeFixture()
        valid = PBFTNewViewCertificate(0, 1, 1, (evidence(1, 1), evidence(2, 1)), 2, None, 1.0)
        object.__setattr__(valid, "selected_prepared_certificate", prepared(0, "f" * 64))
        sender = fixture.nodes[2].cp
        event = Messages.send_new_view(sender, 1, 1.0, sender.timeout_identity(), valid, None)
        self.assertEqual(Transitions.receive_new_view(event.actor.cp, event), "invalid")
        self.assertEqual(event.actor.cp.state.current_view, 0)
        self.assertEqual(event.actor.cp.state.highest_observed_view, 0)
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes[-1].detail, "invalid_new_view_certificate")

    def test_local_commit_may_rejoin_same_safe_digest_but_never_regresses_or_conflicts(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_until(lambda: fixture.nodes[1].cp.state.local_commit_certificate is not None)
        protocol = fixture.nodes[1].cp
        committed = protocol.state.local_commit_certificate
        prepared_certificate = protocol.state.highest_prepared_certificate
        items = (
            evidence(1, 1, prepared_certificate, committed),
            evidence(2, 1, prepared_certificate),
        )
        certificate = PBFTNewViewCertificate(0, 1, 1, items, 2, prepared_certificate, 30.0)
        self.assertTrue(protocol.transition_to_view(1, certificate, 30.0, "manual", protocol.state.locked_block))
        self.assertEqual(protocol.state.current_view, 1)
        self.assertEqual(protocol.state.locked_digest, committed.block_digest)
        self.assertEqual(protocol.state.local_commit_certificate, committed)
        self.assertEqual(protocol.state.committed_height, 1)

    def test_observer_never_participates_in_view_change(self):
        fixture = self.delayed_primary_fixture()
        fixture.run_to_all_observed()
        self.assertFalse(any(record.node_id == 4 for record in LiuRuntimeInstrumentationCollector.view_changes))
        self.assertFalse(
            any(
                record.message_type in (Messages.VIEW_CHANGE, Messages.NEW_VIEW)
                and (record.sender_id == 4 or record.receiver_id == 4)
                for record in LiuRuntimeInstrumentationCollector.protocol_messages
            )
        )

    def test_normal_path_has_no_view_change_and_retains_known_timing(self):
        fixture = LiuPBFTRuntimeFixture()
        fixture.run_to_all_observed()
        finality = LiuRuntimeInstrumentationCollector.protocol_finalities[0]
        self.assertAlmostEqual(finality.consensus_latency_s, 20.2)
        self.assertEqual(finality.finality_path, "pbft_normal_reply_quorum")
        self.assertEqual(LiuRuntimeInstrumentationCollector.view_changes, [])


class PBFTTimeoutBackoffTests(unittest.TestCase):
    @staticmethod
    def run_fixture(initial_timeout, factor=2.0, maximum=40.0, maximum_changes=6):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=initial_timeout, alpha=0.0, beta=0.0)
        for node in fixture.nodes:
            node.cp.runtime_configuration = replace(
                node.cp.runtime_configuration,
                timeout_backoff_factor=factor,
                timeout_max_s=maximum,
                maximum_timeout_view_changes=maximum_changes,
            )
        fixture.run_until(lambda: all(node.last_block.depth == 1 for node in fixture.nodes), maximum=10_000)
        return fixture

    def test_versioned_bounded_exponential_formula(self):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=5.0)
        configuration = fixture.nodes[0].cp.runtime_configuration
        self.assertEqual(configuration.TIMEOUT_BACKOFF_POLICY_VERSION, "liu_pbft_deterministic_timeout_backoff_v1")
        self.assertEqual(configuration.TIMEOUT_BACKOFF_CLASSIFICATION, "RECONSTRUCTION_REQUIRED")
        self.assertEqual([configuration.effective_timeout_s(k) for k in range(9)], [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 320.0, 320.0])

    def test_low_latency_normal_path_needs_no_backoff(self):
        fixture = self.run_fixture(5.0)
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities[0].finality_path, "pbft_normal_reply_quorum")
        self.assertFalse(any(record.event_type == "fired" for record in LiuRuntimeInstrumentationCollector.timeout_adaptations))
        self.assertTrue(all(node.cp.state.timeout_backoff_count == 0 for node in fixture.nodes))

    def test_one_timeout_driven_view_then_success(self):
        self.run_fixture(1.0, maximum=8.0)
        fired = [record for record in LiuRuntimeInstrumentationCollector.timeout_adaptations if record.event_type == "fired"]
        self.assertTrue(fired)
        self.assertEqual(max(record.effective_timeout_s for record in LiuRuntimeInstrumentationCollector.timeout_adaptations), 2.0)
        self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 1)

    def test_multiple_timeouts_then_success_and_factor_two_is_deterministic(self):
        projections = []
        traces = []
        for _ in range(2):
            fixture = self.run_fixture(0.5, factor=2.0, maximum=4.0, maximum_changes=8)
            projections.append(fixture.projection())
            traces.append(tuple(LiuRuntimeInstrumentationCollector.timeout_adaptations))
            self.assertEqual(len(LiuRuntimeInstrumentationCollector.protocol_finalities), 1)
        self.assertEqual(projections[0], projections[1])
        self.assertEqual(traces[0], traces[1])
        self.assertGreater(max(record.consecutive_timeout_view_changes for record in traces[0]), 1)

    def test_factor_two_reaches_finality_with_fewer_events_than_factor_one_point_five(self):
        results = {}
        for factor in (1.5, 2.0):
            fixture = LiuPBFTRuntimeFixture(request_timeout_s=0.5, alpha=0.0, beta=0.0)
            for node in fixture.nodes:
                node.cp.runtime_configuration = replace(
                    node.cp.runtime_configuration,
                    timeout_backoff_factor=factor,
                    timeout_max_s=4.0,
                    maximum_timeout_view_changes=8,
                )
            events = fixture.run_until(lambda: all(node.last_block.depth == 1 for node in fixture.nodes), maximum=10_000)
            results[factor] = (len(events), LiuRuntimeInstrumentationCollector.protocol_finalities[0].finality_time)
        self.assertLess(results[2.0][0], results[1.5][0])
        self.assertLess(results[2.0][1], results[1.5][1])

    def test_timeout_cap_returns_explicit_bounded_failure(self):
        fixture = LiuPBFTRuntimeFixture(request_timeout_s=0.5, alpha=0.0, beta=0.0)
        for node in fixture.nodes:
            node.cp.runtime_configuration = replace(
                node.cp.runtime_configuration,
                timeout_max_s=0.5,
                maximum_timeout_view_changes=2,
            )
        fixture.run_until(
            lambda: all(
                node.cp.state.phase.value == "view_change_waiting"
                and node.cp.state.timeout_backoff_count >= 2
                for node in fixture.nodes[:4] if node.id in node.cp.state.replica_ids
            ),
            maximum=10_000,
        )
        self.assertTrue(any(record.status == "cap_reached_waiting_for_finality" for record in LiuRuntimeInstrumentationCollector.view_changes))
        self.assertEqual(LiuRuntimeInstrumentationCollector.protocol_finalities, [])


if __name__ == "__main__":
    unittest.main()
