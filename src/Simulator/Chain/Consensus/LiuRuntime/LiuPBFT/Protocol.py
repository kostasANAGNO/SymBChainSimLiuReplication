"""Liu PBFT normal-path protocol with a client distinct from the replica primary."""

from Chain.Network import Network
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.Roles import LiuPBFTRoles
from Chain.Consensus.LiuRuntime.Common.RuntimeProtocol import LiuRuntimeConsensusProtocol
from Chain.Consensus.LiuRuntime.LiuPBFT.Configuration import LiuPBFTRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase, LiuPBFTState
from Chain.Consensus.LiuRuntime.LiuPBFT import Messages, Transitions
from Chain.Consensus.LiuRuntime.Processing.NodeComputeQueue import NodeComputeQueue
from Chain.Consensus.LiuRuntime.Processing.ProcessingCostModel import LiuProcessingCostModel
from Chain.Consensus.LiuRuntime.Transport.MessageSizePolicy import LiuPaperMessageSizePolicy
from Chain.Consensus.LiuRuntime.Transport.NetworkAdapter import LiuDirectedTransport
from Utils.LiuRuntimeInstrumentation import (
    LiuRuntimeInstrumentationCollector,
    ProtocolTimeoutAdaptationRecord,
    ProtocolViewChangeRecord,
)


class LiuPBFT(LiuRuntimeConsensusProtocol):
    NAME = "LiuPBFT"

    def __init__(self, node, runtime_configuration: LiuPBFTRuntimeConfiguration | None = None) -> None:
        configuration = runtime_configuration or LiuPBFTRuntimeConfiguration.from_parameters(node)
        super().__init__(node, configuration)
        self.state = LiuPBFTState()
        self.cpu_queue = NodeComputeQueue()
        self.processing_cost = LiuProcessingCostModel(
            configuration.signature_cycles_alpha,
            configuration.mac_cycles_beta,
        )
        self.message_size_policy = LiuPaperMessageSizePolicy()
        self.transport = LiuDirectedTransport(Network.nodes, self.epoch_context, configuration.propagation_delay_s)

    def set_state(self) -> None:
        super().set_state()
        self.state = LiuPBFTState()
        self.cpu_queue = NodeComputeQueue()

    def state_to_string(self) -> str:
        return (
            f"height:{self.state.height} view:{self.state.current_view} phase:{self.state.phase.value} "
            f"client:{self.state.client_id} primary:{self.state.primary_id}"
        )

    def roles_for(self, height: int, view: int | None = None) -> LiuPBFTRoles:
        selected_view = self.state.current_view if view is None else view
        return LiuPBFTRoles.for_height_view(self.epoch_context.epoch_configuration.validator_set, height, selected_view)

    def init(self, time: float, starting_round: int) -> None:
        self.set_state()
        self.start(time, max(0, starting_round))

    def start(self, time: float, starting_round: int):
        self.rounds.round = starting_round
        height = self.node.last_block.depth + 1
        roles = self.roles_for(height, starting_round)
        if self.node.id == roles.client_id:
            phase = LiuPBFTPhase.CLIENT_WAITING
        elif self.node.id in roles.replica_ids:
            phase = LiuPBFTPhase.WAITING_PREPREPARE
        else:
            phase = LiuPBFTPhase.OBSERVER
        self.state = LiuPBFTState(
            phase=phase,
            height=height,
            current_view=starting_round,
            height_start_view=starting_round,
            highest_observed_view=starting_round,
            maximum_effective_timeout_s=self.runtime_configuration.request_timeout_s,
            client_id=roles.client_id,
            primary_id=roles.primary_id,
            replica_ids=roles.replica_ids,
            backup_ids=roles.backup_ids,
        )
        if self.node.id == roles.client_id:
            pending = LiuBlockIdentity.create(
                self.epoch_context.epoch_id,
                height,
                parent_digest(self.node.last_block),
                roles.client_id,
                (),
            )
            Messages.schedule_local(
                self,
                time + self.epoch_context.epoch_configuration.block_interval_s,
                Messages.START_REQUEST,
                pending,
            )
        return None

    def arm_replica_phase_timeouts(self, dispatch_time: float, identity: LiuBlockIdentity) -> None:
        """Start PRE_PREPARE waits when the REQUEST is actually dispatched."""
        for replica_id in self.state.replica_ids:
            replica = next(node for node in self.transport.nodes if node.id == replica_id)
            target = replica.cp
            if (
                target is not None
                and target.NAME == self.NAME
                and target.state.height == identity.height
                and target.state.current_view == self.state.current_view
                and target.state.phase is LiuPBFTPhase.WAITING_PREPREPARE
            ):
                target.schedule_adaptive_timeout(
                    dispatch_time, Messages.VIEW_TIMEOUT, identity, LiuPBFTPhase.WAITING_PREPREPARE
                )

    def effective_timeout_s(self) -> float:
        return self.runtime_configuration.effective_timeout_s(self.state.timeout_backoff_count)

    def schedule_adaptive_timeout(self, phase_entry_time: float, message_type: str, identity, phase):
        duration = self.effective_timeout_s()
        self.state.maximum_effective_timeout_s = max(self.state.maximum_effective_timeout_s, duration)
        LiuRuntimeInstrumentationCollector.timeout_adaptations.append(
            ProtocolTimeoutAdaptationRecord(
                self.NAME,
                self.runtime_configuration.TIMEOUT_BACKOFF_POLICY_VERSION,
                self.epoch_context.epoch_id,
                self.state.height,
                self.node.id,
                self.state.current_view,
                phase.value,
                "scheduled",
                phase_entry_time,
                self.state.timeout_backoff_count,
                duration,
                self.runtime_configuration.request_timeout_s,
                self.runtime_configuration.timeout_max_s,
                self.runtime_configuration.timeout_backoff_factor,
            )
        )
        return Messages.schedule_phase_timeout(
            self,
            phase_entry_time + duration,
            message_type,
            identity,
            phase,
            timeout_backoff_policy=self.runtime_configuration.TIMEOUT_BACKOFF_POLICY_VERSION,
            timeout_backoff_count=self.state.timeout_backoff_count,
            effective_timeout_s=duration,
        )

    def rejoin(self, time: float) -> None:
        self.set_state()
        self.start(time, 0)

    def validate_message(self, event):
        valid, action = super().validate_message(event)
        if not valid or action is not None:
            return valid, action
        if event.payload["type"] != Messages.FINALIZED_BLOCK and event.liu_context.view != self.state.current_view:
            return False, None
        return True, None

    def valid_block_identity(self, block, identity: LiuBlockIdentity) -> bool:
        if not isinstance(identity, LiuBlockIdentity):
            return False
        return super().valid_block_identity(block, identity, self.roles_for(identity.height).client_id)

    def timeout_identity(self) -> LiuBlockIdentity:
        if self.state.block_identity is not None:
            return self.state.block_identity
        return LiuBlockIdentity.create(
            self.epoch_context.epoch_id,
            self.state.height,
            parent_digest(self.node.last_block),
            self.state.client_id,
            (),
        )

    def record_view_event(
        self,
        event_type: str,
        status: str,
        time: float,
        from_view: int,
        target_view: int,
        signer_count: int = 0,
        carried_prepared_digest: str | None = None,
        detail: str | None = None,
        height: int | None = None,
    ) -> None:
        LiuRuntimeInstrumentationCollector.view_changes.append(
            ProtocolViewChangeRecord(
                self.NAME,
                self.epoch_context.epoch_id,
                self.state.height if height is None else height,
                self.node.id,
                from_view,
                target_view,
                event_type,
                status,
                time,
                signer_count,
                carried_prepared_digest,
                detail,
            )
        )

    def safe_value_allowed(self, identity: LiuBlockIdentity) -> bool:
        state = self.state
        if state.locked_digest is None or state.locked_digest == identity.block_digest:
            return True
        certificate = None if state.new_view_certificate is None else state.new_view_certificate.selected_prepared_certificate
        if certificate is None or certificate.block_digest != identity.block_digest:
            return False
        highest_view = -1 if state.highest_prepared_certificate is None else state.highest_prepared_certificate.view
        return certificate.view > highest_view

    def transition_to_view(self, target_view: int, certificate, time: float, trigger_message_id: str, selected_block=None) -> bool:
        state = self.state
        if target_view <= state.current_view or self.node.last_block.depth >= state.height:
            return False
        previous_phase = state.phase
        previous_view = state.current_view
        roles = self.roles_for(state.height, target_view)
        if selected_block is not None:
            selected_prepared = certificate.selected_prepared_certificate
            identity = LiuBlockIdentity(
                selected_prepared.epoch_id,
                selected_prepared.height,
                selected_prepared.parent_digest,
                selected_prepared.block_digest,
            )
            state.block = selected_block.copy()
            state.block_identity = identity
            local_prepared_view = -1 if state.highest_prepared_certificate is None else state.highest_prepared_certificate.view
            if state.locked_digest != selected_prepared.block_digest or selected_prepared.view > local_prepared_view:
                state.highest_prepared_certificate = selected_prepared
                state.locked_digest = selected_prepared.block_digest
                state.locked_block = selected_block.copy()
        if state.block is not None:
            state.block.extra_data["round"] = target_view
        state.current_view = target_view
        state.timeout_backoff_count = max(
            state.timeout_backoff_count, target_view - state.height_start_view
        )
        state.highest_observed_view = max(state.highest_observed_view, target_view)
        self.rounds.round = target_view
        state.client_id = roles.client_id
        state.primary_id = roles.primary_id
        state.replica_ids = roles.replica_ids
        state.backup_ids = roles.backup_ids
        state.accepted_preprepare = None
        state.prepare_votes = {}
        state.commit_votes = {}
        state.prepared_certificate = None
        state.client_reply_votes = {}
        state.sent_prepare = False
        state.sent_commit = False
        state.prepare_processing_started = False
        state.commit_processing_started = False
        state.pending_target_view = None
        state.new_view_certificate = certificate
        if self.node.id == roles.client_id:
            state.phase = LiuPBFTPhase.CLIENT_WAITING
        elif self.node.id == roles.primary_id:
            state.phase = LiuPBFTPhase.WAITING_PREPREPARE
        elif self.node.id in roles.replica_ids:
            state.phase = LiuPBFTPhase.WAITING_PREPREPARE
        else:
            state.phase = LiuPBFTPhase.OBSERVER
        identity = self.timeout_identity()
        self.record_phase(previous_phase, state.phase, time, trigger_message_id, identity)
        self.record_view_event(
            "view_transition",
            "accepted",
            time,
            previous_view,
            target_view,
            len(certificate.signer_ids),
            None if certificate.selected_prepared_certificate is None else certificate.selected_prepared_certificate.block_digest,
        )
        if self.node.id in roles.replica_ids:
            self.schedule_adaptive_timeout(time, Messages.VIEW_TIMEOUT, identity, state.phase)
        return True

    def propose_carried_value(self, time: float, trigger_message_id: str) -> None:
        state = self.state
        certificate = state.new_view_certificate
        if self.node.id != state.primary_id or certificate is None or certificate.selected_prepared_certificate is None:
            return
        identity = state.block_identity
        if state.block is None or identity is None or not self.safe_value_allowed(identity):
            state.safety_failure = "selected_prepared_block_unavailable_or_unsafe"
            self.record_view_event(
                "carried_prepared_certificate",
                "safety_error",
                time,
                state.current_view - 1,
                state.current_view,
                len(certificate.signer_ids),
                certificate.selected_prepared_certificate.block_digest,
                state.safety_failure,
            )
            return
        state.accepted_preprepare = identity.block_digest
        self.record_view_event(
            "carried_prepared_certificate",
            "applied",
            time,
            state.current_view - 1,
            state.current_view,
            len(certificate.signer_ids),
            identity.block_digest,
        )
        for backup_id in state.backup_ids:
            Messages.send_preprepare(self, backup_id, time, state.block, identity)
        self.emit_prepare(time, trigger_message_id)
        for backup_id in state.backup_ids:
            self.retransmit_authorized_proposal(backup_id, time)

    def retransmit_authorized_proposal(self, receiver_id: int, time: float) -> bool:
        """Send the exact established-view proposal once after certificate catch-up."""
        state = self.state
        certificate = state.new_view_certificate
        identity = state.block_identity
        if (
            self.node.id != state.primary_id
            or receiver_id not in state.replica_ids
            or receiver_id == self.node.id
            or certificate is None
            or certificate.target_view != state.current_view
            or state.block is None
            or identity is None
            or state.accepted_preprepare != identity.block_digest
            or not self.safe_value_allowed(identity)
        ):
            return False
        selected = certificate.selected_prepared_certificate
        if selected is not None and selected.block_digest != identity.block_digest:
            return False
        key = (
            self.epoch_context.epoch_id,
            state.height,
            state.current_view,
            identity.block_digest,
            receiver_id,
        )
        if key in state.proposal_retransmissions:
            self.record_view_event(
                "new_view_proposal_retransmission", "duplicate_ignored", time,
                state.current_view, state.current_view,
                carried_prepared_digest=identity.block_digest,
                detail="bounded_once_per_receiver",
            )
            return False
        state.proposal_retransmissions.add(key)
        Messages.send_preprepare(self, receiver_id, time, state.block, identity)
        self.record_view_event(
            "new_view_proposal_retransmission", "sent", time,
            state.current_view, state.current_view,
            carried_prepared_digest=identity.block_digest,
            detail=self.runtime_configuration.NEW_VIEW_PROPOSAL_DELIVERY_VERSION,
        )
        return True

    def emit_prepare(self, time: float, trigger_message_id: str) -> None:
        state = self.state
        if state.sent_prepare or state.accepted_preprepare != state.block_identity.block_digest:
            return
        state.sent_prepare = True
        identity = state.block_identity
        state.prepare_votes.setdefault(identity.block_digest, {})[self.node.id] = trigger_message_id
        previous = state.phase
        state.phase = LiuPBFTPhase.PREPARE_COLLECTING
        self.record_phase(previous, state.phase, time, trigger_message_id, identity)
        self.schedule_adaptive_timeout(time, Messages.VIEW_TIMEOUT, identity, state.phase)
        for receiver_id in state.replica_ids:
            if receiver_id != self.node.id:
                Messages.send_prepare(self, receiver_id, time, identity)

    def emit_commit(self, time: float, trigger_message_id: str) -> None:
        state = self.state
        certificate = state.prepared_certificate
        identity = state.block_identity
        if state.sent_commit or certificate is None:
            return
        if not certificate.validate(
            identity,
            state.current_view,
            state.replica_ids,
            self.runtime_configuration.prepare_quorum,
        ):
            return
        state.sent_commit = True
        state.commit_votes.setdefault(identity.block_digest, {})[self.node.id] = trigger_message_id
        previous = state.phase
        state.phase = LiuPBFTPhase.COMMIT_COLLECTING
        self.record_phase(previous, state.phase, time, trigger_message_id, identity)
        self.schedule_adaptive_timeout(time, Messages.VIEW_TIMEOUT, identity, state.phase)
        for receiver_id in state.replica_ids:
            if receiver_id != self.node.id:
                Messages.send_commit(self, receiver_id, time, identity, certificate)

    @staticmethod
    def handle_event(event) -> str:
        protocol = event.actor.cp
        if protocol is None or protocol.NAME != LiuPBFT.NAME:
            return "different_protocol"
        event_type = event.payload["type"]
        if event_type == Messages.START_REQUEST:
            valid, action = protocol.validate_message(event)
            return "invalid" if not valid or action is not None else Transitions.start_request(protocol, event)
        if event_type == Messages.REQUEST:
            return Transitions.receive_request(protocol, event)
        if event_type == Messages.COMPLETE_REQUEST:
            return Transitions.complete_primary_request(protocol, event)
        if event_type == Messages.PRE_PREPARE:
            return Transitions.receive_preprepare(protocol, event)
        if event_type == Messages.COMPLETE_PREPREPARE:
            return Transitions.complete_backup_preprepare(protocol, event)
        if event_type == Messages.PREPARE:
            return Transitions.receive_prepare(protocol, event)
        if event_type == Messages.COMPLETE_PREPARE_QUORUM:
            return Transitions.complete_prepare_quorum(protocol, event)
        if event_type == Messages.COMMIT:
            return Transitions.receive_commit(protocol, event)
        if event_type == Messages.COMPLETE_COMMIT_QUORUM:
            return Transitions.complete_commit_quorum(protocol, event)
        if event_type == Messages.REPLY:
            return Transitions.receive_reply(protocol, event)
        if event_type == Messages.TIMEOUT:
            return Transitions.request_timeout(protocol, event)
        if event_type == Messages.VIEW_TIMEOUT:
            return Transitions.view_timeout(protocol, event)
        if event_type == Messages.VIEW_CHANGE:
            return Transitions.receive_view_change(protocol, event)
        if event_type == Messages.NEW_VIEW:
            return Transitions.receive_new_view(protocol, event)
        if event_type == Messages.FINALIZED_BLOCK:
            return Transitions.receive_finalized_block(protocol, event)
        return "unhandled"
