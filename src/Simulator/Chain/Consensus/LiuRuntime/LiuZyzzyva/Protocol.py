"""Liu Zyzzyva speculative fast path plus commit-certificate recovery."""

from Chain.Network import Network
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.Roles import LiuZyzzyvaRoles
from Chain.Consensus.LiuRuntime.Common.RuntimeProtocol import LiuRuntimeConsensusProtocol
from Chain.Consensus.LiuRuntime.LiuZyzzyva.Configuration import LiuZyzzyvaRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuZyzzyva.State import LiuZyzzyvaPhase, LiuZyzzyvaState
from Chain.Consensus.LiuRuntime.LiuZyzzyva import Messages, Transitions
from Chain.Consensus.LiuRuntime.Processing.NodeComputeQueue import NodeComputeQueue
from Chain.Consensus.LiuRuntime.Processing.ProcessingCostModel import LiuProcessingCostModel
from Chain.Consensus.LiuRuntime.Transport.MessageSizePolicy import LiuPaperMessageSizePolicy
from Chain.Consensus.LiuRuntime.Transport.NetworkAdapter import LiuDirectedTransport
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector, ProtocolViewChangeRecord


class LiuZyzzyva(LiuRuntimeConsensusProtocol):
    NAME = "LiuZyzzyva"

    def __init__(self, node, runtime_configuration: LiuZyzzyvaRuntimeConfiguration | None = None) -> None:
        configuration = runtime_configuration or LiuZyzzyvaRuntimeConfiguration.from_parameters(node)
        super().__init__(node, configuration)
        self.state = LiuZyzzyvaState()
        self.cpu_queue = NodeComputeQueue()
        self.processing_cost = LiuProcessingCostModel(
            configuration.signature_cycles_alpha,
            configuration.mac_cycles_beta,
        )
        self.message_size_policy = LiuPaperMessageSizePolicy()
        self.transport = LiuDirectedTransport(Network.nodes, self.epoch_context, configuration.propagation_delay_s)

    def set_state(self) -> None:
        super().set_state()
        self.state = LiuZyzzyvaState()
        self.cpu_queue = NodeComputeQueue()

    def state_to_string(self) -> str:
        return (
            f"height:{self.state.height} view:{self.state.current_view} phase:{self.state.phase.value} "
            f"client:{self.state.client_id} primary:{self.state.primary_id}"
        )

    def roles_for(self, height: int, view: int = 0) -> LiuZyzzyvaRoles:
        return LiuZyzzyvaRoles.for_height_view(
            self.epoch_context.epoch_configuration.validator_set,
            height,
            view,
        )

    def init(self, time: float, starting_round: int) -> None:
        self.set_state()
        self.start(time, max(0, starting_round))

    def start(self, time: float, starting_round: int):
        view = max(0, starting_round)
        self.rounds.round = view
        height = self.node.last_block.depth + 1
        roles = self.roles_for(height, view)
        if self.node.id == roles.client_id:
            phase = LiuZyzzyvaPhase.CLIENT_READY
        elif self.node.id == roles.primary_id:
            phase = LiuZyzzyvaPhase.PRIMARY_WAITING_REQUEST
        elif self.node.id in roles.backup_ids:
            phase = LiuZyzzyvaPhase.BACKUP_WAITING_ORDER
        else:
            phase = LiuZyzzyvaPhase.OBSERVER
        self.state = LiuZyzzyvaState(
            phase=phase,
            height=height,
            current_view=view,
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
        """Start replica request/order waits at actual REQUEST dispatch."""
        for replica_id in self.state.replica_ids:
            replica = next(node for node in self.transport.nodes if node.id == replica_id)
            target = replica.cp
            if (
                target is not None
                and target.NAME == self.NAME
                and target.state.height == identity.height
                and target.state.current_view == self.state.current_view
                and target.state.phase in (
                    LiuZyzzyvaPhase.PRIMARY_WAITING_REQUEST,
                    LiuZyzzyvaPhase.BACKUP_WAITING_ORDER,
                )
            ):
                Messages.schedule_view_timeout(
                    target,
                    dispatch_time + 2 * target.runtime_configuration.request_timeout_s,
                    identity,
                    target.state.phase,
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
        roles = self.roles_for(identity.height, self.state.current_view)
        return super().valid_block_identity(block, identity, roles.client_id)

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
        carried_digest: str | None = None,
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
                carried_digest,
                detail,
            )
        )

    def transition_to_view(
        self,
        target_view: int,
        certificate,
        time: float,
        trigger_message_id: str,
        selected_block=None,
    ) -> bool:
        state = self.state
        if (
            target_view <= state.current_view
            or state.local_committed_height == state.height
            or state.fast_finalized_height == state.height
            or state.recovery_finalized_height == state.height
            or self.node.last_block.depth >= state.height
        ):
            return False
        old_view = state.current_view
        previous_phase = state.phase
        roles = self.roles_for(state.height, target_view)
        selected_commit = certificate.selected_commit_certificate
        selected_speculative = certificate.selected_speculative_evidence
        selected = selected_commit if selected_commit is not None else selected_speculative
        if selected is not None:
            identity = LiuBlockIdentity(
                certificate.epoch_id,
                certificate.height,
                selected.parent_digest,
                selected.block_digest,
            )
            if selected_block is None or not self.valid_block_identity(selected_block, identity):
                return False
            state.block = selected_block.copy()
            state.block_identity = identity
            state.strongest_safety_evidence = selected
            if selected_speculative is not None:
                state.highest_speculative_evidence = selected_speculative
            if selected_commit is not None:
                if state.locked_digest is not None and state.locked_digest != selected_commit.block_digest:
                    return False
                state.recovery_commit_certificate = selected_commit
                state.locked_digest = selected_commit.block_digest
                state.locked_block = selected_block.copy()
        elif state.locked_digest is not None:
            return False
        elif self.node.id != roles.client_id:
            state.block = None
            state.block_identity = None

        if state.block is not None:
            state.block.extra_data["round"] = target_view
        state.current_view = target_view
        self.rounds.round = target_view
        state.client_id = roles.client_id
        state.primary_id = roles.primary_id
        state.replica_ids = roles.replica_ids
        state.backup_ids = roles.backup_ids
        state.accepted_order = None
        state.speculative_execution_state = None
        state.speculative_replies = {} if state.block_identity is None else {state.block_identity.block_digest: {}}
        state.seen_speculative_reply_signers = set()
        state.conflicting_speculative_replies = {}
        state.sent_speculative_reply = False
        state.failure_reason = None
        state.recovery_local_commits = {} if state.block_identity is None else {state.block_identity.block_digest: {}}
        state.sent_local_commit = False
        state.pending_target_view = None
        state.new_view_certificate = certificate
        if self.node.id == roles.client_id:
            state.phase = (
                LiuZyzzyvaPhase.RECOVERY_CERTIFICATE_CREATED
                if selected_commit is not None
                else LiuZyzzyvaPhase.CLIENT_WAITING_REPLIES
            )
        elif self.node.id in roles.replica_ids:
            state.phase = (
                LiuZyzzyvaPhase.NEW_VIEW_ESTABLISHED
                if selected_commit is not None
                else (
                    LiuZyzzyvaPhase.PRIMARY_WAITING_REQUEST
                    if self.node.id == roles.primary_id
                    else LiuZyzzyvaPhase.BACKUP_WAITING_ORDER
                )
            )
        else:
            state.phase = LiuZyzzyvaPhase.OBSERVER
        identity = self.timeout_identity()
        self.record_phase(previous_phase, state.phase, time, trigger_message_id, identity)
        self.record_view_event(
            "view_transition",
            "accepted",
            time,
            old_view,
            target_view,
            len(certificate.signer_ids),
            certificate.safe_block_digest,
            "commit_recovery" if selected_commit is not None else "speculative_or_fresh",
        )
        if self.node.id in roles.replica_ids:
            Messages.schedule_view_timeout(
                self,
                time + 2 * self.runtime_configuration.request_timeout_s,
                identity,
                state.phase,
            )
        return True

    @staticmethod
    def handle_event(event) -> str:
        protocol = event.actor.cp
        if protocol is None or protocol.NAME != LiuZyzzyva.NAME:
            return "different_protocol"
        event_type = event.payload["type"]
        if event_type == Messages.START_REQUEST:
            valid, action = protocol.validate_message(event)
            return "invalid" if not valid or action is not None else Transitions.start_request(protocol, event)
        if event_type == Messages.REQUEST:
            return Transitions.receive_request(protocol, event)
        if event_type == Messages.COMPLETE_PRIMARY_ORDER:
            return Transitions.complete_primary_order(protocol, event)
        if event_type == Messages.ORDER_REQUEST:
            return Transitions.receive_order_request(protocol, event)
        if event_type == Messages.COMPLETE_SPECULATIVE:
            return Transitions.complete_speculative_processing(protocol, event)
        if event_type == Messages.SPECULATIVE_REPLY:
            return Transitions.receive_speculative_reply(protocol, event)
        if event_type == Messages.TIMEOUT:
            return Transitions.fast_timeout(protocol, event)
        if event_type == Messages.COMMIT_CERTIFICATE:
            return Transitions.receive_commit_certificate(protocol, event)
        if event_type == Messages.COMPLETE_RECOVERY:
            return Transitions.complete_recovery_processing(protocol, event)
        if event_type == Messages.LOCAL_COMMIT:
            return Transitions.receive_local_commit(protocol, event)
        if event_type == Messages.VIEW_TIMEOUT:
            return Transitions.view_timeout(protocol, event)
        if event_type == Messages.VIEW_CHANGE:
            return Transitions.receive_view_change(protocol, event)
        if event_type == Messages.NEW_VIEW:
            return Transitions.receive_new_view(protocol, event)
        if event_type == Messages.FINALIZED_BLOCK:
            return Transitions.receive_finalized_block(protocol, event)
        return "unhandled"
