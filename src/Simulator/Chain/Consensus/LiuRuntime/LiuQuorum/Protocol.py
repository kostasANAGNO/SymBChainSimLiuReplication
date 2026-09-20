"""The Liu paper's leaderless, no-batching Request/Reply runtime abstraction."""

from Chain.Network import Network
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.Roles import LiuQuorumRoles
from Chain.Consensus.LiuRuntime.Common.RuntimeProtocol import LiuRuntimeConsensusProtocol
from Chain.Consensus.LiuRuntime.LiuQuorum.Configuration import LiuQuorumRuntimeConfiguration
from Chain.Consensus.LiuRuntime.LiuQuorum.State import LiuQuorumPhase, LiuQuorumState
from Chain.Consensus.LiuRuntime.LiuQuorum import Messages, Transitions
from Chain.Consensus.LiuRuntime.Processing.NodeComputeQueue import NodeComputeQueue
from Chain.Consensus.LiuRuntime.Processing.ProcessingCostModel import LiuProcessingCostModel
from Chain.Consensus.LiuRuntime.Transport.MessageSizePolicy import LiuPaperMessageSizePolicy
from Chain.Consensus.LiuRuntime.Transport.NetworkAdapter import LiuDirectedTransport


class LiuQuorum(LiuRuntimeConsensusProtocol):
    NAME = "LiuQuorum"

    def __init__(self, node, runtime_configuration: LiuQuorumRuntimeConfiguration | None = None) -> None:
        configuration = runtime_configuration or LiuQuorumRuntimeConfiguration.from_parameters(node)
        super().__init__(node, configuration)
        self.state = LiuQuorumState()
        self.cpu_queue = NodeComputeQueue()
        self.processing_cost = LiuProcessingCostModel(
            configuration.signature_cycles_alpha,
            configuration.mac_cycles_beta,
        )
        self.message_size_policy = LiuPaperMessageSizePolicy()
        self.transport = LiuDirectedTransport(Network.nodes, self.epoch_context, configuration.propagation_delay_s)

    def set_state(self) -> None:
        super().set_state()
        self.state = LiuQuorumState()
        self.cpu_queue = NodeComputeQueue()

    def state_to_string(self) -> str:
        return f"height:{self.state.height} phase:{self.state.phase.value} client:{self.state.client_id} replies:{len(self.state.replies)}"

    def roles_for(self, height: int) -> LiuQuorumRoles:
        return LiuQuorumRoles.for_height(self.epoch_context.epoch_configuration.validator_set, height)

    def init(self, time: float, starting_round: int) -> None:
        self.set_state()
        self.start(time, starting_round)

    def start(self, time: float, starting_round: int):
        self.rounds.round = starting_round
        height = self.node.last_block.depth + 1
        roles = self.roles_for(height)
        self.state = LiuQuorumState(
            phase=LiuQuorumPhase.OBSERVER if self.node.id not in self.epoch_context.validator_ids else LiuQuorumPhase.WAITING_REQUEST,
            height=height,
            client_id=roles.client_id,
            replica_ids=roles.replica_ids,
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

    def rejoin(self, time: float) -> None:
        self.set_state()
        self.start(time, self.node.last_block.extra_data.get("round", -1) + 1)

    def valid_block_identity(self, block, identity: LiuBlockIdentity) -> bool:
        roles = self.roles_for(identity.height)
        return super().valid_block_identity(block, identity, roles.client_id)

    @staticmethod
    def handle_event(event) -> str:
        protocol = event.actor.cp
        if protocol is None or protocol.NAME != LiuQuorum.NAME:
            return "different_protocol"
        event_type = event.payload["type"]
        if event_type == Messages.START_REQUEST:
            valid, action = protocol.validate_message(event)
            return "invalid" if not valid or action is not None else Transitions.start_request(protocol, event)
        if event_type == Messages.REQUEST:
            return Transitions.receive_request(protocol, event)
        if event_type == Messages.PROCESS_REQUEST_COMPLETE:
            return Transitions.complete_request_processing(protocol, event)
        if event_type == Messages.REPLY:
            return Transitions.receive_reply(protocol, event)
        if event_type == Messages.TIMEOUT:
            valid, action = protocol.validate_message(event)
            return "invalid" if not valid or action is not None else Transitions.request_timeout(protocol, event)
        if event_type == Messages.FINALIZED_BLOCK:
            return Transitions.receive_finalized_block(protocol, event)
        return "unhandled"
