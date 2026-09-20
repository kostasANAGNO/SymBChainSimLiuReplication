"""Thin integration base; protocol transitions remain protocol-specific."""

from abc import abstractmethod
from dataclasses import dataclass

from Chain.Consensus.ConsensusProtocol import ConsensusProtocol
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector, ProtocolPhaseTransitionRecord


@dataclass(slots=True)
class LiuRuntimeRoundState:
    round: int = 0


class LiuRuntimeConsensusProtocol(ConsensusProtocol):
    def __init__(self, node, runtime_configuration) -> None:
        self.node = node
        self.runtime_configuration = runtime_configuration
        self.epoch_context = runtime_configuration.epoch_context
        self.rounds = LiuRuntimeRoundState()

    def set_state(self) -> None:
        self.rounds = LiuRuntimeRoundState()

    def validate_message(self, event):
        context = getattr(event, "liu_context", None)
        if context is None or context.protocol != self.NAME:
            return False, None
        if context.epoch_id != self.epoch_context.epoch_id or context.epoch_hash != self.epoch_context.epoch_hash:
            return False, None
        if context.validator_set_snapshot != self.epoch_context.epoch_configuration.validator_set:
            return False, None
        if self.node.is_validator and not self.node.liu_epoch_ready:
            return False, None
        if context.height < self.node.last_block.depth + 1:
            return False, None
        if context.height > self.node.last_block.depth + 1:
            return True, "backlog"
        return True, None

    def validate_block(self, block, time: float) -> str:
        del time
        return "valid" if block.depth == self.node.last_block.depth + 1 else "invalid"

    def valid_block_identity(self, block, identity: LiuBlockIdentity, client_id: int) -> bool:
        if not isinstance(identity, LiuBlockIdentity) or identity.epoch_id != self.epoch_context.epoch_id:
            return False
        if block.depth != identity.height or block.previous != self.node.last_block.id:
            return False
        if identity.parent_digest != parent_digest(self.node.last_block):
            return False
        expected = LiuBlockIdentity.create(
            self.epoch_context.epoch_id,
            identity.height,
            identity.parent_digest,
            client_id,
            tuple(block.transactions),
        )
        return expected == identity and block.miner == client_id and block.consensus == self.NAME

    def record_phase(
        self,
        previous,
        new,
        time: float,
        trigger_message_id: str,
        identity: LiuBlockIdentity,
        processing_started_at: float | None = None,
        processing_completed_at: float | None = None,
        processing_work_name: str | None = None,
        work=None,
    ) -> None:
        LiuRuntimeInstrumentationCollector.phase_transitions.append(
            ProtocolPhaseTransitionRecord(
                self.NAME,
                identity.epoch_id,
                identity.height,
                identity.block_digest,
                self.node.id,
                previous.value,
                new.value,
                time,
                trigger_message_id,
                processing_started_at,
                processing_completed_at,
                processing_work_name,
                None if work is None else work.signature_operations,
                None if work is None else work.mac_operations,
            )
        )

    def init_round_change(self, time: float) -> None:
        del time

    @abstractmethod
    def start(self, time: float, starting_round: int):
        raise NotImplementedError
