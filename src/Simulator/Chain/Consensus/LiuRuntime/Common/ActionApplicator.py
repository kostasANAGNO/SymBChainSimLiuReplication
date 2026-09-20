"""Atomic LiuAction application at a protocol-finalized epoch boundary."""

from __future__ import annotations

from dataclasses import dataclass

from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Chain.Consensus.LiuRuntime.Common.Identity import parent_digest
from Chain.Consensus.LiuRuntime.Common.ProtocolFactory import LiuRuntimeProtocolFactory
from Chain.ValidatorSet import ValidatorSet
from Liu.Action import LiuAction
from Liu.EpochConfiguration import EpochConfiguration
from Liu.Serialization import CanonicalSerializable
from Utils.LiuRuntimeInstrumentation import EpochLifecycleRecord, LiuRuntimeInstrumentationCollector


_RUNTIME_NAMES = {
    "LIU_QUORUM": "LiuQuorum",
    "PBFT": "LiuPBFT",
    "ZYZZYVA": "LiuZyzzyva",
}


@dataclass(frozen=True, slots=True)
class LiuEpochActivationResult(CanonicalSerializable):
    epoch_configuration: EpochConfiguration
    epoch_context: LiuRuntimeEpochContext
    action_hash: str
    validator_added: tuple[int, ...]
    validator_removed: tuple[int, ...]
    pending_validator_sync: tuple[int, ...]
    activation_time: float

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "activation_time": self.activation_time,
            "epoch_configuration": self.epoch_configuration.to_dict(),
            "epoch_hash": self.epoch_context.epoch_hash,
            "pending_validator_sync": list(self.pending_validator_sync),
            "validator_added": list(self.validator_added),
            "validator_removed": list(self.validator_removed),
        }


class LiuRuntimeActionApplicator:
    """Own the active epoch pointer and immutable historical snapshots."""

    def __init__(
        self,
        nodes,
        current_epoch: LiuRuntimeEpochContext,
        protocol_factories: dict,
    ) -> None:
        self.nodes = tuple(nodes)
        if tuple(node.id for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("runtime nodes must be ordered and have contiguous IDs 0..N-1")
        self.current_epoch = current_epoch
        self._contexts = {current_epoch.epoch_id: current_epoch}
        self._activation_heights = {}
        self._pending_sync: set[int] = set()
        self._protocol_factories = dict(protocol_factories)

    @property
    def epoch_history(self) -> tuple[LiuRuntimeEpochContext, ...]:
        return tuple(self._contexts[key] for key in sorted(self._contexts))

    @property
    def pending_validator_sync(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending_sync))

    def context_for_epoch(self, epoch_id: int, epoch_hash: str | None = None) -> LiuRuntimeEpochContext:
        context = self._contexts[epoch_id]
        if epoch_hash is not None and context.epoch_hash != epoch_hash:
            raise ValueError("historical epoch hash does not match the immutable snapshot")
        return context

    def _record(self, event_type: str, time: float, context, action_hash=None, node_id=None, detail=None) -> None:
        epoch = context.epoch_configuration
        LiuRuntimeInstrumentationCollector.epoch_lifecycle.append(
            EpochLifecycleRecord(
                event_type,
                time,
                epoch.epoch_id,
                epoch.activation_height or 1,
                context.epoch_hash,
                epoch.previous_epoch_hash,
                node_id,
                epoch.validator_set.ids,
                epoch.consensus_protocol.value,
                epoch.block_size_mb,
                epoch.block_interval_s,
                action_hash,
                detail,
            )
        )

    def _validate_boundary(self, current_finalized_height: int) -> tuple[str, object]:
        if isinstance(current_finalized_height, bool) or not isinstance(current_finalized_height, int):
            raise ValueError("current_finalized_height must be an integer")
        if current_finalized_height < 1:
            raise ValueError("dynamic epoch activation requires a finalized non-genesis height")
        old_validators = self.current_epoch.validator_ids
        blocks = []
        for node_id in old_validators:
            node = self.nodes[node_id]
            if node.last_block.depth != current_finalized_height:
                raise ValueError("all current validators must hold the finalized activation parent")
            blocks.append(node.last_block)
        digests = {parent_digest(block) for block in blocks}
        if len(digests) != 1:
            raise ValueError("current validators disagree on the finalized activation parent")
        digest = next(iter(digests))
        expected_protocol = _RUNTIME_NAMES[self.current_epoch.epoch_configuration.consensus_protocol.value]
        if not any(
            record.epoch_id == self.current_epoch.epoch_id
            and record.height == current_finalized_height
            and record.block_digest == digest
            and record.protocol == expected_protocol
            for record in LiuRuntimeInstrumentationCollector.protocol_finalities
        ):
            raise ValueError("epoch activation requires protocol-finality evidence for the boundary height")
        activation_height = current_finalized_height + 1
        if activation_height in self._activation_heights:
            raise ValueError("one active epoch per height invariant violated")
        return digest, blocks[0]

    def validate_finalized_boundary(self, current_finalized_height: int) -> None:
        """Validate the active epoch's completion without changing state."""
        self._validate_boundary(current_finalized_height)

    def validate_action(self, action: LiuAction, current_finalized_height: int):
        """Validate a prospective transition without mutation or RNG use."""
        if not isinstance(action, LiuAction):
            raise ValueError("action must be an immutable LiuAction")
        if action.node_count != len(self.nodes):
            raise ValueError("LiuAction N must match the runtime node population")
        if any(node.liu_epoch_id not in (None, self.current_epoch.epoch_id) for node in self.nodes):
            raise ValueError("node epoch regression/inconsistent active epoch detected")
        if action.validator_count != len(self.current_epoch.validator_ids):
            raise ValueError("dynamic epoch action must select exactly the configured K validators")
        if action.consensus_protocol not in self._protocol_factories:
            raise ValueError("no runtime protocol factory configured for selected consensus")
        _, canonical_parent = self._validate_boundary(current_finalized_height)
        old_epoch = self.current_epoch
        old = old_epoch.epoch_configuration
        prospective_epoch = EpochConfiguration(
            old.epoch_id + 1,
            ValidatorSet(action.validator_ids),
            action.consensus_protocol,
            action.block_size_mb,
            action.block_interval_s,
            current_finalized_height + 1,
            old_epoch.epoch_hash,
        )
        prospective_context = LiuRuntimeEpochContext(
            prospective_epoch,
            old_epoch.node_profiles,
            old_epoch.link_state_matrix,
            old_epoch.threat_scenario,
        )
        self._protocol_factories[action.consensus_protocol].configuration_for(prospective_context)
        return canonical_parent

    def apply(
        self,
        action: LiuAction,
        current_finalized_height: int,
        time: float,
        next_link_state_matrix=None,
    ) -> LiuEpochActivationResult:
        canonical_parent = self.validate_action(action, current_finalized_height)

        old_epoch = self.current_epoch
        old = old_epoch.epoch_configuration
        validators = ValidatorSet(action.validator_ids)
        epoch = EpochConfiguration(
            old.epoch_id + 1,
            validators,
            action.consensus_protocol,
            action.block_size_mb,
            action.block_interval_s,
            current_finalized_height + 1,
            old_epoch.epoch_hash,
        )
        context = LiuRuntimeEpochContext(
            epoch,
            old_epoch.node_profiles,
            old_epoch.link_state_matrix if next_link_state_matrix is None else next_link_state_matrix,
            old_epoch.threat_scenario,
        )
        factory: LiuRuntimeProtocolFactory = self._protocol_factories[action.consensus_protocol]

        # Validate and construct every object before mutating any node pointer.
        protocols = tuple(factory.create(node, context) for node in self.nodes)
        canonical_digest = parent_digest(canonical_parent)
        readiness = tuple(
            not validators.contains(node.id)
            or (
                node.last_block.depth == current_finalized_height
                and parent_digest(node.last_block) == canonical_digest
            )
            for node in self.nodes
        )
        added = tuple(sorted(set(validators.ids) - set(old.validator_set.ids)))
        removed = tuple(sorted(set(old.validator_set.ids) - set(validators.ids)))
        pending = tuple(node.id for node, ready in zip(self.nodes, readiness) if node.id in validators.ids and not ready)

        for node, protocol, ready in zip(self.nodes, protocols, readiness):
            node.activate_liu_epoch(validators, epoch.epoch_id, ready)
            if validators.contains(node.id) and not ready:
                node.state.synced = False
            node.reconfiguration_state.configuration.block_size = epoch.block_size_mb
            node.reconfiguration_state.configuration.block_time = epoch.block_interval_s
            node.cp = protocol
        for node, protocol, ready in zip(self.nodes, protocols, readiness):
            if ready:
                protocol.init(time, 0)

        self.current_epoch = context
        self._contexts[epoch.epoch_id] = context
        self._activation_heights[epoch.activation_height] = epoch.epoch_id
        self._pending_sync = set(pending)
        action_hash = action.deterministic_hash()
        self._record("epoch_transition", time, context, action_hash, detail=f"from_epoch={old.epoch_id}")
        self._record("action_applied", time, context, action_hash)
        for node_id in added:
            self._record("validator_added", time, context, action_hash, node_id)
        for node_id in removed:
            self._record("validator_removed", time, context, action_hash, node_id)
        if old.consensus_protocol is not epoch.consensus_protocol:
            self._record("protocol_switched", time, context, action_hash, detail=f"from={old.consensus_protocol.value}")
        if old.block_size_mb != epoch.block_size_mb:
            self._record("block_size_changed", time, context, action_hash, detail=f"from={old.block_size_mb}")
        if old.block_interval_s != epoch.block_interval_s:
            self._record("block_interval_changed", time, context, action_hash, detail=f"from={old.block_interval_s}")
        for node_id in pending:
            self._record("validator_sync_started", time, context, action_hash, node_id)
        return LiuEpochActivationResult(epoch, context, action_hash, added, removed, pending, time)

    def complete_validator_sync(self, node_id: int, source_node_id: int, time: float) -> None:
        if node_id not in self._pending_sync:
            raise ValueError("node is not awaiting validator synchronization")
        if source_node_id not in self.current_epoch.validator_ids:
            raise ValueError("synchronization source must be a current validator")
        node = self.nodes[node_id]
        source = self.nodes[source_node_id]
        target_depth = self.current_epoch.epoch_configuration.activation_height - 1
        if source.last_block.depth < target_depth:
            raise ValueError("synchronization source has not reached the activation parent")
        if len(node.blockchain) > len(source.blockchain):
            raise ValueError("validator chain is ahead of synchronization source")
        for depth, local in enumerate(node.blockchain):
            if local.id != source.blockchain[depth].id:
                raise ValueError("validator has a conflicting chain prefix")
        for block in source.blockchain[len(node.blockchain) : target_depth + 1]:
            node.add_block(block.copy(), time, cause="synchronization")
        if node.last_block.depth != target_depth or parent_digest(node.last_block) != parent_digest(source.blockchain[target_depth]):
            raise ValueError("validator synchronization did not reach the canonical activation parent")
        node.state.synced = True
        node.activate_liu_epoch(self.current_epoch.epoch_configuration.validator_set, self.current_epoch.epoch_id, True)
        node.cp.init(time, 0)
        self._pending_sync.remove(node_id)
        self._record("validator_sync_completed", time, self.current_epoch, node_id=node_id, detail=f"source={source_node_id}")
