"""Protocol-specific LiuQuorum Request/Reply transitions."""

from Parameters import Parameters

from Chain.Block import Block
from Chain.TransactionFactory import TransactionFactory
from Chain.Consensus.LiuRuntime.Common.Certificates import LiuQuorumReplyCertificate
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import timeout_scope_matches
from Chain.Consensus.LiuRuntime.LiuQuorum import Messages
from Chain.Consensus.LiuRuntime.LiuQuorum.AdmissibleTimeout import estimated_slowest_required_round_trip_s
from Chain.Consensus.LiuRuntime.LiuQuorum.Configuration import ReplicaFaultPolicy
from Chain.Consensus.LiuRuntime.LiuQuorum.State import LiuQuorumPhase
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Utils.LiuRuntimeInstrumentation import (
    LiuRuntimeInstrumentationCollector,
    ProtocolCertificateRecord,
    ProtocolFinalityRecord,
)


def start_request(protocol, event) -> str:
    state = protocol.state
    if protocol.node.id != state.client_id or state.finalized:
        return "invalid"
    transactions, transaction_size = TransactionFactory.execute_transactions(
        protocol.node.reconfiguration_state.configuration,
        protocol.node.pool,
        event.time,
    )
    if not transactions:
        Messages.schedule_local(protocol, event.time + 1.0, Messages.START_REQUEST, event.liu_context.block_identity)
        return "handled"

    size = transaction_size + Parameters.data["base_block_size"]
    identity = LiuBlockIdentity.create(
        protocol.epoch_context.epoch_id,
        state.height,
        parent_digest(protocol.node.last_block),
        protocol.node.id,
        tuple(transactions),
    )
    block = Block(
        depth=state.height,
        id=int(identity.block_digest[:15], 16),
        previous=protocol.node.last_block.id,
        time_created=event.time,
        miner=protocol.node.id,
        transactions=transactions,
        size=size,
        consensus=protocol.NAME,
    )
    block.extra_data = {
        "configuration_depth": protocol.epoch_context.epoch_id,
        "liu_identity": identity.to_dict(),
        "round": protocol.rounds.round,
    }
    previous = state.phase
    state.phase = LiuQuorumPhase.WAITING_REPLIES
    state.block = block
    state.block_identity = identity
    state.request_sent_at = event.time
    state.replies = {}
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)

    configured_size = protocol.epoch_context.epoch_configuration.block_size_mb
    request_size = protocol.message_size_policy.request_size_mb(configured_size)
    for replica_id in state.replica_ids:
        Messages.send_request(protocol, replica_id, event.time, block, identity, request_size)
    estimated_round_trip = estimated_slowest_required_round_trip_s(
        link_state_matrix=protocol.epoch_context.link_state_matrix,
        capability_ghz=protocol.epoch_context.capability_ghz,
        processing_cost=protocol.processing_cost,
        message_size_policy=protocol.message_size_policy,
        propagation_delay_s=protocol.runtime_configuration.propagation_delay_s,
        client_id=state.client_id,
        replica_ids=state.replica_ids,
        block_size_mb=configured_size,
    )
    configuration = protocol.runtime_configuration
    scheduled_timeout = configuration.request_timeout_for(estimated_round_trip)
    Messages.schedule_timeout(
        protocol,
        event.time + scheduled_timeout,
        identity,
        state.phase,
    )
    return "new_state"


def receive_request(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    block = event.payload["block"]
    if event.liu_context.block_identity != identity:
        return "invalid"
    if protocol.node.id not in state.replica_ids or event.creator.id != state.client_id:
        return "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    if state.phase is not LiuQuorumPhase.WAITING_REQUEST:
        return "handled" if state.block_identity == identity else "invalid"

    fault = protocol.runtime_configuration.fault_for(protocol.node.id)
    if fault is not None and fault.policy is ReplicaFaultPolicy.OMIT:
        protocol.record_phase(state.phase, LiuQuorumPhase.FAILED, event.time, event.liu_context.logical_message_id, identity)
        state.phase = LiuQuorumPhase.FAILED
        return "handled"

    duration = protocol.processing_cost.duration_s(
        LiuProcessingWork.quorum_replica_work(),
        protocol.epoch_context.capability_ghz(protocol.node.id),
    )
    reservation = protocol.cpu_queue.reserve(event.time, duration)
    previous = state.phase
    state.phase = LiuQuorumPhase.PROCESSING_REQUEST
    state.block = block
    state.block_identity = identity
    protocol.record_phase(
        previous,
        state.phase,
        event.time,
        event.liu_context.logical_message_id,
        identity,
        reservation.processing_start,
        reservation.processing_end,
    )
    Messages.schedule_processing_complete(protocol, reservation.processing_end, identity, reservation)
    return "new_state"


def complete_request_processing(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.liu_context.block_identity
    if state.phase is not LiuQuorumPhase.PROCESSING_REQUEST or state.block_identity != identity:
        return "invalid"
    reservation = event.liu_processing_reservation
    previous = state.phase
    state.phase = LiuQuorumPhase.REPLICA_ACCEPTED
    protocol.record_phase(
        previous,
        state.phase,
        event.time,
        event.liu_context.logical_message_id,
        identity,
        reservation.processing_start,
        reservation.processing_end,
    )
    reply_time = event.time
    reply_digest = identity.block_digest
    fault = protocol.runtime_configuration.fault_for(protocol.node.id)
    if fault is not None:
        if fault.policy is ReplicaFaultPolicy.CORRUPT:
            reply_digest = "0" * 64
        elif fault.policy is ReplicaFaultPolicy.DELAY:
            reply_time += fault.delay_s
    configured_size = protocol.epoch_context.epoch_configuration.block_size_mb
    reply_size = protocol.message_size_policy.reply_size_mb(configured_size)
    Messages.send_reply(protocol, state.client_id, reply_time, identity, reply_digest, reply_size)
    return "new_state"


def receive_reply(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    sender_id = event.creator.id
    if event.liu_context.block_identity != identity:
        return "invalid"
    if protocol.node.id != state.client_id or state.phase is not LiuQuorumPhase.WAITING_REPLIES:
        return "invalid"
    if identity != state.block_identity or sender_id not in state.replica_ids:
        return "invalid"
    if event.payload["reply_digest"] != identity.block_digest:
        return "invalid"
    if sender_id in state.replies:
        return "handled"
    state.replies[sender_id] = event.liu_context.logical_message_id
    if len(state.replies) < len(state.replica_ids):
        return "handled"

    certificate = LiuQuorumReplyCertificate(
        identity,
        state.client_id,
        tuple(state.replies),
        len(state.replica_ids),
        event.time,
    )
    certificate_hash = certificate.deterministic_hash()
    LiuRuntimeInstrumentationCollector.certificates.append(
        ProtocolCertificateRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            certificate_hash,
            "liu_quorum_all_replica_replies",
            certificate.signer_ids,
            certificate.required_replies,
            event.time,
        )
    )
    LiuRuntimeInstrumentationCollector.protocol_finalities.append(
        ProtocolFinalityRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            identity.parent_digest,
            state.client_id,
            state.request_sent_at,
            event.time,
            "all_replica_replies",
            certificate_hash,
        )
    )
    previous = state.phase
    state.phase = LiuQuorumPhase.FINALIZED
    state.finalized = True
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    protocol.node.add_block(state.block.copy(), event.time)
    configured_size = protocol.epoch_context.epoch_configuration.block_size_mb
    announcement_size = protocol.message_size_policy.finalized_announcement_size_mb(configured_size)
    for receiver in protocol.transport.nodes:
        if receiver.id != protocol.node.id:
            Messages.send_finalized_block(protocol, receiver.id, event.time, state.block, identity, certificate, announcement_size)
    protocol.start(event.time, protocol.rounds.round + 1)
    return "new_state"


def request_timeout(protocol, event) -> str:
    state = protocol.state
    if (
        state.phase is not LiuQuorumPhase.WAITING_REPLIES
        or event.liu_context.block_identity != state.block_identity
        or not timeout_scope_matches(protocol, event)
    ):
        return "handled"
    previous = state.phase
    state.phase = LiuQuorumPhase.FAILED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, state.block_identity)
    return "new_state"


def receive_finalized_block(protocol, event) -> str:
    context = event.liu_context
    identity = event.payload["identity"]
    block = event.payload["block"]
    certificate = event.payload["certificate"]
    roles = protocol.roles_for(identity.height)
    valid, action = protocol.validate_message(event)
    if not valid:
        return "invalid"
    if action is not None:
        return action
    if context.block_identity != identity:
        return "invalid"
    if event.creator.id != roles.client_id:
        return "invalid"
    if not certificate.validate(roles.client_id, roles.replica_ids, identity):
        return "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    protocol.node.add_block(block.copy(), event.time)
    protocol.start(event.time, context.view + 1)
    return "new_state"
