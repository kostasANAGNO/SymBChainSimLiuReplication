"""Transitions for the Liu Zyzzyva speculative and recovery paths."""

from Parameters import Parameters

from Chain.Block import Block
from Chain.TransactionFactory import TransactionFactory
from Chain.Consensus.LiuRuntime.Common.Certificates import (
    ZyzzyvaCommitCertificate,
    ZyzzyvaFastReplyCertificate,
    ZyzzyvaNewViewCertificate,
    ZyzzyvaRecoveryFinalityCertificate,
    ZyzzyvaSpeculativeEvidence,
    ZyzzyvaViewChangeEvidence,
    select_zyzzyva_safe_value,
)
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import timeout_scope_matches
from Chain.Consensus.LiuRuntime.LiuZyzzyva import Messages
from Chain.Consensus.LiuRuntime.LiuZyzzyva.State import LiuZyzzyvaPhase
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Utils.LiuRuntimeInstrumentation import (
    LiuRuntimeInstrumentationCollector,
    ProtocolCertificateRecord,
    ProtocolFailureRecord,
    ProtocolFinalityRecord,
)


def _reserve_processing(
    protocol,
    event,
    work,
    phase,
    completion_type: str,
    work_name: str,
    processing_identity=None,
    **metadata,
) -> None:
    duration = protocol.processing_cost.duration_s(work, protocol.epoch_context.capability_ghz(protocol.node.id))
    reservation = protocol.cpu_queue.reserve(event.time, duration)
    identity = event.liu_context.block_identity if processing_identity is None else processing_identity
    previous = protocol.state.phase
    protocol.state.phase = phase
    if phase is LiuZyzzyvaPhase.PROCESSING_SPECULATIVE:
        protocol.state.speculative_execution_state = "processing"
    protocol.record_phase(
        previous,
        phase,
        event.time,
        event.liu_context.logical_message_id,
        identity,
        reservation.processing_start,
        reservation.processing_end,
        work_name,
        work,
    )
    Messages.schedule_local(
        protocol,
        reservation.processing_end,
        completion_type,
        identity,
        liu_processing_reservation=reservation,
        liu_processing_work=work,
        **metadata,
    )


def _record_failure(protocol, event, reason: str) -> None:
    identity = event.liu_context.block_identity
    LiuRuntimeInstrumentationCollector.protocol_failures.append(
        ProtocolFailureRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            event.liu_context.view,
            protocol.node.id,
            LiuZyzzyvaPhase.PENDING_RECOVERY.value,
            reason,
            event.time,
            identity.block_digest,
        )
    )


def _start_recovery_if_possible(protocol, event) -> str:
    state = protocol.state
    if state.recovery_commit_certificate is not None:
        return "handled"
    replies = state.speculative_replies.get(state.block_identity.block_digest, {})
    threshold = protocol.runtime_configuration.speculative_recovery_quorum
    if len(replies) < threshold:
        return "new_state"
    selected_signers = tuple(sorted(replies))[:threshold]
    identity = state.block_identity
    certificate = ZyzzyvaCommitCertificate(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        selected_signers,
        threshold,
        event.time,
    )
    state.recovery_commit_certificate = certificate
    state.strongest_safety_evidence = certificate
    state.recovery_local_commits = {identity.block_digest: {}}
    certificate_hash = certificate.deterministic_hash()
    LiuRuntimeInstrumentationCollector.certificates.append(
        ProtocolCertificateRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            certificate_hash,
            "zyzzyva_recovery_commit_certificate",
            certificate.signer_ids,
            certificate.threshold,
            event.time,
        )
    )
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.RECOVERY_CERTIFICATE_CREATED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    Messages.schedule_phase_timeout(
        protocol,
        event.time + protocol.runtime_configuration.request_timeout_s,
        Messages.TIMEOUT,
        identity,
        state.phase,
    )
    for replica_id in state.replica_ids:
        Messages.send_commit_certificate(protocol, replica_id, event.time, identity, certificate)
    return "new_state"


def _pending_recovery(protocol, event, reason: str) -> str:
    state = protocol.state
    if state.fast_finalized_height == state.height or state.recovery_finalized_height == state.height:
        return "invalid"
    if state.failure_reason is None:
        state.failure_reason = reason
    elif reason not in state.failure_reason:
        state.failure_reason = f"{state.failure_reason};{reason}"
    _record_failure(protocol, event, reason)
    if state.phase is not LiuZyzzyvaPhase.PENDING_RECOVERY:
        previous = state.phase
        state.phase = LiuZyzzyvaPhase.PENDING_RECOVERY
        protocol.record_phase(
            previous,
            state.phase,
            event.time,
            event.liu_context.logical_message_id,
            event.liu_context.block_identity,
        )
    return _start_recovery_if_possible(protocol, event)


def start_request(protocol, event) -> str:
    state = protocol.state
    if protocol.node.id != state.client_id or state.phase is not LiuZyzzyvaPhase.CLIENT_READY:
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
        state.client_id,
        tuple(transactions),
    )
    block = Block(
        depth=state.height,
        id=int(identity.block_digest[:15], 16),
        previous=protocol.node.last_block.id,
        time_created=event.time,
        miner=state.client_id,
        transactions=transactions,
        size=size,
        consensus=protocol.NAME,
    )
    block.extra_data = {
        "configuration_depth": protocol.epoch_context.epoch_id,
        "liu_identity": identity.to_dict(),
        "round": state.current_view,
    }
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.CLIENT_WAITING_REPLIES
    state.block = block
    state.block_identity = identity
    state.request_sent_at = event.time
    state.speculative_replies = {identity.block_digest: {}}
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    Messages.send_request(protocol, state.primary_id, event.time, block, identity)
    protocol.arm_replica_phase_timeouts(event.time, identity)
    Messages.schedule_phase_timeout(
        protocol,
        event.time + protocol.runtime_configuration.request_timeout_s,
        Messages.TIMEOUT,
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
    if protocol.node.id != state.primary_id or event.creator.id != state.client_id:
        return "invalid"
    if state.phase is not LiuZyzzyvaPhase.PRIMARY_WAITING_REQUEST:
        return "handled" if state.block_identity == identity else "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    state.block = block
    state.block_identity = identity
    state.accepted_order = identity.block_digest
    _reserve_processing(
        protocol,
        event,
        LiuProcessingWork.zyzzyva_primary_order_work(len(state.replica_ids)),
        LiuZyzzyvaPhase.PRIMARY_PROCESSING_ORDER,
        Messages.COMPLETE_PRIMARY_ORDER,
        "zyzzyva_primary_order",
    )
    return "new_state"


def complete_primary_order(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuZyzzyvaPhase.PRIMARY_PROCESSING_ORDER
        or state.block_identity != identity
        or state.accepted_order != identity.block_digest
    ):
        return "invalid"
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.ORDERED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    for backup_id in state.backup_ids:
        Messages.send_order_request(protocol, backup_id, event.time, state.block, identity)
    _reserve_processing(
        protocol,
        event,
        LiuProcessingWork.zyzzyva_speculative_replica_work(),
        LiuZyzzyvaPhase.PROCESSING_SPECULATIVE,
        Messages.COMPLETE_SPECULATIVE,
        "zyzzyva_speculative_execution",
    )
    return "new_state"


def receive_order_request(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    block = event.payload["block"]
    if event.liu_context.block_identity != identity:
        return "invalid"
    if protocol.node.id not in state.backup_ids or event.creator.id != state.primary_id:
        return "invalid"
    if state.phase is not LiuZyzzyvaPhase.BACKUP_WAITING_ORDER:
        return "handled" if state.block_identity == identity else "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    state.block = block
    state.block_identity = identity
    state.accepted_order = identity.block_digest
    _reserve_processing(
        protocol,
        event,
        LiuProcessingWork.zyzzyva_speculative_replica_work(),
        LiuZyzzyvaPhase.PROCESSING_SPECULATIVE,
        Messages.COMPLETE_SPECULATIVE,
        "zyzzyva_speculative_execution",
    )
    return "new_state"


def complete_speculative_processing(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuZyzzyvaPhase.PROCESSING_SPECULATIVE
        or state.block_identity != identity
        or state.accepted_order != identity.block_digest
        or state.sent_speculative_reply
    ):
        return "invalid"
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.SPECULATIVELY_EXECUTED
    state.speculative_execution_state = "executed"
    state.sent_speculative_reply = True
    evidence = ZyzzyvaSpeculativeEvidence(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        (protocol.node.id,),
        1,
        event.time,
    )
    if (
        state.highest_speculative_evidence is None
        or evidence.originating_view > state.highest_speculative_evidence.originating_view
    ):
        state.highest_speculative_evidence = evidence
        if state.recovery_commit_certificate is None:
            state.strongest_safety_evidence = evidence
    reservation = event.liu_processing_reservation
    protocol.record_phase(
        previous,
        state.phase,
        event.time,
        event.liu_context.logical_message_id,
        identity,
        reservation.processing_start,
        reservation.processing_end,
        "zyzzyva_speculative_execution_complete",
        event.liu_processing_work,
    )
    # DES_RECONSTRUCTION for Liu's "faulty replicas may send incorrect messages" (Sec.III-B):
    # a Byzantine replica emits a deliberately mismatched digest, forcing the client to
    # trigger the Eq.(17) recovery path. Honest replicas send the true block digest.
    reply_digest = identity.block_digest
    if protocol.node.id in protocol.runtime_configuration.faulty_replica_ids:
        reply_digest = "0" * 64
    Messages.send_speculative_reply(protocol, state.client_id, event.time, identity, reply_digest)
    Messages.schedule_view_timeout(
        protocol,
        event.time + 2 * protocol.runtime_configuration.request_timeout_s,
        identity,
        state.phase,
    )
    return "new_state"


def receive_speculative_reply(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    sender_id = event.creator.id
    if event.liu_context.block_identity != identity:
        return "invalid"
    if protocol.node.id != state.client_id or state.phase not in (
        LiuZyzzyvaPhase.CLIENT_WAITING_REPLIES,
        LiuZyzzyvaPhase.PENDING_RECOVERY,
    ):
        return "invalid"
    if identity != state.block_identity or sender_id not in state.replica_ids:
        return "invalid"
    if sender_id in state.seen_speculative_reply_signers:
        return "handled"
    state.seen_speculative_reply_signers.add(sender_id)
    if event.payload["reply_digest"] != identity.block_digest:
        conflicting_digest = event.payload["reply_digest"]
        state.conflicting_speculative_replies.setdefault(conflicting_digest, {})[sender_id] = (
            event.liu_context.logical_message_id
        )
        return _pending_recovery(protocol, event, f"conflicting_speculative_reply_from_{sender_id}")
    replies = state.speculative_replies.setdefault(identity.block_digest, {})
    replies[sender_id] = event.liu_context.logical_message_id
    if state.phase is LiuZyzzyvaPhase.PENDING_RECOVERY:
        return _start_recovery_if_possible(protocol, event)
    if len(replies) < protocol.runtime_configuration.required_fast_replies:
        return "handled"
    if tuple(sorted(replies)) != tuple(sorted(state.replica_ids)):
        return _pending_recovery(protocol, event, "fast_reply_signer_set_mismatch")

    certificate = ZyzzyvaFastReplyCertificate(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        tuple(replies),
        protocol.runtime_configuration.required_fast_replies,
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
            "zyzzyva_fast_all_replica_replies" if state.current_view == 0 else "zyzzyva_view_change_fast_all_replica_replies",
            certificate.signer_ids,
            certificate.threshold,
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
            "zyzzyva_fast_all_replica_replies"
            if state.current_view == 0
            else "zyzzyva_view_change_fast_all_replica_replies",
            certificate_hash,
        )
    )
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.FAST_FINALIZED
    state.fast_finalized_height = state.height
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    protocol.node.add_block(state.block.copy(), event.time, cause="protocol_finality")
    for receiver in protocol.transport.nodes:
        if receiver.id != protocol.node.id:
            Messages.send_finalized_block(protocol, receiver.id, event.time, state.block, identity, certificate)
    protocol.start(event.time, 0)
    return "new_state"


def fast_timeout(protocol, event) -> str:
    state = protocol.state
    if (
        event.liu_context.view != state.current_view
        or event.liu_context.height != state.height
        or event.liu_context.block_identity != state.block_identity
        or state.phase not in (
            LiuZyzzyvaPhase.CLIENT_WAITING_REPLIES,
            LiuZyzzyvaPhase.PENDING_RECOVERY,
            LiuZyzzyvaPhase.RECOVERY_CERTIFICATE_CREATED,
        )
        or not timeout_scope_matches(protocol, event)
    ):
        return "handled"
    if state.phase is LiuZyzzyvaPhase.RECOVERY_CERTIFICATE_CREATED:
        _record_failure(protocol, event, "recovery_local_commit_timeout")
        return "handled"
    replies = state.speculative_replies.get(state.block_identity.block_digest, {})
    missing = tuple(sorted(set(state.replica_ids) - set(replies)))
    return _pending_recovery(protocol, event, f"missing_fast_replies:{missing}")


def receive_commit_certificate(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    certificate = event.payload["commit_certificate"]
    roles = protocol.roles_for(identity.height, event.liu_context.view)
    if event.liu_context.block_identity != identity:
        return "invalid"
    if protocol.node.id not in roles.replica_ids or event.creator.id != roles.client_id:
        return "invalid"
    if not certificate.validate(
        identity,
        event.liu_context.view,
        roles.replica_ids,
        protocol.runtime_configuration.speculative_recovery_quorum,
    ):
        return "invalid"
    if (
        state.block_identity != identity
        or state.accepted_order != identity.block_digest
        or state.block is None
        or not protocol.valid_block_identity(state.block, identity)
    ):
        return "invalid"
    if state.local_committed_height == identity.height:
        return "handled" if state.local_commit_digest == identity.block_digest else "invalid"
    if state.recovery_commit_certificate is not None:
        return (
            "handled"
            if state.recovery_commit_certificate.deterministic_hash() == certificate.deterministic_hash()
            else "invalid"
        )
    state.recovery_commit_certificate = certificate
    state.strongest_safety_evidence = certificate
    state.locked_digest = identity.block_digest
    state.locked_block = state.block.copy()
    work = LiuProcessingWork.zyzzyva_recovery_work(
        protocol.node.id == roles.primary_id,
        protocol.runtime_configuration.tolerated_faults,
    )
    _reserve_processing(
        protocol,
        event,
        work,
        LiuZyzzyvaPhase.RECOVERY_PROCESSING,
        Messages.COMPLETE_RECOVERY,
        "zyzzyva_recovery_commit_certificate_processing",
        liu_commit_certificate=certificate,
    )
    Messages.schedule_view_timeout(
        protocol,
        event.time + 2 * protocol.runtime_configuration.request_timeout_s,
        identity,
        LiuZyzzyvaPhase.RECOVERY_PROCESSING,
    )
    return "new_state"


def complete_recovery_processing(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    certificate = event.liu_commit_certificate
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuZyzzyvaPhase.RECOVERY_PROCESSING
        or state.block_identity != identity
        or state.recovery_commit_certificate != certificate
        or state.local_committed_height is not None
    ):
        return "invalid"
    if state.accepted_order != identity.block_digest:
        return "invalid"
    state.local_commit_digest = identity.block_digest
    state.local_commit_certificate = certificate
    state.local_committed_height = identity.height
    state.sent_local_commit = True
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.LOCAL_COMMITTED
    reservation = event.liu_processing_reservation
    protocol.record_phase(
        previous,
        state.phase,
        event.time,
        event.liu_context.logical_message_id,
        identity,
        reservation.processing_start,
        reservation.processing_end,
        "zyzzyva_recovery_processing_complete",
        event.liu_processing_work,
    )
    Messages.send_local_commit(protocol, state.client_id, event.time, identity, certificate)
    return "new_state"


def receive_local_commit(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    sender_id = event.creator.id
    certificate = state.recovery_commit_certificate
    if event.liu_context.block_identity != identity or identity != state.block_identity:
        return "invalid"
    if protocol.node.id != state.client_id or state.phase is not LiuZyzzyvaPhase.RECOVERY_CERTIFICATE_CREATED:
        return "invalid"
    if sender_id not in state.replica_ids or certificate is None:
        return "invalid"
    if (
        event.payload["local_commit_digest"] != identity.block_digest
        or event.payload["commit_certificate_hash"] != certificate.deterministic_hash()
    ):
        return "invalid"
    commits = state.recovery_local_commits.setdefault(identity.block_digest, {})
    if sender_id in commits:
        return "handled"
    commits[sender_id] = event.liu_context.logical_message_id
    threshold = protocol.runtime_configuration.local_commit_quorum
    if len(commits) < threshold:
        return "handled"
    selected_signers = tuple(sorted(commits))[:threshold]
    finality_certificate = ZyzzyvaRecoveryFinalityCertificate(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        state.client_id,
        selected_signers,
        threshold,
        certificate.deterministic_hash(),
        event.time,
    )
    finality_hash = finality_certificate.deterministic_hash()
    LiuRuntimeInstrumentationCollector.certificates.append(
        ProtocolCertificateRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            finality_hash,
            "zyzzyva_recovery_finality_certificate",
            finality_certificate.signer_ids,
            finality_certificate.threshold,
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
            "zyzzyva_recovery_local_commit_quorum"
            if state.current_view == 0
            else "zyzzyva_view_change_recovery_local_commit_quorum",
            finality_hash,
        )
    )
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.RECOVERY_FINALIZED
    state.recovery_finalized_height = state.height
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    protocol.node.add_block(state.block.copy(), event.time, cause="protocol_finality")
    for receiver in protocol.transport.nodes:
        if receiver.id != protocol.node.id:
            Messages.send_finalized_block(
                protocol,
                receiver.id,
                event.time,
                state.block,
                identity,
                finality_certificate,
                certificate,
            )
    protocol.start(event.time, 0)
    return "new_state"


def _valid_view_change_context(protocol, event) -> bool:
    context = event.liu_context
    epoch = protocol.epoch_context
    return (
        context.protocol == protocol.NAME
        and context.epoch_id == epoch.epoch_id
        and context.epoch_hash == epoch.epoch_hash
        and context.validator_set_snapshot == epoch.epoch_configuration.validator_set
        and context.height == protocol.state.height
    )


def _identity_for_safety_evidence(evidence) -> LiuBlockIdentity:
    return LiuBlockIdentity(evidence.epoch_id, evidence.height, evidence.parent_digest, evidence.block_digest)


def _resume_commit_recovery(protocol, event, certificate) -> None:
    state = protocol.state
    identity = state.block_identity
    state.recovery_commit_certificate = certificate
    state.strongest_safety_evidence = certificate
    state.locked_digest = certificate.block_digest
    state.locked_block = state.block.copy()
    state.accepted_order = certificate.block_digest
    work = LiuProcessingWork.zyzzyva_recovery_work(
        protocol.node.id == state.primary_id,
        protocol.runtime_configuration.tolerated_faults,
    )
    protocol.record_view_event(
        "resumed_path",
        "commit_certificate_recovery",
        event.time,
        state.current_view - 1,
        state.current_view,
        carried_digest=identity.block_digest,
    )
    _reserve_processing(
        protocol,
        event,
        work,
        LiuZyzzyvaPhase.RECOVERY_PROCESSING,
        Messages.COMPLETE_RECOVERY,
        "zyzzyva_view_change_recovery_processing",
        processing_identity=identity,
        liu_commit_certificate=certificate,
    )


def _resume_speculative_or_fresh(protocol, event, selected_speculative) -> None:
    state = protocol.state
    if selected_speculative is not None and protocol.node.id == state.primary_id:
        state.accepted_order = state.block_identity.block_digest
        protocol.record_view_event(
            "resumed_path",
            "speculative_reorder",
            event.time,
            state.current_view - 1,
            state.current_view,
            carried_digest=state.block_identity.block_digest,
        )
        _reserve_processing(
            protocol,
            event,
            LiuProcessingWork.zyzzyva_primary_order_work(len(state.replica_ids)),
            LiuZyzzyvaPhase.PRIMARY_PROCESSING_ORDER,
            Messages.COMPLETE_PRIMARY_ORDER,
            "zyzzyva_view_change_carried_order",
            processing_identity=state.block_identity,
        )
    elif selected_speculative is None and protocol.node.id == state.client_id:
        if state.block is None or state.block_identity is None:
            state.safety_failure = "fresh_view_client_has_no_pending_block"
            return
        protocol.record_view_event(
            "resumed_path",
            "fresh_pending_request",
            event.time,
            state.current_view - 1,
            state.current_view,
            carried_digest=state.block_identity.block_digest,
        )
        Messages.send_request(protocol, state.primary_id, event.time, state.block, state.block_identity)
    if protocol.node.id == state.client_id and state.block_identity is not None:
        Messages.schedule_phase_timeout(
            protocol,
            event.time + protocol.runtime_configuration.request_timeout_s,
            Messages.TIMEOUT,
            state.block_identity,
            state.phase,
        )


def _register_view_change(protocol, evidence, evidence_block, event) -> str:
    state = protocol.state
    time = event.time
    trigger_message_id = event.liu_context.logical_message_id
    target_view = evidence.target_view
    roles = protocol.roles_for(state.height, target_view)
    if protocol.node.id != roles.primary_id:
        protocol.record_view_event(
            "view_change_received", "rejected", time, state.current_view, target_view, detail="not_target_primary"
        )
        return "invalid"
    if target_view <= state.current_view:
        protocol.record_view_event(
            "view_change_received", "ignored", time, state.current_view, target_view, detail="stale_view"
        )
        return "handled"
    if evidence.epoch_id != protocol.epoch_context.epoch_id or evidence.height != state.height:
        protocol.record_view_event(
            "view_change_received", "rejected", time, state.current_view, target_view, detail="wrong_epoch_or_height"
        )
        return "invalid"
    if not evidence.validate(roles.replica_ids, protocol.runtime_configuration.speculative_recovery_quorum):
        protocol.record_view_event(
            "view_change_received", "rejected", time, state.current_view, target_view, detail="invalid_evidence"
        )
        return "invalid"
    carried = evidence.commit_certificate or evidence.highest_speculative_evidence
    if carried is not None:
        identity = _identity_for_safety_evidence(carried)
        if evidence_block is None or not protocol.valid_block_identity(evidence_block, identity):
            protocol.record_view_event(
                "view_change_received",
                "rejected",
                time,
                state.current_view,
                target_view,
                detail="missing_or_invalid_evidence_block",
            )
            return "invalid"
        state.view_change_blocks.setdefault(target_view, {})[carried.block_digest] = evidence_block.copy()
    elif evidence_block is not None:
        return "invalid"

    by_sender: dict[int, ZyzzyvaViewChangeEvidence] = state.view_change_evidence.setdefault(target_view, {})
    if evidence.sender_id in by_sender:
        protocol.record_view_event(
            "view_change_received", "duplicate_ignored", time, state.current_view, target_view, len(by_sender)
        )
        return "handled"
    by_sender[evidence.sender_id] = evidence
    protocol.record_view_event(
        "view_change_received",
        "accepted",
        time,
        state.current_view,
        target_view,
        len(by_sender),
        None if carried is None else carried.block_digest,
    )
    threshold = protocol.runtime_configuration.view_change_quorum
    if len(by_sender) < threshold:
        return "handled"

    selected_evidence = tuple(by_sender[sender_id] for sender_id in sorted(by_sender)[:threshold])
    selected_commit, selected_speculative, safety_error = select_zyzzyva_safe_value(
        selected_evidence,
        protocol.runtime_configuration.speculative_safety_threshold,
    )
    selected = selected_commit if selected_commit is not None else selected_speculative
    if safety_error is None and state.locked_digest is not None:
        if selected_commit is None or selected_commit.block_digest != state.locked_digest:
            safety_error = "new_view_conflicts_with_local_commit_lock"
    if safety_error is not None:
        state.safety_failure = safety_error
        previous = state.phase
        state.phase = LiuZyzzyvaPhase.FAILED
        protocol.record_phase(previous, state.phase, time, trigger_message_id, protocol.timeout_identity())
        protocol.record_view_event(
            "new_view_certificate",
            "safety_error",
            time,
            state.current_view,
            target_view,
            len(selected_evidence),
            detail=safety_error,
        )
        return "invalid"
    selected_block = None
    if selected is not None:
        selected_block = state.view_change_blocks.get(target_view, {}).get(selected.block_digest)
        if selected_block is None:
            state.safety_failure = "selected_safety_block_missing"
            return "invalid"
    selected_kind = "commit_certificate" if selected_commit is not None else (
        "speculative_evidence" if selected_speculative is not None else "fresh_pending_value"
    )
    protocol.record_view_event(
        "safe_value_selected",
        selected_kind,
        time,
        state.current_view,
        target_view,
        len(selected_evidence),
        None if selected is None else selected.block_digest,
    )
    carried_hash = None if selected is None else selected.deterministic_hash()
    certificate = ZyzzyvaNewViewCertificate(
        protocol.epoch_context.epoch_id,
        state.height,
        target_view,
        roles.primary_id,
        selected_evidence,
        threshold,
        protocol.runtime_configuration.speculative_safety_threshold,
        None if selected is None else selected.block_digest,
        None if selected is None else selected.parent_digest,
        carried_hash,
        selected_commit,
        selected_speculative,
        time,
    )
    identity = protocol.timeout_identity() if selected is None else _identity_for_safety_evidence(selected)
    LiuRuntimeInstrumentationCollector.certificates.append(
        ProtocolCertificateRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            certificate.deterministic_hash(),
            "zyzzyva_new_view",
            certificate.signer_ids,
            certificate.threshold,
            time,
        )
    )
    old_view = state.current_view
    if not protocol.transition_to_view(target_view, certificate, time, trigger_message_id, selected_block):
        return "invalid"
    protocol.record_view_event(
        "new_view_certificate",
        "created",
        time,
        old_view,
        target_view,
        len(certificate.signer_ids),
        certificate.safe_block_digest,
    )
    for participant_id in roles.replica_ids + (roles.client_id,):
        if participant_id != protocol.node.id:
            Messages.send_new_view(protocol, participant_id, time, identity, certificate, selected_block)
    if selected_commit is not None:
        _resume_commit_recovery(protocol, event, selected_commit)
    else:
        _resume_speculative_or_fresh(protocol, event, selected_speculative)
    return "new_state"


def view_timeout(protocol, event) -> str:
    state = protocol.state
    context = event.liu_context
    ignored_reason = None
    if (
        event.payload.get("epoch_id") != context.epoch_id
        or event.payload.get("height") != context.height
        or event.payload.get("view") != context.view
        or event.payload.get("phase") is None
    ):
        ignored_reason = "timeout_payload_context_mismatch"
    elif not _valid_view_change_context(protocol, event):
        ignored_reason = "stale_epoch_or_height"
    elif protocol.node.id not in state.replica_ids:
        ignored_reason = "non_replica"
    elif context.view != state.current_view:
        ignored_reason = "stale_view"
    elif not timeout_scope_matches(protocol, event):
        ignored_reason = "stale_phase"
    elif (
        state.local_committed_height == state.height
        or state.fast_finalized_height == state.height
        or state.recovery_finalized_height == state.height
        or protocol.node.last_block.depth >= state.height
    ):
        ignored_reason = "height_committed_or_completed"
    elif state.pending_target_view is not None and state.pending_target_view > state.current_view:
        ignored_reason = "view_change_already_sent"
    if ignored_reason is not None:
        protocol.record_view_event(
            "recovery_timeout",
            "ignored",
            event.time,
            context.view,
            context.view + 1,
            detail=ignored_reason,
            height=context.height,
        )
        return "handled"

    target_view = state.current_view + 1
    state.pending_target_view = target_view
    previous = state.phase
    state.phase = LiuZyzzyvaPhase.VIEW_CHANGE_WAITING
    identity = protocol.timeout_identity()
    evidence = ZyzzyvaViewChangeEvidence(
        protocol.epoch_context.epoch_id,
        state.height,
        target_view,
        protocol.node.id,
        state.highest_speculative_evidence,
        state.recovery_commit_certificate,
        state.local_commit_digest,
        state.locked_digest,
    )
    carried = evidence.commit_certificate or evidence.highest_speculative_evidence
    protocol.record_phase(previous, state.phase, event.time, context.logical_message_id, identity)
    protocol.record_view_event(
        "recovery_timeout",
        "fired",
        event.time,
        state.current_view,
        target_view,
        carried_digest=None if carried is None else carried.block_digest,
    )
    protocol.record_view_event(
        "view_change_started",
        "started",
        event.time,
        state.current_view,
        target_view,
        carried_digest=None if carried is None else carried.block_digest,
    )
    protocol.record_view_event(
        "evidence_carried",
        "commit_certificate" if evidence.commit_certificate is not None else (
            "speculative_evidence" if evidence.highest_speculative_evidence is not None else "none"
        ),
        event.time,
        state.current_view,
        target_view,
        carried_digest=None if carried is None else carried.block_digest,
    )
    protocol.record_view_event(
        "view_change_sent",
        "sent",
        event.time,
        state.current_view,
        target_view,
        carried_digest=None if carried is None else carried.block_digest,
    )
    roles = protocol.roles_for(state.height, target_view)
    evidence_block = state.locked_block if evidence.commit_certificate is not None else state.block
    if protocol.node.id == roles.primary_id:
        return _register_view_change(protocol, evidence, evidence_block, event)
    Messages.send_view_change(protocol, roles.primary_id, event.time, identity, evidence, evidence_block)
    return "new_state"


def receive_view_change(protocol, event) -> str:
    if not _valid_view_change_context(protocol, event):
        protocol.record_view_event(
            "view_change_received",
            "ignored",
            event.time,
            protocol.state.current_view,
            event.liu_context.view,
            detail="stale_epoch_or_height",
        )
        return "handled"
    evidence = event.payload["evidence"]
    if (
        event.creator.id != evidence.sender_id
        or event.liu_context.view != evidence.target_view
        or evidence.epoch_id != event.liu_context.epoch_id
        or evidence.height != event.liu_context.height
    ):
        return "invalid"
    return _register_view_change(
        protocol,
        evidence,
        event.payload["evidence_block"],
        event,
    )


def receive_new_view(protocol, event) -> str:
    state = protocol.state
    certificate = event.payload["new_view_certificate"]
    target_view = certificate.target_view
    roles = protocol.roles_for(state.height, target_view)

    def reject(reason: str) -> str:
        protocol.record_view_event(
            "new_view_received",
            "rejected",
            event.time,
            state.current_view,
            target_view,
            len(certificate.signer_ids),
            certificate.safe_block_digest,
            reason,
        )
        return "invalid"

    if not _valid_view_change_context(protocol, event):
        return reject("stale_epoch_or_height")
    if certificate.epoch_id != protocol.epoch_context.epoch_id or certificate.height != state.height:
        return reject("certificate_epoch_or_height_mismatch")
    if event.liu_context.view != target_view or target_view <= state.current_view:
        return reject("stale_or_mismatched_target_view")
    if event.creator.id != roles.primary_id or certificate.new_primary_id != roles.primary_id:
        return reject("sender_is_not_new_primary")
    if protocol.node.id not in roles.replica_ids and protocol.node.id != roles.client_id:
        return reject("observer_not_participant")
    if state.local_committed_height == state.height:
        return reject("locally_committed_node_cannot_regress")
    if not certificate.validate(
        roles.replica_ids,
        roles.primary_id,
        protocol.runtime_configuration.view_change_quorum,
        protocol.runtime_configuration.speculative_safety_threshold,
        protocol.runtime_configuration.speculative_recovery_quorum,
    ):
        return reject("invalid_new_view_certificate")
    selected = certificate.selected_commit_certificate or certificate.selected_speculative_evidence
    selected_block = event.payload["selected_block"]
    if (
        protocol.node.id == roles.client_id
        and state.recovery_commit_certificate is not None
        and certificate.selected_commit_certificate != state.recovery_commit_certificate
    ):
        return reject("new_view_omits_client_commit_certificate")
    if selected is None:
        if selected_block is not None or state.locked_digest is not None:
            return reject("fresh_value_conflicts_with_existing_lock")
    else:
        identity = _identity_for_safety_evidence(selected)
        if selected_block is None or not protocol.valid_block_identity(selected_block, identity):
            return reject("selected_block_does_not_match_safety_evidence")
        if state.locked_digest is not None:
            if certificate.selected_commit_certificate is None or state.locked_digest != selected.block_digest:
                return reject("selected_value_conflicts_with_local_lock")
    if protocol.node.id == roles.client_id and selected is None and (state.block is None or state.block_identity is None):
        return reject("client_has_no_pending_request")
    old_view = state.current_view
    if not protocol.transition_to_view(
        target_view,
        certificate,
        event.time,
        event.liu_context.logical_message_id,
        selected_block,
    ):
        return reject("view_transition_rejected")
    protocol.record_view_event(
        "new_view_received",
        "accepted",
        event.time,
        old_view,
        target_view,
        len(certificate.signer_ids),
        certificate.safe_block_digest,
    )
    if certificate.selected_commit_certificate is not None:
        if protocol.node.id in roles.replica_ids:
            _resume_commit_recovery(protocol, event, certificate.selected_commit_certificate)
        else:
            protocol.record_view_event(
                "resumed_path",
                "waiting_for_recovery_local_commits",
                event.time,
                old_view,
                target_view,
                carried_digest=certificate.safe_block_digest,
            )
    else:
        _resume_speculative_or_fresh(protocol, event, certificate.selected_speculative_evidence)
    return "new_state"


def receive_finalized_block(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid:
        return "invalid"
    if action is not None:
        return action
    state = protocol.state
    identity = event.payload["identity"]
    block = event.payload["block"]
    finality_certificate = event.payload["finality_certificate"]
    commit_certificate = event.payload["commit_certificate"]
    roles = protocol.roles_for(identity.height, event.liu_context.view)
    if event.liu_context.block_identity != identity or event.creator.id != roles.client_id:
        return "invalid"
    if isinstance(finality_certificate, ZyzzyvaFastReplyCertificate):
        if commit_certificate is not None or not finality_certificate.validate(
            identity, event.liu_context.view, roles.replica_ids
        ):
            return "invalid"
    elif isinstance(finality_certificate, ZyzzyvaRecoveryFinalityCertificate):
        if commit_certificate is None or not finality_certificate.validate(
            identity,
            event.liu_context.view,
            roles.client_id,
            roles.replica_ids,
            protocol.runtime_configuration.local_commit_quorum,
            commit_certificate,
        ):
            return "invalid"
    else:
        return "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    if state.local_committed_height == identity.height and state.local_commit_digest != identity.block_digest:
        return "invalid"
    protocol.node.add_block(block.copy(), event.time, cause="certified_finalized_block_announcement")
    protocol.start(event.time, 0)
    return "new_state"
