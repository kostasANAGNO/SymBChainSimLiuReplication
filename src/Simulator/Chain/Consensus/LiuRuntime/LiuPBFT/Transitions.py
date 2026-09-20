"""Safety-preserving LiuPBFT normal-path and view-change transitions."""

from Parameters import Parameters

from Chain.Block import Block
from Chain.TransactionFactory import TransactionFactory
from Chain.Consensus.LiuRuntime.Common.Certificates import (
    PBFTCommitCertificate,
    PBFTFinalityCertificate,
    PBFTNewViewCertificate,
    PBFTPreparedCertificate,
    PBFTViewChangeEvidence,
    select_pbft_safe_prepared,
)
from Chain.Consensus.LiuRuntime.Common.Identity import LiuBlockIdentity, parent_digest
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import timeout_scope_matches
from Chain.Consensus.LiuRuntime.LiuPBFT import Messages
from Chain.Consensus.LiuRuntime.LiuPBFT.State import LiuPBFTPhase
from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork
from Utils.LiuRuntimeInstrumentation import (
    LiuRuntimeInstrumentationCollector,
    ProtocolCertificateRecord,
    ProtocolFinalityRecord,
    ProtocolTimeoutAdaptationRecord,
)


def _record_certificate(protocol, identity, certificate, certificate_type: str) -> None:
    LiuRuntimeInstrumentationCollector.certificates.append(
        ProtocolCertificateRecord(
            protocol.NAME,
            identity.epoch_id,
            identity.height,
            identity.block_digest,
            certificate.deterministic_hash(),
            certificate_type,
            certificate.signer_ids,
            certificate.threshold,
            certificate.created_at,
        )
    )


def _reserve_processing(protocol, event, work, phase, completion_type: str, work_name: str, **metadata) -> None:
    duration = protocol.processing_cost.duration_s(work, protocol.epoch_context.capability_ghz(protocol.node.id))
    reservation = protocol.cpu_queue.reserve(event.time, duration)
    previous = protocol.state.phase
    protocol.state.phase = phase
    protocol.record_phase(
        previous,
        phase,
        event.time,
        event.liu_context.logical_message_id,
        event.liu_context.block_identity,
        reservation.processing_start,
        reservation.processing_end,
        work_name,
        work,
    )
    Messages.schedule_local(
        protocol,
        reservation.processing_end,
        completion_type,
        event.liu_context.block_identity,
        liu_processing_reservation=reservation,
        liu_processing_work=work,
        **metadata,
    )


def start_request(protocol, event) -> str:
    state = protocol.state
    if protocol.node.id != state.client_id or state.phase is not LiuPBFTPhase.CLIENT_WAITING:
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
    state.block = block
    state.block_identity = identity
    state.request_sent_at = event.time
    Messages.send_request(protocol, state.primary_id, event.time, block, identity)
    protocol.arm_replica_phase_timeouts(event.time, identity)
    protocol.schedule_adaptive_timeout(event.time, Messages.TIMEOUT, identity, state.phase)
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
    if state.accepted_preprepare is not None or state.phase is not LiuPBFTPhase.WAITING_PREPREPARE:
        return "handled" if state.block_identity == identity else "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    if state.current_view > 0 and state.new_view_certificate is not None:
        if state.new_view_certificate.selected_prepared_certificate is not None:
            return "invalid"
    if not protocol.safe_value_allowed(identity):
        return "invalid"
    state.block = block
    state.block_identity = identity
    _reserve_processing(
        protocol,
        event,
        LiuProcessingWork.pbft_primary_request_work(),
        LiuPBFTPhase.PROCESSING_REQUEST,
        Messages.COMPLETE_REQUEST,
        "pbft_primary_request",
    )
    return "new_state"


def complete_primary_request(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuPBFTPhase.PROCESSING_REQUEST
        or state.block_identity != identity
    ):
        return "invalid"
    state.accepted_preprepare = identity.block_digest
    for backup_id in state.backup_ids:
        Messages.send_preprepare(protocol, backup_id, event.time, state.block, identity)
    protocol.emit_prepare(event.time, event.liu_context.logical_message_id)
    if state.current_view > 0:
        for backup_id in state.backup_ids:
            protocol.retransmit_authorized_proposal(backup_id, event.time)
    return "new_state"


def receive_preprepare(protocol, event) -> str:
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
    if state.accepted_preprepare is not None or state.phase is not LiuPBFTPhase.WAITING_PREPREPARE:
        return "handled" if state.block_identity == identity else "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    if state.current_view > 0:
        new_view_certificate = event.payload.get("new_view_certificate")
        if new_view_certificate is None or new_view_certificate != state.new_view_certificate:
            return "invalid"
    if not protocol.safe_value_allowed(identity):
        return "invalid"
    state.block = block
    state.block_identity = identity
    _reserve_processing(
        protocol,
        event,
        LiuProcessingWork.pbft_backup_preprepare_work(),
        LiuPBFTPhase.PROCESSING_PREPREPARE,
        Messages.COMPLETE_PREPREPARE,
        "pbft_backup_preprepare",
    )
    return "new_state"


def complete_backup_preprepare(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuPBFTPhase.PROCESSING_PREPREPARE
        or state.block_identity != identity
    ):
        return "invalid"
    state.accepted_preprepare = identity.block_digest
    protocol.emit_prepare(event.time, event.liu_context.logical_message_id)
    return "new_state"


def receive_prepare(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    signer = event.creator.id
    if event.liu_context.block_identity != identity or protocol.node.id not in state.replica_ids:
        return "invalid"
    if signer not in state.replica_ids or identity != state.block_identity:
        return "invalid"
    if state.accepted_preprepare != identity.block_digest:
        return "backlog"
    votes = state.prepare_votes.setdefault(identity.block_digest, {})
    if signer in votes:
        return "handled"
    votes[signer] = event.liu_context.logical_message_id
    if len(votes) >= protocol.runtime_configuration.prepare_quorum and not state.prepare_processing_started:
        state.prepare_processing_started = True
        signers = tuple(sorted(votes))[: protocol.runtime_configuration.prepare_quorum]
        _reserve_processing(
            protocol,
            event,
            LiuProcessingWork.pbft_prepare_quorum_work(len(state.replica_ids)),
            LiuPBFTPhase.PREPARE_PROCESSING,
            Messages.COMPLETE_PREPARE_QUORUM,
            "pbft_prepare_quorum",
            liu_certificate_signers=signers,
        )
        return "new_state"
    return "handled"


def complete_prepare_quorum(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuPBFTPhase.PREPARE_PROCESSING
        or state.prepared_certificate is not None
    ):
        return "invalid"
    certificate = PBFTPreparedCertificate(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        event.liu_certificate_signers,
        protocol.runtime_configuration.prepare_quorum,
        event.time,
    )
    if state.locked_digest is not None and state.locked_digest != identity.block_digest:
        state.safety_failure = "attempted_conflicting_prepare_lock"
        protocol.record_view_event(
            "prepared_certificate",
            "safety_error",
            event.time,
            state.current_view,
            state.current_view,
            len(certificate.signer_ids),
            identity.block_digest,
            state.safety_failure,
        )
        return "invalid"
    state.prepared_certificate = certificate
    if state.highest_prepared_certificate is None or certificate.view > state.highest_prepared_certificate.view:
        state.highest_prepared_certificate = certificate
    state.locked_digest = identity.block_digest
    state.locked_block = state.block.copy()
    _record_certificate(protocol, identity, certificate, "pbft_prepared")
    previous = state.phase
    state.phase = LiuPBFTPhase.PREPARED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    protocol.emit_commit(event.time, event.liu_context.logical_message_id)
    return "new_state"


def receive_commit(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    signer = event.creator.id
    sender_prepared = event.payload["prepared_certificate"]
    if event.liu_context.block_identity != identity or protocol.node.id not in state.replica_ids:
        return "invalid"
    if signer not in state.replica_ids or identity != state.block_identity:
        return "invalid"
    if not sender_prepared.validate(
        identity,
        state.current_view,
        state.replica_ids,
        protocol.runtime_configuration.prepare_quorum,
    ):
        return "invalid"
    if state.prepared_certificate is None:
        return "backlog"
    votes = state.commit_votes.setdefault(identity.block_digest, {})
    if signer in votes:
        return "handled"
    votes[signer] = event.liu_context.logical_message_id
    if (
        len(votes) >= protocol.runtime_configuration.commit_quorum
        and not state.commit_processing_started
        and (
            state.committed_height is None
            or (
                state.local_commit_certificate is not None
                and state.local_commit_certificate.block_digest == identity.block_digest
                and state.local_commit_certificate.view < state.current_view
            )
        )
    ):
        state.commit_processing_started = True
        signers = tuple(sorted(votes))[: protocol.runtime_configuration.commit_quorum]
        _reserve_processing(
            protocol,
            event,
            LiuProcessingWork.pbft_commit_quorum_work(len(state.replica_ids)),
            LiuPBFTPhase.COMMIT_PROCESSING,
            Messages.COMPLETE_COMMIT_QUORUM,
            "pbft_commit_quorum",
            liu_certificate_signers=signers,
        )
        return "new_state"
    return "handled"


def complete_commit_quorum(protocol, event) -> str:
    state = protocol.state
    identity = event.liu_context.block_identity
    if (
        event.liu_context.view != state.current_view
        or state.phase is not LiuPBFTPhase.COMMIT_PROCESSING
        or (
            state.committed_height is not None
            and (
                state.local_commit_certificate is None
                or state.local_commit_certificate.block_digest != identity.block_digest
                or state.local_commit_certificate.view >= state.current_view
            )
        )
    ):
        return "invalid"
    if state.locked_digest != identity.block_digest:
        state.safety_failure = "local_commit_conflicts_with_prepared_lock"
        return "invalid"
    certificate = PBFTCommitCertificate(
        identity.epoch_id,
        identity.height,
        state.current_view,
        identity.block_digest,
        identity.parent_digest,
        event.liu_certificate_signers,
        protocol.runtime_configuration.commit_quorum,
        event.time,
    )
    state.local_commit_certificate = certificate
    state.committed_height = identity.height
    _record_certificate(protocol, identity, certificate, "pbft_local_commit")
    previous = state.phase
    state.phase = LiuPBFTPhase.LOCAL_COMMITTED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    Messages.send_reply(protocol, state.client_id, event.time, identity, certificate)
    previous = state.phase
    state.phase = LiuPBFTPhase.REPLIED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    return "new_state"


def receive_reply(protocol, event) -> str:
    valid, action = protocol.validate_message(event)
    if not valid or action is not None:
        return "invalid"
    state = protocol.state
    identity = event.payload["identity"]
    signer = event.creator.id
    certificate = event.payload["commit_certificate"]
    if protocol.node.id != state.client_id or state.phase is not LiuPBFTPhase.CLIENT_WAITING:
        return "invalid"
    if event.liu_context.block_identity != identity or identity != state.block_identity or signer not in state.replica_ids:
        return "invalid"
    if not certificate.validate(identity, state.current_view, state.replica_ids, protocol.runtime_configuration.commit_quorum):
        return "invalid"
    replies = state.client_reply_votes.setdefault(identity.block_digest, {})
    if signer in replies:
        return "handled"
    replies[signer] = certificate
    if len(replies) < protocol.runtime_configuration.reply_quorum:
        return "handled"

    selected = tuple(sorted(replies.items()))[: protocol.runtime_configuration.reply_quorum]
    finality_certificate = PBFTFinalityCertificate(
        identity,
        state.current_view,
        state.client_id,
        tuple((node_id, commit_certificate.deterministic_hash()) for node_id, commit_certificate in selected),
        protocol.runtime_configuration.reply_quorum,
        event.time,
    )
    _record_certificate(protocol, identity, finality_certificate, "pbft_client_reply_quorum")
    finality_hash = finality_certificate.deterministic_hash()
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
            "pbft_normal_reply_quorum" if state.current_view == 0 else "pbft_view_change_reply_quorum",
            finality_hash,
        )
    )
    previous = state.phase
    state.phase = LiuPBFTPhase.PROTOCOL_FINALIZED
    protocol.record_phase(previous, state.phase, event.time, event.liu_context.logical_message_id, identity)
    protocol.node.add_block(state.block.copy(), event.time, cause="protocol_finality")
    commit_certificates = tuple(selected)
    for receiver in protocol.transport.nodes:
        if receiver.id != protocol.node.id:
            Messages.send_finalized_block(
                protocol,
                receiver.id,
                event.time,
                state.block,
                identity,
                finality_certificate,
                commit_certificates,
            )
    protocol.start(event.time, 0)
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
    commit_certificates = event.payload["commit_certificates"]
    roles = protocol.roles_for(identity.height, event.liu_context.view)
    if event.liu_context.block_identity != identity or event.creator.id != roles.client_id:
        return "invalid"
    if finality_certificate.view != event.liu_context.view:
        return "invalid"
    if not protocol.valid_block_identity(block, identity):
        return "invalid"
    if not finality_certificate.validate(
        roles.client_id,
        roles.replica_ids,
        identity,
        protocol.runtime_configuration.reply_quorum,
        commit_certificates,
        protocol.runtime_configuration.commit_quorum,
    ):
        return "invalid"
    protocol.node.add_block(block.copy(), event.time, cause="certified_finalized_block_announcement")
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


def _identity_for_prepared(certificate: PBFTPreparedCertificate) -> LiuBlockIdentity:
    return LiuBlockIdentity(
        certificate.epoch_id,
        certificate.height,
        certificate.parent_digest,
        certificate.block_digest,
    )


def _register_view_change(protocol, evidence, prepared_block, time: float, trigger_message_id: str) -> str:
    state = protocol.state
    target_view = evidence.target_view
    roles = protocol.roles_for(state.height, target_view)
    if protocol.node.id != roles.primary_id:
        protocol.record_view_event("view_change_received", "rejected", time, state.current_view, target_view, detail="not_target_primary")
        return "invalid"
    if evidence.epoch_id != protocol.epoch_context.epoch_id or evidence.height != state.height:
        protocol.record_view_event(
            "view_change_received", "rejected", time, state.current_view, target_view, detail="wrong_epoch_or_height"
        )
        return "invalid"
    if not evidence.validate(
        roles.replica_ids,
        protocol.runtime_configuration.prepare_quorum,
        protocol.runtime_configuration.commit_quorum,
    ):
        protocol.record_view_event("view_change_received", "rejected", time, state.current_view, target_view, detail="invalid_evidence")
        return "invalid"
    state.highest_observed_view = max(state.highest_observed_view, target_view)
    if target_view <= state.current_view:
        certificate = state.new_view_certificate
        if (
            certificate is not None
            and certificate.target_view == target_view
        ):
            certificate_hash = certificate.deterministic_hash()
            relay_key = (certificate_hash, evidence.sender_id)
            if relay_key not in state.catchup_responses:
                state.catchup_responses.add(relay_key)
                selected = certificate.selected_prepared_certificate
                selected_block = None if selected is None else state.locked_block
                Messages.send_new_view(
                    protocol,
                    evidence.sender_id,
                    time,
                    protocol.timeout_identity(),
                    certificate,
                    selected_block,
                )
                protocol.retransmit_authorized_proposal(evidence.sender_id, time)
                protocol.record_view_event(
                    "new_view_catchup_relay",
                    "sent",
                    time,
                    state.current_view,
                    certificate.target_view,
                    len(certificate.signer_ids),
                    None if selected is None else selected.block_digest,
                    "liu_pbft_view_catchup_v1",
                )
            else:
                protocol.record_view_event(
                    "new_view_catchup_relay",
                    "duplicate_ignored",
                    time,
                    state.current_view,
                    certificate.target_view,
                    len(certificate.signer_ids),
                    detail="bounded_per_certificate_sender",
                )
        protocol.record_view_event("view_change_received", "ignored", time, state.current_view, target_view, detail="stale_view")
        return "handled"
    if evidence.highest_prepared_certificate is not None:
        certificate = evidence.highest_prepared_certificate
        identity = _identity_for_prepared(certificate)
        if prepared_block is None or not protocol.valid_block_identity(prepared_block, identity):
            protocol.record_view_event(
                "view_change_received", "rejected", time, state.current_view, target_view, detail="missing_or_invalid_prepared_block"
            )
            return "invalid"
        state.view_change_blocks.setdefault(target_view, {})[certificate.block_digest] = prepared_block.copy()
    elif prepared_block is not None:
        return "invalid"

    evidence_by_sender: dict[int, PBFTViewChangeEvidence] = state.view_change_evidence.setdefault(target_view, {})
    if evidence.sender_id in evidence_by_sender:
        protocol.record_view_event(
            "view_change_received",
            "duplicate_ignored",
            time,
            state.current_view,
            target_view,
            len(evidence_by_sender),
        )
        return "handled"
    evidence_by_sender[evidence.sender_id] = evidence
    protocol.record_view_event(
        "view_change_received",
        "accepted",
        time,
        state.current_view,
        target_view,
        len(evidence_by_sender),
        None if evidence.highest_prepared_certificate is None else evidence.highest_prepared_certificate.block_digest,
    )
    threshold = protocol.runtime_configuration.view_change_quorum
    if len(evidence_by_sender) < threshold:
        return "handled"

    selected_evidence = tuple(evidence_by_sender[sender_id] for sender_id in sorted(evidence_by_sender)[:threshold])
    selected_prepared, safety_error = select_pbft_safe_prepared(selected_evidence)
    if safety_error is None and state.locked_digest is not None:
        local_view = -1 if state.highest_prepared_certificate is None else state.highest_prepared_certificate.view
        if selected_prepared is None:
            safety_error = "new_view_omits_local_prepared_lock"
        elif selected_prepared.block_digest != state.locked_digest and selected_prepared.view <= local_view:
            safety_error = "selected_value_does_not_supersede_primary_lock"
    if safety_error is not None:
        state.safety_failure = safety_error
        previous = state.phase
        state.phase = LiuPBFTPhase.FAILED
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
    if selected_prepared is not None:
        selected_block = state.view_change_blocks.get(target_view, {}).get(selected_prepared.block_digest)
        if selected_block is None:
            state.safety_failure = "selected_prepared_block_missing"
            protocol.record_view_event(
                "new_view_certificate",
                "safety_error",
                time,
                state.current_view,
                target_view,
                len(selected_evidence),
                selected_prepared.block_digest,
                state.safety_failure,
            )
            return "invalid"
    certificate = PBFTNewViewCertificate(
        protocol.epoch_context.epoch_id,
        state.height,
        target_view,
        selected_evidence,
        threshold,
        selected_prepared,
        time,
    )
    identity = protocol.timeout_identity() if selected_prepared is None else _identity_for_prepared(selected_prepared)
    _record_certificate(protocol, identity, certificate, "pbft_new_view")
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
        None if selected_prepared is None else selected_prepared.block_digest,
    )
    for validator_id in roles.replica_ids + (roles.client_id,):
        if validator_id != protocol.node.id:
            Messages.send_new_view(protocol, validator_id, time, identity, certificate, selected_block)
    if selected_prepared is not None:
        protocol.propose_carried_value(time, trigger_message_id)
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
    elif state.committed_height == state.height or protocol.node.last_block.depth >= state.height:
        ignored_reason = "height_committed_or_completed"
    elif state.pending_target_view is not None and state.pending_target_view > state.current_view:
        ignored_reason = "view_change_already_sent"
    if ignored_reason is not None:
        protocol.record_view_event(
            "timeout",
            "ignored",
            event.time,
            context.view,
            context.view + 1,
            detail=ignored_reason,
            height=context.height,
        )
        return "handled"

    configuration = protocol.runtime_configuration
    effective_timeout = float(event.payload.get("effective_timeout_s", protocol.effective_timeout_s()))
    LiuRuntimeInstrumentationCollector.timeout_adaptations.append(
        ProtocolTimeoutAdaptationRecord(
            protocol.NAME,
            configuration.TIMEOUT_BACKOFF_POLICY_VERSION,
            protocol.epoch_context.epoch_id,
            state.height,
            protocol.node.id,
            state.current_view,
            state.phase.value,
            "fired",
            event.time,
            state.timeout_backoff_count,
            effective_timeout,
            configuration.request_timeout_s,
            configuration.timeout_max_s,
            configuration.timeout_backoff_factor,
        )
    )
    if state.timeout_backoff_count >= configuration.maximum_timeout_view_changes:
        previous = state.phase
        state.phase = LiuPBFTPhase.VIEW_CHANGE_WAITING
        protocol.record_phase(previous, state.phase, event.time, context.logical_message_id, protocol.timeout_identity())
        protocol.record_view_event(
            "timeout_backoff", "cap_reached_waiting_for_finality", event.time, state.current_view,
            state.current_view, detail="bounded_local_timer_stopped",
        )
        return "handled"

    target_view = state.current_view + 1
    state.timeout_backoff_count += 1
    state.highest_observed_view = max(state.highest_observed_view, target_view)
    state.pending_target_view = target_view
    previous = state.phase
    state.phase = LiuPBFTPhase.VIEW_CHANGE_WAITING
    identity = protocol.timeout_identity()
    evidence = PBFTViewChangeEvidence(
        protocol.epoch_context.epoch_id,
        state.height,
        target_view,
        protocol.node.id,
        state.highest_prepared_certificate,
        state.local_commit_certificate,
    )
    protocol.record_phase(previous, state.phase, event.time, context.logical_message_id, identity)
    protocol.record_view_event(
        "timeout",
        "fired",
        event.time,
        state.current_view,
        target_view,
        carried_prepared_digest=(
            None if state.highest_prepared_certificate is None else state.highest_prepared_certificate.block_digest
        ),
    )
    roles = protocol.roles_for(state.height, target_view)
    protocol.record_view_event(
        "view_change_sent",
        "sent",
        event.time,
        state.current_view,
        target_view,
        carried_prepared_digest=(
            None if state.highest_prepared_certificate is None else state.highest_prepared_certificate.block_digest
        ),
    )
    prepared_block = state.locked_block if state.highest_prepared_certificate is not None else None
    if protocol.node.id == roles.primary_id:
        return _register_view_change(protocol, evidence, prepared_block, event.time, context.logical_message_id)
    Messages.send_view_change(protocol, roles.primary_id, event.time, identity, evidence, prepared_block)
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
        event.payload["prepared_block"],
        event.time,
        event.liu_context.logical_message_id,
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
            None if certificate.selected_prepared_certificate is None else certificate.selected_prepared_certificate.block_digest,
            reason,
        )
        return "invalid"

    if not _valid_view_change_context(protocol, event):
        return reject("stale_epoch_or_height")
    if event.liu_context.view != target_view or target_view <= state.current_view:
        return reject("stale_or_mismatched_target_view")
    if event.creator.id != roles.primary_id:
        return reject("sender_is_not_new_primary")
    if protocol.node.id not in roles.replica_ids and protocol.node.id != roles.client_id:
        return reject("observer_not_participant")
    if not certificate.validate(
        roles.replica_ids,
        protocol.runtime_configuration.view_change_quorum,
        protocol.runtime_configuration.prepare_quorum,
        protocol.runtime_configuration.commit_quorum,
    ):
        return reject("invalid_new_view_certificate")
    state.highest_observed_view = max(state.highest_observed_view, target_view)
    selected = certificate.selected_prepared_certificate
    selected_block = event.payload["selected_block"]
    if selected is None:
        if selected_block is not None or state.locked_digest is not None:
            return reject("fresh_value_conflicts_with_existing_lock")
    else:
        identity = _identity_for_prepared(selected)
        if selected_block is None or not protocol.valid_block_identity(selected_block, identity):
            return reject("selected_block_does_not_match_prepared_certificate")
        if state.locked_digest is not None and state.locked_digest != selected.block_digest:
            local_view = -1 if state.highest_prepared_certificate is None else state.highest_prepared_certificate.view
            if selected.view <= local_view:
                return reject("selected_value_does_not_supersede_local_lock")
    if state.committed_height == state.height:
        if selected is None or state.locked_digest != selected.block_digest:
            return reject("locally_committed_node_requires_same_safe_digest")
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
        None if selected is None else selected.block_digest,
    )
    if protocol.node.id == roles.client_id and selected is None:
        Messages.send_request(protocol, roles.primary_id, event.time, state.block, state.block_identity)
        protocol.schedule_adaptive_timeout(event.time, Messages.TIMEOUT, state.block_identity, state.phase)
    return "new_state"


def request_timeout(protocol, event) -> str:
    state = protocol.state
    context = event.liu_context
    if (
        not _valid_view_change_context(protocol, event)
        or context.view != state.current_view
        or state.committed_height == state.height
        or protocol.node.last_block.depth >= state.height
        or state.block_identity != context.block_identity
        or not timeout_scope_matches(protocol, event)
    ):
        protocol.record_view_event(
            "client_timeout",
            "ignored",
            event.time,
            context.view,
            context.view + 1,
            detail="stale_or_height_completed",
            height=context.height,
        )
        return "handled"
    protocol.record_view_event(
        "client_timeout",
        "fired_waiting_for_replica_view_change",
        event.time,
        state.current_view,
        state.current_view + 1,
    )
    return "handled"
