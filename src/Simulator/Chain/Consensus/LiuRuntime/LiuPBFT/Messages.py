"""Local and directed LiuPBFT normal-path events."""

from Engine.Event import Event
from Chain.Consensus.LiuRuntime.Common.EventContext import LiuEventContext
from Chain.Consensus.LiuRuntime.Common.Identity import logical_message_id
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import scoped_timeout_payload


START_REQUEST = "lp_start_request"
REQUEST = "lp_request"
COMPLETE_REQUEST = "lp_complete_request"
PRE_PREPARE = "lp_pre_prepare"
COMPLETE_PREPREPARE = "lp_complete_preprepare"
PREPARE = "lp_prepare"
COMPLETE_PREPARE_QUORUM = "lp_complete_prepare_quorum"
COMMIT = "lp_commit"
COMPLETE_COMMIT_QUORUM = "lp_complete_commit_quorum"
REPLY = "lp_reply"
TIMEOUT = "lp_request_timeout"
FINALIZED_BLOCK = "lp_finalized_block"
VIEW_TIMEOUT = "lp_view_timeout"
VIEW_CHANGE = "lp_view_change"
NEW_VIEW = "lp_new_view"


def context_for(
    protocol,
    identity,
    message_type: str,
    sender_id: int,
    receiver_id: int,
    view: int | None = None,
) -> LiuEventContext:
    epoch = protocol.epoch_context
    selected_view = protocol.state.current_view if view is None else view
    return LiuEventContext(
        protocol.NAME,
        epoch.epoch_id,
        epoch.epoch_hash,
        epoch.epoch_configuration.validator_set,
        identity.height,
        selected_view,
        identity,
        logical_message_id(identity, message_type, sender_id, receiver_id, selected_view),
    )


def schedule_local(protocol, time: float, message_type: str, identity, **metadata):
    context = context_for(protocol, identity, message_type, protocol.node.id, protocol.node.id)
    event = Event(protocol.handle_event, protocol.node, time, {"CP": protocol.NAME, "type": message_type})
    event.liu_context = context
    for name, value in metadata.items():
        setattr(event, name, value)
    protocol.node.add_event(event)
    return event


def schedule_phase_timeout(protocol, time: float, message_type: str, identity, phase, **metadata):
    event = schedule_local(protocol, time, message_type, identity, **metadata)
    scoped_timeout_payload(event, phase)
    return event


def schedule_view_timeout(protocol, time: float, identity, phase=None):
    """Schedule a timeout whose payload and immutable context bind the same view."""
    return schedule_phase_timeout(protocol, time, VIEW_TIMEOUT, identity, protocol.state.phase if phase is None else phase)


def _send(protocol, receiver_id: int, time: float, message_type: str, identity, payload: dict):
    context = context_for(protocol, identity, message_type, protocol.node.id, receiver_id)
    payload = {"CP": protocol.NAME, "identity": identity, "type": message_type, **payload}
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, size)


def send_request(protocol, receiver_id: int, time: float, block, identity):
    return _send(protocol, receiver_id, time, REQUEST, identity, {"block": block.copy()})


def send_preprepare(protocol, receiver_id: int, time: float, block, identity):
    certificate = protocol.state.new_view_certificate
    return _send(
        protocol,
        receiver_id,
        time,
        PRE_PREPARE,
        identity,
        {"block": block.copy(), "new_view_certificate": certificate},
    )


def send_prepare(protocol, receiver_id: int, time: float, identity):
    return _send(protocol, receiver_id, time, PREPARE, identity, {})


def send_commit(protocol, receiver_id: int, time: float, identity, prepared_certificate):
    return _send(protocol, receiver_id, time, COMMIT, identity, {"prepared_certificate": prepared_certificate})


def send_reply(protocol, receiver_id: int, time: float, identity, commit_certificate):
    return _send(protocol, receiver_id, time, REPLY, identity, {"commit_certificate": commit_certificate})


def send_finalized_block(
    protocol,
    receiver_id: int,
    time: float,
    block,
    identity,
    finality_certificate,
    commit_certificates,
):
    return _send(
        protocol,
        receiver_id,
        time,
        FINALIZED_BLOCK,
        identity,
        {
            "block": block.copy(),
            "commit_certificates": commit_certificates,
            "finality_certificate": finality_certificate,
        },
    )


def send_view_change(protocol, receiver_id: int, time: float, identity, evidence, prepared_block):
    context = context_for(protocol, identity, VIEW_CHANGE, protocol.node.id, receiver_id, evidence.target_view)
    payload = {
        "CP": protocol.NAME,
        "evidence": evidence,
        "identity": identity,
        "prepared_block": None if prepared_block is None else prepared_block.copy(),
        "type": VIEW_CHANGE,
    }
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, size)


def send_new_view(protocol, receiver_id: int, time: float, identity, certificate, selected_block):
    context = context_for(protocol, identity, NEW_VIEW, protocol.node.id, receiver_id, certificate.target_view)
    payload = {
        "CP": protocol.NAME,
        "new_view_certificate": certificate,
        "identity": identity,
        "selected_block": None if selected_block is None else selected_block.copy(),
        "type": NEW_VIEW,
    }
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, size)
