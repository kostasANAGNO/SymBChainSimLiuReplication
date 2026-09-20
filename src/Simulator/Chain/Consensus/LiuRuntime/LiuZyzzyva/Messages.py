"""Local and directed events for the Liu Zyzzyva fast and recovery paths."""

from Engine.Event import Event
from Chain.Consensus.LiuRuntime.Common.EventContext import LiuEventContext
from Chain.Consensus.LiuRuntime.Common.Identity import logical_message_id
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import scoped_timeout_payload


START_REQUEST = "lz_start_request"
REQUEST = "lz_request"
COMPLETE_PRIMARY_ORDER = "lz_complete_primary_order"
ORDER_REQUEST = "lz_order_request"
COMPLETE_SPECULATIVE = "lz_complete_speculative"
SPECULATIVE_REPLY = "lz_speculative_reply"
TIMEOUT = "lz_fast_timeout"
COMMIT_CERTIFICATE = "lz_commit_certificate"
COMPLETE_RECOVERY = "lz_complete_recovery"
LOCAL_COMMIT = "lz_local_commit"
FINALIZED_BLOCK = "lz_finalized_block"
VIEW_TIMEOUT = "lz_view_timeout"
VIEW_CHANGE = "lz_view_change"
NEW_VIEW = "lz_new_view"


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


def schedule_phase_timeout(protocol, time: float, message_type: str, identity, phase):
    event = schedule_local(protocol, time, message_type, identity)
    scoped_timeout_payload(event, phase)
    return event


def schedule_view_timeout(protocol, time: float, identity, phase=None):
    return schedule_phase_timeout(protocol, time, VIEW_TIMEOUT, identity, protocol.state.phase if phase is None else phase)


def _send(protocol, receiver_id: int, time: float, message_type: str, identity, payload: dict):
    context = context_for(protocol, identity, message_type, protocol.node.id, receiver_id)
    message = {"CP": protocol.NAME, "identity": identity, "type": message_type, **payload}
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, message, protocol.handle_event, context, size)


def send_request(protocol, receiver_id: int, time: float, block, identity):
    return _send(protocol, receiver_id, time, REQUEST, identity, {"block": block.copy()})


def send_order_request(protocol, receiver_id: int, time: float, block, identity):
    return _send(protocol, receiver_id, time, ORDER_REQUEST, identity, {"block": block.copy()})


def send_speculative_reply(protocol, receiver_id: int, time: float, identity, reply_digest: str):
    return _send(protocol, receiver_id, time, SPECULATIVE_REPLY, identity, {"reply_digest": reply_digest})


def send_commit_certificate(protocol, receiver_id: int, time: float, identity, certificate):
    return _send(protocol, receiver_id, time, COMMIT_CERTIFICATE, identity, {"commit_certificate": certificate})


def send_local_commit(protocol, receiver_id: int, time: float, identity, commit_certificate):
    return _send(
        protocol,
        receiver_id,
        time,
        LOCAL_COMMIT,
        identity,
        {
            "commit_certificate_hash": commit_certificate.deterministic_hash(),
            "local_commit_digest": identity.block_digest,
        },
    )


def send_view_change(protocol, receiver_id: int, time: float, identity, evidence, evidence_block):
    context = context_for(protocol, identity, VIEW_CHANGE, protocol.node.id, receiver_id, evidence.target_view)
    payload = {
        "CP": protocol.NAME,
        "evidence": evidence,
        "evidence_block": None if evidence_block is None else evidence_block.copy(),
        "identity": identity,
        "type": VIEW_CHANGE,
    }
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, size)


def send_new_view(protocol, receiver_id: int, time: float, identity, certificate, selected_block):
    context = context_for(protocol, identity, NEW_VIEW, protocol.node.id, receiver_id, certificate.target_view)
    payload = {
        "CP": protocol.NAME,
        "identity": identity,
        "new_view_certificate": certificate,
        "selected_block": None if selected_block is None else selected_block.copy(),
        "type": NEW_VIEW,
    }
    size = protocol.message_size_policy.request_size_mb(protocol.epoch_context.epoch_configuration.block_size_mb)
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, size)


def send_finalized_block(
    protocol,
    receiver_id: int,
    time: float,
    block,
    identity,
    finality_certificate,
    commit_certificate=None,
):
    return _send(
        protocol,
        receiver_id,
        time,
        FINALIZED_BLOCK,
        identity,
        {
            "block": block.copy(),
            "commit_certificate": commit_certificate,
            "finality_certificate": finality_certificate,
        },
    )
