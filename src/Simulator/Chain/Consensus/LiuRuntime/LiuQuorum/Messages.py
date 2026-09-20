"""Creation of local and directed LiuQuorum events."""

from Engine.Event import Event
from Chain.Consensus.LiuRuntime.Common.EventContext import LiuEventContext
from Chain.Consensus.LiuRuntime.Common.Identity import logical_message_id
from Chain.Consensus.LiuRuntime.Common.TimeoutPolicy import scoped_timeout_payload


START_REQUEST = "lq_start_request"
REQUEST = "lq_request"
PROCESS_REQUEST_COMPLETE = "lq_process_request_complete"
REPLY = "lq_reply"
TIMEOUT = "lq_request_timeout"
FINALIZED_BLOCK = "lq_finalized_block"


def context_for(protocol, identity, message_type: str, sender_id: int, receiver_id: int) -> LiuEventContext:
    epoch = protocol.epoch_context
    return LiuEventContext(
        protocol=protocol.NAME,
        epoch_id=epoch.epoch_id,
        epoch_hash=epoch.epoch_hash,
        validator_set_snapshot=epoch.epoch_configuration.validator_set,
        height=identity.height,
        view=protocol.rounds.round,
        block_identity=identity,
        logical_message_id=logical_message_id(identity, message_type, sender_id, receiver_id, protocol.rounds.round),
    )


def schedule_local(protocol, time: float, message_type: str, identity):
    context = context_for(protocol, identity, message_type, protocol.node.id, protocol.node.id)
    payload = {"CP": protocol.NAME, "type": message_type}
    event = Event(protocol.handle_event, protocol.node, time, payload)
    event.liu_context = context
    protocol.node.add_event(event)
    return event


def schedule_processing_complete(protocol, time: float, identity, reservation):
    event = schedule_local(protocol, time, PROCESS_REQUEST_COMPLETE, identity)
    event.liu_processing_reservation = reservation
    return event


def schedule_timeout(protocol, time: float, identity, phase):
    event = schedule_local(protocol, time, TIMEOUT, identity)
    scoped_timeout_payload(event, phase)
    return event


def send_request(protocol, receiver_id: int, time: float, block, identity, payload_size_mb: float):
    context = context_for(protocol, identity, REQUEST, protocol.node.id, receiver_id)
    payload = {"CP": protocol.NAME, "block": block.copy(), "identity": identity, "type": REQUEST}
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, payload_size_mb)


def send_reply(protocol, receiver_id: int, time: float, identity, reply_digest: str, payload_size_mb: float):
    context = context_for(protocol, identity, REPLY, protocol.node.id, receiver_id)
    payload = {"CP": protocol.NAME, "identity": identity, "reply_digest": reply_digest, "type": REPLY}
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, payload_size_mb)


def send_finalized_block(protocol, receiver_id: int, time: float, block, identity, certificate, payload_size_mb: float):
    context = context_for(protocol, identity, FINALIZED_BLOCK, protocol.node.id, receiver_id)
    payload = {
        "CP": protocol.NAME,
        "block": block.copy(),
        "certificate": certificate,
        "identity": identity,
        "type": FINALIZED_BLOCK,
    }
    return protocol.transport.send(protocol.node, receiver_id, time, payload, protocol.handle_event, context, payload_size_mb)
