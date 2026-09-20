"""Fixed directed R_ij transport without changing legacy Network arithmetic."""

from Engine.Event import MessageEvent
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector, ProtocolMessageRecord


class LiuDirectedTransport:
    def __init__(self, nodes, epoch_context, propagation_delay_s: float = 0.0) -> None:
        if propagation_delay_s < 0:
            raise ValueError("propagation_delay_s cannot be negative")
        self.nodes = tuple(nodes)
        self.epoch_context = epoch_context
        self.propagation_delay_s = float(propagation_delay_s)

    def send(self, sender, receiver_id: int, sent_at: float, payload: dict, handler, context, payload_size_mb: float):
        if payload_size_mb <= 0:
            raise ValueError("Liu message payload size must be positive MB")
        receiver = self.nodes[receiver_id]
        rate = self.epoch_context.link_state_matrix.rate(sender.id, receiver_id)
        serialization = 8.0 * payload_size_mb / rate
        arrival = sent_at + serialization + self.propagation_delay_s
        event = MessageEvent(handler, sender, arrival, payload, -1, receiver)
        event.recipient_scope = "direct"
        event.liu_context = context
        receiver.add_event(event)
        LiuRuntimeInstrumentationCollector.protocol_messages.append(
            ProtocolMessageRecord(
                logical_message_id=context.logical_message_id,
                protocol=context.protocol,
                message_type=payload["type"],
                epoch_id=context.epoch_id,
                height=context.height,
                block_digest=context.block_identity.block_digest,
                parent_digest=context.block_identity.parent_digest,
                sender_id=sender.id,
                receiver_id=receiver_id,
                sent_at=sent_at,
                arrival_at=arrival,
                rate_mbps=rate,
                payload_size_mb=payload_size_mb,
                serialization_delay_s=serialization,
                propagation_delay_s=self.propagation_delay_s,
            )
        )
        return event

