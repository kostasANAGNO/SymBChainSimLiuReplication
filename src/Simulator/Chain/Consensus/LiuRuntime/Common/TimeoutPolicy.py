"""Versioned phase-relative timeout semantics shared by Liu runtime protocols."""


TIMEOUT_POLICY_VERSION = "liu_runtime_phase_relative_timeout_v2"


def phase_value(phase) -> str:
    """Return the canonical serialized phase name used in timeout payloads."""
    return phase.value if hasattr(phase, "value") else str(phase)


def scoped_timeout_payload(event, phase) -> None:
    """Bind a local timeout to its immutable epoch/height/view/phase context."""
    event.payload.update(
        {
            "epoch_id": event.liu_context.epoch_id,
            "height": event.liu_context.height,
            "view": event.liu_context.view,
            "phase": phase_value(phase),
            "timeout_policy": TIMEOUT_POLICY_VERSION,
        }
    )


def timeout_scope_matches(protocol, event) -> bool:
    """Whether the timer still belongs to the node's current protocol phase."""
    context = event.liu_context
    payload = event.payload
    return (
        payload.get("timeout_policy") == TIMEOUT_POLICY_VERSION
        and payload.get("epoch_id") == context.epoch_id
        and payload.get("height") == context.height
        and payload.get("view") == context.view
        and payload.get("phase") == phase_value(protocol.state.phase)
    )
