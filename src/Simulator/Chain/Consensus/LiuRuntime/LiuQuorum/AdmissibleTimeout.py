"""State/action-aware LiuQuorum request timeout (RECONSTRUCTION_REQUIRED).

`liu_quorum_state_action_timeout_v1`

Liu et al. state only that a consensus timeout ``T`` exists; no numerical value
and no sizing rule are reported. The frozen 5 s pilot timeout is a reconstruction
that is smaller than the real all-replies round-trip of a valid large-block
action over a slow directed-link snapshot (e.g. an 8 MB block whose slowest
required replica sits on a 25 Mbps link in both directions needs 5.12 s). The
client then fails mid-execution even though every replica accepted and every
reply is eventually produced.

This policy sizes the client's single request timer *before dispatch* from the
observable epoch state ``S_t`` (directed link matrix, node capabilities) and the
action ``A_t`` (block size, selected validators), so a legitimately slow but
correct round-trip is not failed. It changes no protocol semantics: LiuQuorum
still requires all ``K-1`` replies, keeps ``F=0``, and the Liu reward and the
C1/C2/C3 constraints are untouched. The timer keeps its role as a liveness
fallback for a replica that never replies.

The estimate uses only ``S_t`` and ``A_t`` — never a reward, Q-value, replay
sample, future state, global RNG, or any DES outcome — so it is deterministic
and introduces no hidden action pruning: every action still executes.
"""

from Chain.Consensus.LiuRuntime.Processing.ProcessingWork import LiuProcessingWork


QUORUM_TIMEOUT_POLICY_VERSION = "liu_quorum_state_action_timeout_v1"


def estimated_slowest_required_round_trip_s(
    *,
    link_state_matrix,
    capability_ghz,
    processing_cost,
    message_size_policy,
    propagation_delay_s: float,
    client_id: int,
    replica_ids,
    block_size_mb: float,
) -> float:
    """Slowest client->replica->client completion over the required replica set.

    LiuQuorum finalizes only when *all* required replies arrive, so the relevant
    runtime reference is the maximum single-replica round-trip, not the mean.
    Each replica path mirrors the existing directed transport and Liu CPU model:
    request serialization + replica processing + reply serialization (plus the
    configured per-hop propagation delay).
    """
    replica_ids = tuple(replica_ids)
    if not replica_ids:
        raise ValueError("LiuQuorum timeout estimate requires at least one required replica")
    if propagation_delay_s < 0:
        raise ValueError("propagation_delay_s cannot be negative")
    work = LiuProcessingWork.quorum_replica_work()
    request_mb = message_size_policy.request_size_mb(block_size_mb)
    reply_mb = message_size_policy.reply_size_mb(block_size_mb)
    slowest = 0.0
    for replica_id in replica_ids:
        request_s = 8.0 * request_mb / link_state_matrix.rate(client_id, replica_id) + propagation_delay_s
        reply_s = 8.0 * reply_mb / link_state_matrix.rate(replica_id, client_id) + propagation_delay_s
        processing_s = processing_cost.duration_s(work, capability_ghz(replica_id))
        slowest = max(slowest, request_s + processing_s + reply_s)
    return slowest
