"""The paper's simplified, leaderless LiuQuorum analytical abstraction."""

from PaperReference.AnalyticalConsensus import AnalyticalConsensusInput, AnalyticalConsensusModel, AnalyticalConsensusResult
from Liu.Protocol import LiuConsensusProtocol


class LiuQuorumAnalyticalModel(AnalyticalConsensusModel):
    def evaluate(self, analytical_input: AnalyticalConsensusInput) -> AnalyticalConsensusResult:
        if not isinstance(analytical_input, AnalyticalConsensusInput):
            raise ValueError("analytical_input must be AnalyticalConsensusInput")
        if analytical_input.protocol is not LiuConsensusProtocol.LIU_QUORUM:
            raise ValueError("LiuQuorumAnalyticalModel requires protocol LIU_QUORUM")

        client = analytical_input.client_validator_id
        replicas = tuple(validator_id for validator_id in analytical_input.validator_ids if validator_id != client)
        alpha = analytical_input.signature_verification_cycles_alpha
        beta = analytical_input.mac_operation_cycles_beta
        replica_cycles = alpha + 2 * beta
        validation_delay = max(
            self._processing_seconds(replica_cycles, analytical_input.capability_ghz_for(replica))
            for replica in replicas
        )

        t1 = self._bounded_transfer_seconds(analytical_input, analytical_input.block_size_mb, client, replicas)
        t2 = max(
            min(
                analytical_input.block_size_mb
                * analytical_input.transfer_size_factor
                / analytical_input.link_rate_mbps(replica, client),
                analytical_input.network_timeout_s,
            )
            for replica in replicas
        )
        return self._result(
            analytical_input,
            t1 + t2,
            validation_delay,
            [
                ("analytical_path", "two_hop_no_primary_no_batching"),
                ("replica_cpu_cycles_per_request", replica_cycles),
                ("t1_client_broadcast_s", t1),
                ("t2_replica_reply_s", t2),
            ],
        )
