"""Liu Appendix B PBFT performance abstraction, independent of runtime PBFT."""

from Liu.AnalyticalConsensus import AnalyticalConsensusInput, AnalyticalConsensusModel, AnalyticalConsensusResult
from Liu.Protocol import LiuConsensusProtocol


class PBFTAnalyticalModel(AnalyticalConsensusModel):
    def evaluate(self, analytical_input: AnalyticalConsensusInput) -> AnalyticalConsensusResult:
        if not isinstance(analytical_input, AnalyticalConsensusInput):
            raise ValueError("analytical_input must be AnalyticalConsensusInput")
        if analytical_input.protocol is not LiuConsensusProtocol.PBFT:
            raise ValueError("PBFTAnalyticalModel requires protocol PBFT")

        client = analytical_input.client_validator_id
        primary = analytical_input.primary_validator_id
        replicas = tuple(validator_id for validator_id in analytical_input.validator_ids if validator_id != client)
        backups = tuple(validator_id for validator_id in replicas if validator_id != primary)
        batch_size = analytical_input.batch_size_m
        k = analytical_input.validator_count_k
        faults = analytical_input.faulty_replica_count
        alpha = analytical_input.signature_verification_cycles_alpha
        beta = analytical_input.mac_operation_cycles_beta

        primary_cycles = batch_size * alpha + (2 * batch_size + 4 * (k + faults - 1)) * beta
        backup_cycles = batch_size * alpha + (batch_size + 4 * (k + faults - 1)) * beta
        per_request_processing = [
            self._processing_seconds(
                primary_cycles if validator_id == primary else backup_cycles,
                analytical_input.capability_ghz_for(validator_id),
            )
            / batch_size
            for validator_id in replicas
        ]
        validation_delay = max(per_request_processing)

        payload_mb = batch_size * analytical_input.block_size_mb
        t1 = self._bounded_transfer_seconds(analytical_input, payload_mb, client, (primary,))
        t2 = self._bounded_transfer_seconds(analytical_input, payload_mb, primary, backups)
        t3 = self._bounded_all_to_all_seconds(analytical_input, payload_mb, replicas)
        t4 = self._bounded_all_to_all_seconds(analytical_input, payload_mb, replicas)
        t5 = max(
            min(
                payload_mb * analytical_input.transfer_size_factor / analytical_input.link_rate_mbps(replica, client),
                analytical_input.network_timeout_s,
            )
            for replica in replicas
        )
        delivery_delay = (t1 + t2 + t3 + t4 + t5) / batch_size

        return self._result(
            analytical_input,
            delivery_delay,
            validation_delay,
            [
                ("analytical_path", "pbft"),
                ("backup_cpu_cycles_per_batch", backup_cycles),
                ("faulty_replica_count", faults),
                ("primary_cpu_cycles_per_batch", primary_cycles),
                ("t1_request_s_per_batch", t1),
                ("t2_pre_prepare_s_per_batch", t2),
                ("t3_prepare_s_per_batch", t3),
                ("t4_commit_s_per_batch", t4),
                ("t5_reply_s_per_batch", t5),
            ],
        )
