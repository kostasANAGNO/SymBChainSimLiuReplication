"""Liu Appendix B Zyzzyva fast/recovery performance abstraction."""

from PaperReference.AnalyticalConsensus import AnalyticalConsensusInput, AnalyticalConsensusModel, AnalyticalConsensusResult, ZyzzyvaAnalyticalPath
from Liu.Protocol import LiuConsensusProtocol


class ZyzzyvaAnalyticalModel(AnalyticalConsensusModel):
    def evaluate(self, analytical_input: AnalyticalConsensusInput) -> AnalyticalConsensusResult:
        if not isinstance(analytical_input, AnalyticalConsensusInput):
            raise ValueError("analytical_input must be AnalyticalConsensusInput")
        if analytical_input.protocol is not LiuConsensusProtocol.ZYZZYVA:
            raise ValueError("ZyzzyvaAnalyticalModel requires protocol ZYZZYVA")

        client = analytical_input.client_validator_id
        primary = analytical_input.primary_validator_id
        replicas = tuple(validator_id for validator_id in analytical_input.validator_ids if validator_id != client)
        backups = tuple(validator_id for validator_id in replicas if validator_id != primary)
        batch_size = analytical_input.batch_size_m
        k = analytical_input.validator_count_k
        faults = analytical_input.faulty_replica_count
        alpha = analytical_input.signature_verification_cycles_alpha
        beta = analytical_input.mac_operation_cycles_beta
        recovery = analytical_input.zyzzyva_path is ZyzzyvaAnalyticalPath.RECOVERY

        if recovery:
            primary_cycles = batch_size * alpha + (4 * batch_size + k + faults - 1) * beta
            backup_cycles = batch_size * alpha + (3 * batch_size + 1) * beta
        else:
            primary_cycles = batch_size * alpha + (2 * batch_size + k - 1) * beta
            backup_cycles = batch_size * alpha + (batch_size + 1) * beta
        validation_delay = max(
            self._processing_seconds(
                primary_cycles if validator_id == primary else backup_cycles,
                analytical_input.capability_ghz_for(validator_id),
            )
            / batch_size
            for validator_id in replicas
        )

        payload_mb = batch_size * analytical_input.block_size_mb
        t1 = self._bounded_transfer_seconds(analytical_input, payload_mb, client, (primary,))
        t2 = self._bounded_transfer_seconds(analytical_input, payload_mb, primary, backups)
        t3 = max(
            min(
                payload_mb * analytical_input.transfer_size_factor / analytical_input.link_rate_mbps(replica, client),
                analytical_input.network_timeout_s,
            )
            for replica in replicas
        )
        phase_delays = t1 + t2 + t3
        diagnostics = [
            ("analytical_path", "recovery" if recovery else "fast"),
            ("backup_cpu_cycles_per_batch", backup_cycles),
            ("faulty_replica_count", faults),
            ("primary_cpu_cycles_per_batch", primary_cycles),
            ("t1_request_s_per_batch", t1),
            ("t2_order_request_s_per_batch", t2),
            ("t3_speculative_reply_s_per_batch", t3),
        ]
        if recovery:
            t4 = self._bounded_transfer_seconds(analytical_input, payload_mb, client, replicas)
            t5 = t3
            phase_delays += analytical_input.recovery_delay_s + t4 + t5
            diagnostics.extend(
                [
                    ("recovery_delay_s_per_batch", analytical_input.recovery_delay_s),
                    ("t4_commit_s_per_batch", t4),
                    ("t5_reply_s_per_batch", t5),
                ]
            )
        delivery_delay = phase_delays / batch_size
        return self._result(analytical_input, delivery_delay, validation_delay, diagnostics)
