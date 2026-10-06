"""Pure immutable domain types for the Liu analytical consensus equations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import isclose
from typing import Any, TypeAlias

from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol, require_liu_protocol
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer


class LiuTransmissionUnitPolicy(str, Enum):
    """Versioned interpretation of the paper's dimensionally ambiguous S^B/R terms."""

    PAPER_LITERAL_MB_PER_MBPS_V1 = "paper_literal_mb_per_mbps_v1"
    SI_MEGABYTE_TO_MEGABIT_V1 = "si_megabyte_to_megabit_v1"

    @property
    def megabyte_to_rate_numerator_factor(self) -> float:
        if self is LiuTransmissionUnitPolicy.PAPER_LITERAL_MB_PER_MBPS_V1:
            return 1.0
        return 8.0


class ZyzzyvaAnalyticalPath(str, Enum):
    FAST = "fast"
    RECOVERY = "recovery"


DiagnosticScalar: TypeAlias = str | int | float | bool


def _require_optional_validator_id(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return require_integer(value, field_name)


@dataclass(frozen=True, slots=True)
class AnalyticalConsensusInput(CanonicalSerializable):
    """All explicit inputs needed by Liu Appendix B equations (15)-(17)."""

    validator_ids: tuple[int, ...]
    validator_count_k: int
    protocol: LiuConsensusProtocol
    client_validator_id: int
    primary_validator_id: int | None
    block_size_mb: float
    transaction_size_bytes: int
    batch_size_m: int
    block_interval_s: float
    computational_capabilities_ghz: tuple[float, ...]
    directed_link_rates_mbps: LinkStateMatrix
    signature_verification_cycles_alpha: float
    mac_operation_cycles_beta: float
    network_timeout_s: float
    faulty_replica_count: int
    zyzzyva_path: ZyzzyvaAnalyticalPath | None
    recovery_delay_s: float
    transmission_unit_policy: LiuTransmissionUnitPolicy

    def __post_init__(self) -> None:
        validator_ids = tuple(self.validator_ids)
        validator_count = require_integer(self.validator_count_k, "validator_count_k", minimum=1)
        if len(validator_ids) != validator_count:
            raise ValueError("validator_ids must contain exactly K entries")
        if any(isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 0 for node_id in validator_ids):
            raise ValueError("validator IDs must be non-negative integers")
        if len(set(validator_ids)) != validator_count:
            raise ValueError("validator IDs must be unique")

        protocol = require_liu_protocol(self.protocol)
        client_id = require_integer(self.client_validator_id, "client_validator_id")
        if client_id not in validator_ids:
            raise ValueError("client_validator_id must belong to validator_ids")
        primary_id = _require_optional_validator_id(self.primary_validator_id, "primary_validator_id")
        if primary_id is not None and primary_id not in validator_ids:
            raise ValueError("primary_validator_id must belong to validator_ids")
        if primary_id == client_id:
            raise ValueError("the Liu client/block producer and primary must be different validators")

        block_size = require_finite_number(self.block_size_mb, "block_size_mb", positive=True)
        transaction_size = require_integer(self.transaction_size_bytes, "transaction_size_bytes", minimum=1)
        batch_size = require_integer(self.batch_size_m, "batch_size_m", minimum=1)
        block_interval = require_finite_number(self.block_interval_s, "block_interval_s", positive=True)
        capabilities = tuple(
            require_finite_number(value, f"computational_capabilities_ghz[{index}]", positive=True)
            for index, value in enumerate(self.computational_capabilities_ghz)
        )
        if len(capabilities) != validator_count:
            raise ValueError("computational_capabilities_ghz must contain one value per ordered validator ID")
        if not isinstance(self.directed_link_rates_mbps, LinkStateMatrix):
            raise ValueError("directed_link_rates_mbps must be a LinkStateMatrix")
        if self.directed_link_rates_mbps.node_count != validator_count:
            raise ValueError("directed_link_rates_mbps must be a KxK matrix aligned with validator_ids")

        alpha = require_finite_number(
            self.signature_verification_cycles_alpha,
            "signature_verification_cycles_alpha",
            non_negative=True,
        )
        beta = require_finite_number(self.mac_operation_cycles_beta, "mac_operation_cycles_beta", non_negative=True)
        timeout = require_finite_number(self.network_timeout_s, "network_timeout_s", positive=True)
        faulty_count = require_integer(self.faulty_replica_count, "faulty_replica_count")
        recovery_delay = require_finite_number(self.recovery_delay_s, "recovery_delay_s", non_negative=True)
        if not isinstance(self.transmission_unit_policy, LiuTransmissionUnitPolicy):
            raise ValueError("transmission_unit_policy must be an explicit LiuTransmissionUnitPolicy")

        path = self.zyzzyva_path
        fault_bound = (validator_count - 1) // 3
        if protocol in (LiuConsensusProtocol.PBFT, LiuConsensusProtocol.ZYZZYVA):
            if validator_count < 3:
                raise ValueError("PBFT and Zyzzyva analytical equations require a client, primary, and at least one backup")
            if primary_id is None:
                raise ValueError("PBFT and Zyzzyva require an explicit primary_validator_id")
            if faulty_count > fault_bound:
                raise ValueError("faulty_replica_count exceeds floor((K-1)/3)")

        if protocol is LiuConsensusProtocol.PBFT:
            if path is not None:
                raise ValueError("zyzzyva_path applies only to Zyzzyva")
            if recovery_delay != 0.0:
                raise ValueError("recovery_delay_s applies only to the Zyzzyva recovery path")
        elif protocol is LiuConsensusProtocol.ZYZZYVA:
            if not isinstance(path, ZyzzyvaAnalyticalPath):
                raise ValueError("Zyzzyva requires an explicit ZyzzyvaAnalyticalPath")
            if path is ZyzzyvaAnalyticalPath.FAST:
                if faulty_count != 0:
                    raise ValueError("the Liu Zyzzyva fast path assumes zero faulty replicas")
                if recovery_delay != 0.0:
                    raise ValueError("the Zyzzyva fast path cannot include a recovery delay")
            elif faulty_count < 1 or recovery_delay <= 0.0:
                raise ValueError("the Liu Zyzzyva recovery path requires at least one fault and a positive recovery delay")
        else:
            if primary_id is not None:
                raise ValueError("LiuQuorum has no primary")
            if path is not None:
                raise ValueError("zyzzyva_path applies only to Zyzzyva")
            if batch_size != 1:
                raise ValueError("the LiuQuorum abstraction does not support batching; batch_size_m must equal 1")
            if faulty_count != 0:
                raise ValueError("LiuQuorum has tolerated-fault bound F^2 = 0")
            if recovery_delay != 0.0:
                raise ValueError("LiuQuorum has no recovery-delay term")
            if validator_count < 2:
                raise ValueError("LiuQuorum requires a client and at least one replica")

        object.__setattr__(self, "validator_ids", validator_ids)
        object.__setattr__(self, "validator_count_k", validator_count)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "client_validator_id", client_id)
        object.__setattr__(self, "primary_validator_id", primary_id)
        object.__setattr__(self, "block_size_mb", block_size)
        object.__setattr__(self, "transaction_size_bytes", transaction_size)
        object.__setattr__(self, "batch_size_m", batch_size)
        object.__setattr__(self, "block_interval_s", block_interval)
        object.__setattr__(self, "computational_capabilities_ghz", capabilities)
        object.__setattr__(self, "signature_verification_cycles_alpha", alpha)
        object.__setattr__(self, "mac_operation_cycles_beta", beta)
        object.__setattr__(self, "network_timeout_s", timeout)
        object.__setattr__(self, "faulty_replica_count", faulty_count)
        object.__setattr__(self, "recovery_delay_s", recovery_delay)

    def position_for(self, validator_id: int) -> int:
        try:
            return self.validator_ids.index(validator_id)
        except ValueError as error:
            raise KeyError(f"validator ID {validator_id} is not in the analytical input") from error

    def capability_ghz_for(self, validator_id: int) -> float:
        return self.computational_capabilities_ghz[self.position_for(validator_id)]

    def link_rate_mbps(self, sender_id: int, receiver_id: int) -> float:
        return self.directed_link_rates_mbps.rate(self.position_for(sender_id), self.position_for(receiver_id))

    @property
    def tolerated_faults(self) -> int:
        if self.protocol is LiuConsensusProtocol.LIU_QUORUM:
            return 0
        return (self.validator_count_k - 1) // 3

    @property
    def transfer_size_factor(self) -> float:
        return self.transmission_unit_policy.megabyte_to_rate_numerator_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size_m": self.batch_size_m,
            "block_interval_s": self.block_interval_s,
            "block_size_mb": self.block_size_mb,
            "client_validator_id": self.client_validator_id,
            "computational_capabilities_ghz": list(self.computational_capabilities_ghz),
            "directed_link_rates_mbps": self.directed_link_rates_mbps.to_dict(),
            "faulty_replica_count": self.faulty_replica_count,
            "mac_operation_cycles_beta": self.mac_operation_cycles_beta,
            "network_timeout_s": self.network_timeout_s,
            "primary_validator_id": self.primary_validator_id,
            "protocol": self.protocol.value,
            "recovery_delay_s": self.recovery_delay_s,
            "signature_verification_cycles_alpha": self.signature_verification_cycles_alpha,
            "transaction_size_bytes": self.transaction_size_bytes,
            "transmission_unit_policy": self.transmission_unit_policy.value,
            "validator_count_k": self.validator_count_k,
            "validator_ids": list(self.validator_ids),
            "zyzzyva_path": None if self.zyzzyva_path is None else self.zyzzyva_path.value,
        }


@dataclass(frozen=True, slots=True)
class AnalyticalConsensusResult(CanonicalSerializable):
    protocol: LiuConsensusProtocol
    delivery_delay_s: float
    validation_delay_s: float
    consensus_delay_s: float
    block_interval_s: float
    finality_delay_s: float
    tolerated_faults: int
    diagnostics: tuple[tuple[str, DiagnosticScalar], ...]

    def __post_init__(self) -> None:
        protocol = require_liu_protocol(self.protocol)
        delivery = require_finite_number(self.delivery_delay_s, "delivery_delay_s", non_negative=True)
        validation = require_finite_number(self.validation_delay_s, "validation_delay_s", non_negative=True)
        consensus = require_finite_number(self.consensus_delay_s, "consensus_delay_s", non_negative=True)
        block_interval = require_finite_number(self.block_interval_s, "block_interval_s", positive=True)
        finality = require_finite_number(self.finality_delay_s, "finality_delay_s", non_negative=True)
        tolerated_faults = require_integer(self.tolerated_faults, "tolerated_faults")
        if not isclose(consensus, delivery + validation, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("consensus_delay_s must equal delivery_delay_s + validation_delay_s")
        if not isclose(finality, block_interval + consensus, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("finality_delay_s must equal block_interval_s + consensus_delay_s")

        diagnostics = tuple(self.diagnostics)
        normalized: list[tuple[str, DiagnosticScalar]] = []
        names: set[str] = set()
        for item in diagnostics:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("diagnostics must contain immutable (name, scalar) tuples")
            name, value = item
            if not isinstance(name, str) or not name:
                raise ValueError("diagnostic names must be non-empty strings")
            if name in names:
                raise ValueError("diagnostic names must be unique")
            if isinstance(value, float):
                value = require_finite_number(value, f"diagnostics[{name}]")
            elif not isinstance(value, (str, int, bool)):
                raise ValueError("diagnostic values must be JSON scalar values")
            names.add(name)
            normalized.append((name, value))

        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "delivery_delay_s", delivery)
        object.__setattr__(self, "validation_delay_s", validation)
        object.__setattr__(self, "consensus_delay_s", consensus)
        object.__setattr__(self, "block_interval_s", block_interval)
        object.__setattr__(self, "finality_delay_s", finality)
        object.__setattr__(self, "tolerated_faults", tolerated_faults)
        object.__setattr__(self, "diagnostics", tuple(sorted(normalized)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_interval_s": self.block_interval_s,
            "consensus_delay_s": self.consensus_delay_s,
            "delivery_delay_s": self.delivery_delay_s,
            "diagnostics": [{"name": name, "value": value} for name, value in self.diagnostics],
            "finality_delay_s": self.finality_delay_s,
            "protocol": self.protocol.value,
            "tolerated_faults": self.tolerated_faults,
            "validation_delay_s": self.validation_delay_s,
        }


class AnalyticalConsensusModel(ABC):
    EQUATION_VERSION = "liu_appendix_b_equations_15_17_v1"

    @abstractmethod
    def evaluate(self, analytical_input: AnalyticalConsensusInput) -> AnalyticalConsensusResult:
        """Evaluate one of the Liu Appendix B analytical consensus abstractions."""

    @staticmethod
    def _result(
        analytical_input: AnalyticalConsensusInput,
        delivery_delay_s: float,
        validation_delay_s: float,
        diagnostics: list[tuple[str, DiagnosticScalar]],
    ) -> AnalyticalConsensusResult:
        consensus_delay = delivery_delay_s + validation_delay_s
        return AnalyticalConsensusResult(
            protocol=analytical_input.protocol,
            delivery_delay_s=delivery_delay_s,
            validation_delay_s=validation_delay_s,
            consensus_delay_s=consensus_delay,
            block_interval_s=analytical_input.block_interval_s,
            finality_delay_s=analytical_input.block_interval_s + consensus_delay,
            tolerated_faults=analytical_input.tolerated_faults,
            diagnostics=tuple(
                diagnostics
                + [
                    ("equation_version", AnalyticalConsensusModel.EQUATION_VERSION),
                    ("message_payload_model", "paper_uses_block_size_S_B_for_each_phase"),
                    ("network_timeout_operator", "min(serialization_delay,T)"),
                    ("processing_unit_policy", "cpu_cycles/(GHz*1e9)"),
                    ("transaction_size_usage", "not_used_by_appendix_b_equations_15_17"),
                    ("transmission_unit_policy", analytical_input.transmission_unit_policy.value),
                ]
            ),
        )

    @staticmethod
    def _processing_seconds(cpu_cycles: float, capability_ghz: float) -> float:
        return cpu_cycles / (capability_ghz * 1_000_000_000.0)

    @staticmethod
    def _bounded_transfer_seconds(
        analytical_input: AnalyticalConsensusInput,
        payload_mb: float,
        sender_id: int,
        receiver_ids: tuple[int, ...],
    ) -> float:
        if not receiver_ids:
            raise ValueError("a Liu delivery phase requires at least one receiver")
        worst_unbounded = max(
            payload_mb * analytical_input.transfer_size_factor / analytical_input.link_rate_mbps(sender_id, receiver_id)
            for receiver_id in receiver_ids
        )
        return min(worst_unbounded, analytical_input.network_timeout_s)

    @staticmethod
    def _bounded_all_to_all_seconds(
        analytical_input: AnalyticalConsensusInput,
        payload_mb: float,
        participant_ids: tuple[int, ...],
    ) -> float:
        rates = [
            analytical_input.link_rate_mbps(sender_id, receiver_id)
            for sender_id in participant_ids
            for receiver_id in participant_ids
            if sender_id != receiver_id
        ]
        if not rates:
            raise ValueError("a Liu all-to-all phase requires at least two participants")
        worst_unbounded = payload_mb * analytical_input.transfer_size_factor / min(rates)
        return min(worst_unbounded, analytical_input.network_timeout_s)
