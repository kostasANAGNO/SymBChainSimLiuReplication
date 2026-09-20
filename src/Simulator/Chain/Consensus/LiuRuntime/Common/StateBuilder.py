"""Build immutable Liu states from active runtime snapshots without prediction."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable

from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Liu.Serialization import CanonicalSerializable
from Liu.Spatial import SpatialProfileSet
from Liu.SpatialIntensity import GeographicGiniResult, GridSpatialIntensityModel
from Liu.State import LiuState
from Liu.Validation import require_finite_number
from Utils.DecentralizationMetrics import canonical_pairwise_gini
from Utils.Instrumentation import TransactionCreationRecord
from Utils.LiuRuntimeInstrumentation import LiuRuntimeInstrumentationCollector, RuntimeLiuStateRecord


@dataclass(frozen=True, slots=True)
class LiuRuntimeStateBuildResult(CanonicalSerializable):
    epoch_id: int
    state: LiuState
    state_hash: str
    chi_policy_version: str
    transaction_size_unit_policy: str
    chi_source: str
    chi_sample_count: int
    action_hash: str | None
    previous_state_hash: str | None
    stake_gini: float | None
    stake_gini_unavailable_reason: str | None
    geographic_gini: GeographicGiniResult | None

    def to_dict(self) -> dict:
        return {
            "action_hash": self.action_hash,
            "chi_policy_version": self.chi_policy_version,
            "transaction_size_unit_policy": self.transaction_size_unit_policy,
            "chi_sample_count": self.chi_sample_count,
            "chi_source": self.chi_source,
            "epoch_id": self.epoch_id,
            "geographic_gini": None if self.geographic_gini is None else self.geographic_gini.to_dict(),
            "previous_state_hash": self.previous_state_hash,
            "stake_gini": self.stake_gini,
            "stake_gini_unavailable_reason": self.stake_gini_unavailable_reason,
            "state": self.state.to_dict(),
            "state_hash": self.state_hash,
        }


class LiuRuntimeStateBuilder:
    CHI_POLICY_VERSION = "configured_then_previous_epoch_empirical_mean_v1"
    TRANSACTION_SIZE_UNIT_POLICY = "simulator_mb_to_si_bytes_v1"

    def __init__(
        self,
        spatial_profiles: SpatialProfileSet,
        configured_workload_mean_transaction_size_mb: float,
        geographic_model: GridSpatialIntensityModel | None = None,
    ) -> None:
        if not isinstance(spatial_profiles, SpatialProfileSet):
            raise ValueError("spatial_profiles must be an immutable SpatialProfileSet")
        self.spatial_profiles = spatial_profiles
        self.configured_workload_mean_transaction_size_mb = require_finite_number(
            configured_workload_mean_transaction_size_mb,
            "configured_workload_mean_transaction_size_mb",
            positive=True,
        )
        if geographic_model is not None:
            if not isinstance(geographic_model, GridSpatialIntensityModel):
                raise ValueError("geographic_model must be a GridSpatialIntensityModel")
            if geographic_model.spatial_profiles != spatial_profiles:
                raise ValueError("geographic model must use the StateBuilder spatial profiles")
        self.geographic_model = geographic_model

    @classmethod
    def from_parameters(
        cls,
        spatial_profiles: SpatialProfileSet,
        geographic_model: GridSpatialIntensityModel | None = None,
    ) -> "LiuRuntimeStateBuilder":
        from Parameters import Parameters

        configured_mean_mb = (
            float(Parameters.application["base_transaction_size"])
            + float(Parameters.application["Tsize"])
        )
        return cls(spatial_profiles, configured_mean_mb, geographic_model)

    @staticmethod
    def _sha256_or_none(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field} must be a SHA-256 hexadecimal string")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{field} must be a SHA-256 hexadecimal string") from error
        return value

    def build(
        self,
        epoch_context: LiuRuntimeEpochContext,
        *,
        observation_time: float,
        is_initial_state: bool,
        previous_epoch_transactions: Iterable[TransactionCreationRecord] = (),
        action_hash: str | None = None,
        previous_state_hash: str | None = None,
    ) -> LiuRuntimeStateBuildResult:
        if not isinstance(epoch_context, LiuRuntimeEpochContext):
            raise ValueError("epoch_context must be a LiuRuntimeEpochContext")
        observation_time = require_finite_number(observation_time, "observation_time", non_negative=True)
        action_hash = self._sha256_or_none(action_hash, "action_hash")
        previous_state_hash = self._sha256_or_none(previous_state_hash, "previous_state_hash")
        profiles = epoch_context.node_profiles.profiles
        node_count = len(profiles)
        if self.spatial_profiles.node_count != node_count:
            raise ValueError("spatial profiles and runtime profiles must describe the same N")
        stakes = tuple(profile.stake_tokens for profile in profiles)
        capabilities = tuple(profile.computational_capability_ghz for profile in profiles)
        if any(value is None for value in stakes):
            raise ValueError("runtime LiuState requires explicit stake for every node")
        if any(value is None for value in capabilities):
            raise ValueError("runtime LiuState requires explicit computational capability for every node")

        records = tuple(previous_epoch_transactions)
        if not all(isinstance(record, TransactionCreationRecord) for record in records):
            raise ValueError("previous_epoch_transactions must contain TransactionCreationRecord values")
        if is_initial_state:
            if records:
                raise ValueError("initial state chi must not consume transaction records")
            chi_mb = self.configured_workload_mean_transaction_size_mb
            chi_source = "configured_workload_mean"
            sample_count = 0
        else:
            if not records:
                raise ValueError("previous completed epoch has no transaction samples for empirical chi")
            chi_mb = fmean(record.transaction_size for record in records)
            chi_source = "previous_completed_epoch_empirical_mean"
            sample_count = len(records)
        chi_bytes = chi_mb * 1_000_000.0
        state = LiuState(
            chi_bytes,
            tuple(float(value) for value in stakes),
            self.spatial_profiles,
            tuple(float(value) for value in capabilities),
            epoch_context.link_state_matrix,
        )
        state_hash = state.deterministic_hash()
        validator_ids = epoch_context.validator_ids
        validator_stakes = tuple(state.stakes_tokens[node_id] for node_id in validator_ids)
        try:
            stake_gini = canonical_pairwise_gini(validator_stakes)
            stake_reason = None
        except ValueError as error:
            stake_gini = None
            stake_reason = str(error)
        geographic = None if self.geographic_model is None else self.geographic_model.evaluate(validator_ids)
        result = LiuRuntimeStateBuildResult(
            epoch_context.epoch_id,
            state,
            state_hash,
            self.CHI_POLICY_VERSION,
            self.TRANSACTION_SIZE_UNIT_POLICY,
            chi_source,
            sample_count,
            action_hash,
            previous_state_hash,
            stake_gini,
            stake_reason,
            geographic,
        )
        LiuRuntimeInstrumentationCollector.runtime_states.append(
            RuntimeLiuStateRecord(
                epoch_context.epoch_id,
                state_hash,
                chi_bytes,
                self.CHI_POLICY_VERSION,
                self.TRANSACTION_SIZE_UNIT_POLICY,
                chi_source,
                sample_count,
                epoch_context.link_state_matrix.deterministic_hash(),
                action_hash,
                previous_state_hash,
                observation_time,
            )
        )
        return result
