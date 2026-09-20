"""Immutable, provenance-carrying manifests for Liu experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_integer


MANIFEST_SCHEMA_VERSION = "liu_experiment_manifest_v1"


class LiuParameterClassification(str, Enum):
    PAPER_EXACT = "PAPER_EXACT"
    PAPER_DERIVED = "PAPER_DERIVED"
    RECONSTRUCTION_REQUIRED = "RECONSTRUCTION_REQUIRED"
    EXPERIMENT_CONTROL = "EXPERIMENT_CONTROL"


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"manifest values must be JSON-compatible, got {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LiuParameterProvenance(CanonicalSerializable):
    parameter: str
    paper_value: str | None
    paper_location: str
    classification: LiuParameterClassification
    runtime_representation: str
    chosen_reconstruction: str | None
    justification: str

    def __post_init__(self) -> None:
        for name in ("parameter", "paper_location", "runtime_representation", "justification"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.classification, LiuParameterClassification):
            raise ValueError("classification must be a LiuParameterClassification")
        if self.classification in {
            LiuParameterClassification.RECONSTRUCTION_REQUIRED,
            LiuParameterClassification.EXPERIMENT_CONTROL,
        } and not self.chosen_reconstruction:
            raise ValueError("reconstructed/control parameters require an explicit chosen value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_reconstruction": self.chosen_reconstruction,
            "classification": self.classification.value,
            "justification": self.justification,
            "paper_location": self.paper_location,
            "paper_value": self.paper_value,
            "parameter": self.parameter,
            "runtime_representation": self.runtime_representation,
        }


@dataclass(frozen=True, slots=True)
class LiuSeedBundle(CanonicalSerializable):
    simulator_workload: int
    fsmc: int
    candidate_generation: int
    exploration: int
    replay_sampling: int
    network_initialization: int
    training_torch: int

    def __post_init__(self) -> None:
        names = (
            "simulator_workload", "fsmc", "candidate_generation", "exploration",
            "replay_sampling", "network_initialization", "training_torch",
        )
        values = tuple(require_integer(getattr(self, name), name) for name in names)
        if len(set(values)) != len(values):
            raise ValueError("seed bundle streams must use distinct seeds")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class LiuReconstructionChoice(CanonicalSerializable):
    policy: str
    version: str
    selected_value: str
    rationale: str
    sensitivity_required: bool

    def __post_init__(self) -> None:
        for name in ("policy", "version", "selected_value", "rationale"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.sensitivity_required, bool):
            raise ValueError("sensitivity_required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "rationale": self.rationale,
            "selected_value": self.selected_value,
            "sensitivity_required": self.sensitivity_required,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LiuExperimentManifest(CanonicalSerializable):
    experiment_name: str
    seed_bundle: LiuSeedBundle
    paper_parameter_provenance: tuple[LiuParameterProvenance, ...]
    simulator_configuration: Any
    state_action_domains: Any
    constraint_thresholds: Any
    dqn_configuration: Any
    candidate_configuration: Any
    episode_configuration: Any
    reconstruction_registry: tuple[LiuReconstructionChoice, ...]
    schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be non-empty")
        if not isinstance(self.seed_bundle, LiuSeedBundle):
            raise ValueError("seed_bundle must be LiuSeedBundle")
        provenance = tuple(self.paper_parameter_provenance)
        registry = tuple(self.reconstruction_registry)
        if not provenance or len({item.parameter for item in provenance}) != len(provenance):
            raise ValueError("parameter provenance must be non-empty and unique")
        if len({item.policy for item in registry}) != len(registry):
            raise ValueError("reconstruction policies must be unique")
        object.__setattr__(self, "paper_parameter_provenance", tuple(sorted(provenance, key=lambda item: item.parameter)))
        object.__setattr__(self, "reconstruction_registry", tuple(sorted(registry, key=lambda item: item.policy)))
        for name in (
            "simulator_configuration", "state_action_domains", "constraint_thresholds",
            "dqn_configuration", "candidate_configuration", "episode_configuration",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def manifest_sha256(self) -> str:
        return self.deterministic_hash()

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_configuration": _thaw(self.candidate_configuration),
            "constraint_thresholds": _thaw(self.constraint_thresholds),
            "dqn_configuration": _thaw(self.dqn_configuration),
            "episode_configuration": _thaw(self.episode_configuration),
            "experiment_name": self.experiment_name,
            "paper_parameter_provenance": [item.to_dict() for item in self.paper_parameter_provenance],
            "reconstruction_registry": [item.to_dict() for item in self.reconstruction_registry],
            "schema_version": self.schema_version,
            "seed_bundle": self.seed_bundle.to_dict(),
            "simulator_configuration": _thaw(self.simulator_configuration),
            "state_action_domains": _thaw(self.state_action_domains),
        }

    def artifact_dict(self) -> dict[str, Any]:
        result = self.to_dict()
        result["manifest_sha256"] = self.manifest_sha256
        return result
