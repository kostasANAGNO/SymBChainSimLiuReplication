"""Continuous Eq.(3) geographic intensity for the DES digital-twin C1.

Liu defines geographic decentralization through the continuous intensity lambda(x)
(Eq. 3), NOT through a grid histogram of the selected validators. This module wraps the
verified `ReferenceCore.geographic_gini_eq3` evaluator behind an explicit, normalized
lambda(x) scenario model so the SAME lambda drives both the DES-operational C1 and the
Liu analytical reference oracle.

The functional form of lambda(x) is PAPER_UNDERSPECIFIED (OUR_RECONSTRUCTION). This module
does NOT solve the general "x -> lambda" estimation problem: lambda(x) is supplied as
explicit spatial-scenario metadata, and its integral is enforced to equal K (Section III-B).
The old grid-histogram estimator remains available only as a labelled legacy diagnostic
(`geographic_gini_legacy_grid`) and must never gate reward_des_realized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from Liu.ReferenceCore import (
    GeographicGiniConfig,
    assert_lambda_normalized,
    geographic_gini_eq3,
    integral_of_lambda,
)
from Liu.Serialization import CanonicalSerializable
from Liu.Validation import require_finite_number, require_integer

CONTINUOUS_ESTIMATOR_VERSION = "continuous_eq3_lambda_v1"


@dataclass(frozen=True, slots=True)
class ContinuousGeographicResult(CanonicalSerializable):
    """Eq.(3) G(lambda) computed from an explicit normalized intensity field."""

    estimator_version: str
    geographic_gini: float
    integrated_intensity: float
    producer_count_k: int
    resolution: int
    lambda_form: str

    def to_dict(self) -> dict:
        return {
            "estimator_version": self.estimator_version,
            "geographic_gini": self.geographic_gini,
            "integrated_intensity": self.integrated_intensity,
            "lambda_form": self.lambda_form,
            "producer_count_k": self.producer_count_k,
            "resolution": self.resolution,
        }


class ContinuousSpatialIntensityModel:
    """Explicit lambda(x) scenario model evaluating Eq.(3) G(lambda).

    G(lambda) is a scalar property of the producer intensity field; for this controlled
    milestone it does not depend on which K subset is selected (the per-selection x->lambda
    estimation is deferred and PAPER_UNDERSPECIFIED). The integral is enforced to equal K.
    """

    def __init__(
        self,
        lambda_fn: Callable[[float, float], float],
        producer_count_k: int,
        *,
        lambda_form: str,
        config: GeographicGiniConfig | None = None,
        normalization_tol: float = 1e-3,
    ) -> None:
        if not callable(lambda_fn):
            raise ValueError("lambda_fn must be callable lambda(x, y) -> density")
        self._lambda_fn = lambda_fn
        self.producer_count_k = require_integer(producer_count_k, "producer_count_k", minimum=1)
        if not isinstance(lambda_form, str) or not lambda_form:
            raise ValueError("lambda_form must be a non-empty descriptive string")
        self.lambda_form = lambda_form
        self.config = config or GeographicGiniConfig()
        self.normalization_tol = require_finite_number(
            normalization_tol, "normalization_tol", positive=True
        )
        # Enforce the paper normalization int_Xi lambda(x) dx = K (Section III-B).
        self._integrated = assert_lambda_normalized(
            self._lambda_fn, self.producer_count_k, self.config, tol=self.normalization_tol
        )

    def geographic_gini(self) -> float:
        return geographic_gini_eq3(self._lambda_fn, self.config)

    def integrated_intensity(self) -> float:
        return integral_of_lambda(self._lambda_fn, self.config)

    def evaluate(self) -> ContinuousGeographicResult:
        return ContinuousGeographicResult(
            estimator_version=CONTINUOUS_ESTIMATOR_VERSION,
            geographic_gini=self.geographic_gini(),
            integrated_intensity=self._integrated,
            producer_count_k=self.producer_count_k,
            resolution=self.config.resolution,
            lambda_form=self.lambda_form,
        )


def constant_intensity(producer_count_k: int, config: GeographicGiniConfig | None = None):
    """Homogeneous lambda(x) = K / area (degenerate: G(lambda) == 0). Neutral sanity field."""
    config = config or GeographicGiniConfig()
    area = (config.x1 - config.x0) * (config.y1 - config.y0)
    density = producer_count_k / area

    def lambda_fn(x: float, y: float) -> float:
        return density

    return lambda_fn


def planar_gradient_intensity(
    producer_count_k: int,
    slope: float,
    config: GeographicGiniConfig | None = None,
):
    """Non-constant lambda(x,y) = (K/area) * (1 + slope*(u - 0.5)), u the normalized x-coord.

    Symmetric about the region centre so int_Xi lambda = K for all slope in [0, 2). This is
    the simplest reproducible NON-constant intensity for a controlled test (OUR_RECONSTRUCTION;
    Liu does not publish lambda(x)'s functional form).
    """
    config = config or GeographicGiniConfig()
    if not 0.0 <= slope < 2.0:
        raise ValueError("slope must be in [0, 2) to keep lambda non-negative and normalized")
    width = config.x1 - config.x0
    area = width * (config.y1 - config.y0)
    base = producer_count_k / area

    def lambda_fn(x: float, y: float) -> float:
        u = (x - config.x0) / width
        return base * (1.0 + slope * (u - 0.5))

    return lambda_fn
