"""Liu paper constants and shared reference helpers (no consensus model, no DES).

Holds what the DES/DRL path legitimately needs from the paper: Table-I reference parameters,
the stake Gini (Eq. 2), the continuous geographic Gini (Eq. 3) and the nominal Omega (Eq. 1).
The Appendix-B analytical consensus model and the paper-vs-DES comparison live in
``PaperReference`` and are never imported from here.

Paper: Liu et al., IEEE TII 2019, DOI 10.1109/TII.2019.2897805. Equation citations inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Callable, Sequence

from Utils.DecentralizationMetrics import canonical_pairwise_gini

# ----------------------------------------------------------------------------- #
# C1 decentralization
# ----------------------------------------------------------------------------- #
def paper_stake_gini(producer_stakes: Sequence[float]) -> float:
    """Eq. (2): G(Upsilon) = sum_i sum_j |Y_i - Y_j| / (2 K sum_i Y_i) over the K producers."""
    stakes = tuple(float(value) for value in producer_stakes)
    if len(stakes) < 1:
        raise ValueError("stake Gini requires the selected block producers")
    return canonical_pairwise_gini(stakes)


@dataclass(frozen=True, slots=True)
class GeographicGiniConfig:
    """Numerical-integration control for Eq. (3). Region Xi = [x0,x1] x [y0,y1]."""

    x0: float = 0.0
    x1: float = 1.0
    y0: float = 0.0
    y1: float = 1.0
    resolution: int = 64          # cells per axis (grid is resolution x resolution)
    convergence_tol: float = 1e-3  # |G(2R) - G(R)| must be <= tol when convergence is checked

    def __post_init__(self) -> None:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("geographic region must have positive extent")
        if self.resolution < 2:
            raise ValueError("geographic Gini resolution must be >= 2")
        if self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be positive")


def _cell_centers(config: GeographicGiniConfig, resolution: int):
    dx = (config.x1 - config.x0) / resolution
    dy = (config.y1 - config.y0) / resolution
    cells = []
    for i in range(resolution):
        cx = config.x0 + (i + 0.5) * dx
        for j in range(resolution):
            cy = config.y0 + (j + 0.5) * dy
            cells.append((cx, cy))
    return cells, dx * dy


def _geographic_gini_at(lambda_fn: Callable[[float, float], float], config: GeographicGiniConfig, resolution: int) -> float:
    """Discrete evaluation of Eq. (3): int int |lambda(x)-lambda(y)| dy dx / (2 int int lambda(x) dy dx)."""
    cells, cell_area = _cell_centers(config, resolution)
    densities = [float(lambda_fn(cx, cy)) for cx, cy in cells]
    if any(value < 0 for value in densities):
        raise ValueError("lambda(x) density must be non-negative")
    region_area = (config.x1 - config.x0) * (config.y1 - config.y0)
    integral_lambda = sum(densities) * cell_area  # int_Xi lambda(x) dx
    denominator = 2.0 * integral_lambda * region_area  # 2 int_Xi int_Xi lambda(x) dy dx
    if denominator <= 0:
        raise ValueError("geographic Gini is undefined for an all-zero density")
    # Equal-area cells => the double integral int int |lambda(x)-lambda(y)| reduces to the
    # (equal-weight) pairwise sum sum_i sum_j |a_i-a_j|, computed exactly in O(m log m):
    # for ascending-sorted a, sum_i sum_j |a_i-a_j| = 2 * sum_i (2i - m - 1) * a_i (1-indexed).
    ordered = sorted(densities)
    m = len(ordered)
    pairwise = 2.0 * sum((2 * (i + 1) - m - 1) * value for i, value in enumerate(ordered))
    numerator = pairwise * cell_area * cell_area
    return numerator / denominator


def geographic_gini_eq3(
    lambda_fn: Callable[[float, float], float],
    config: GeographicGiniConfig | None = None,
    *,
    check_convergence: bool = False,
) -> float:
    """Eq. (3) geographic Gini G(lambda) over the continuous intensity lambda(x).

    lambda_fn(x, y) is the (PAPER_UNDERSPECIFIED) producer-intensity density; Eq. (3) itself
    is PAPER_EXACT. This evaluates the density directly and is NOT a pairwise-distance surrogate.
    With check_convergence, the grid is doubled and the change must be <= config.convergence_tol.
    """
    config = config or GeographicGiniConfig()
    value = _geographic_gini_at(lambda_fn, config, config.resolution)
    if check_convergence:
        finer = _geographic_gini_at(lambda_fn, config, config.resolution * 2)
        if abs(finer - value) > config.convergence_tol:
            raise ValueError(
                f"geographic Gini not converged at resolution {config.resolution}: "
                f"|{finer} - {value}| = {abs(finer - value)} > {config.convergence_tol}"
            )
        return finer
    return value


def integral_of_lambda(lambda_fn: Callable[[float, float], float], config: GeographicGiniConfig | None = None) -> float:
    """int_Xi lambda(x) dx (should equal K for a normalized producer intensity)."""
    config = config or GeographicGiniConfig()
    cells, cell_area = _cell_centers(config, config.resolution)
    return sum(float(lambda_fn(cx, cy)) for cx, cy in cells) * cell_area


def assert_lambda_normalized(
    lambda_fn: Callable[[float, float], float],
    producer_count_k: int,
    config: GeographicGiniConfig | None = None,
    *,
    tol: float = 1e-3,
) -> float:
    """Enforce the paper normalization int_Xi lambda(x) dx = K (Section III-B); raise otherwise."""
    config = config or GeographicGiniConfig()
    integral = integral_of_lambda(lambda_fn, config)
    if abs(integral - producer_count_k) > tol:
        raise ValueError(
            f"lambda(x) is not normalized to K={producer_count_k}: int lambda = {integral} "
            f"(|diff| {abs(integral - producer_count_k)} > tol {tol})"
        )
    return integral


# ----------------------------------------------------------------------------- #
# Reference parameters
# ----------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LiuReferenceParameters:
    signature_verification_cycles_alpha: float   # Table I alpha
    mac_operation_cycles_beta: float             # Table I beta
    network_timeout_s: float                     # consensus timeout T (min{.,T})
    finality_multiplier_omega: float             # Table I omega (Eq. 8)
    stake_gini_threshold_eta_s: float            # Table I eta_s (Eq. 4)
    geographic_gini_threshold_eta_l: float       # Table I eta_l (Eq. 5)
    transaction_size_bytes: int                  # chi (Table I)
    # The next two fields are consumed only by the PaperReference analytical model; the DES ignores them.
    pbft_zyzzyva_batch_m: int = 3                # Table I M=3 for PBFT/Zyzzyva
    recovery_delay_s: float = 0.0                # Zyzzyva t_r (recovery path)
    megabytes_to_bytes: float = 1_000_000.0      # S_B (MB) -> bytes; 1e6 (unspecified 1e6 vs 2^20)

    def __post_init__(self) -> None:
        if self.pbft_zyzzyva_batch_m < 1:
            raise ValueError("batch M must be >= 1")
        if self.finality_multiplier_omega <= 1:
            raise ValueError("omega must be > 1 (Eq. 8)")


def nominal_throughput_tps(block_size_mb: float, block_interval_s: float, params: LiuReferenceParameters) -> float:
    """Liu's Omega (Eq. 1): floor(S_B / chi) / T_I, with S_B converted to bytes by `megabytes_to_bytes`."""
    capacity = floor(block_size_mb * params.megabytes_to_bytes / params.transaction_size_bytes)
    return capacity / block_interval_s
