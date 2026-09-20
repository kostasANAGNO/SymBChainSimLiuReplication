"""LIU_DYNAMIC_DES_ENVIRONMENT — Phase 4.

S_t → A_t → real DES → R_t → S_{t+1}   without resetting to S_0.

Network dynamics: link rates evolve each epoch via an FSMC (Finite-State Markov Channel).
The FSMC uses 3 quantized rate levels (OUR_RECONSTRUCTION — paper specifies range only):

  Levels: {LOW=10 Mbps, MID=55 Mbps, HIGH=100 Mbps}
  Transition matrix (shared for all directed links):
      FROM \\ TO  LOW    MID    HIGH
      LOW         0.70   0.20   0.10
      MID         0.15   0.70   0.15
      HIGH        0.10   0.20   0.70

Initialization: each link rate 10+((i+j)%91) is quantized to the nearest level.

Classification:
  FSMC level count, rates, transition matrix: OUR_RECONSTRUCTION
  FSMC structure (Markov channel per link): PAPER_EXACT (Sec III-B)
  Rate range [10, 100] Mbps: PAPER_EXACT (Table I)
  Dynamic state update (R evolves per epoch): PAPER_EXACT (Eq.10 with t-subscript)
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))

from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.DigitalTwinPaired import LiuPaperReferenceRecord
from Liu.LinkFSMC import (
    LinkFSMCState,
    LinkRateLevels,
    LinkTransitionMatrix,
    LinkTransitionTensor,
)
from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters, evaluate_reference
from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    C2DeadlineExceeded,
    DESQueueExhausted,
    LiuSingleActionSnapshot,
    run_single_action,
)

# ── FSMC configuration (OUR_RECONSTRUCTION) ───────────────────────────────────
FSMC_LEVELS = LinkRateLevels(rates_mbps=(10.0, 55.0, 100.0))

_TRANSITION_ROWS = (
    (0.70, 0.20, 0.10),  # from LOW
    (0.15, 0.70, 0.15),  # from MID
    (0.10, 0.20, 0.70),  # from HIGH
)
FSMC_TRANSITION = LinkTransitionMatrix(probabilities=_TRANSITION_ROWS)


def _quantize_rate(r: float) -> float:
    """Map a continuous rate to the nearest FSMC level."""
    levels = FSMC_LEVELS.rates_mbps
    return min(levels, key=lambda lvl: abs(lvl - r))


def _make_fsmc_state(N: int, initial_link_rows: tuple) -> LinkFSMCState:
    """Build initial FSMC state by quantizing the deterministic link rates."""
    quantized = tuple(
        tuple(None if j == i else _quantize_rate(initial_link_rows[i][j]) for j in range(N))
        for i in range(N)
    )
    links = LinkStateMatrix.from_rows(quantized)
    tensor = LinkTransitionTensor.shared(N, FSMC_TRANSITION)
    return LinkFSMCState(FSMC_LEVELS, tensor, links)


class DESFailureReason(str, Enum):
    """Why a decision epoch did not finalize within the Liu constraint.

    FINALIZED_WITHIN_C2_LIMIT — DES reached finality before the Liu C2 deadline
        (T_F = T_I + T_C <= omega * T_I).  Normal training data.

    C2_DEADLINE_EXCEEDED — AUTHORITATIVE Liu C2 failure.  The DES was still running
        when simulated time exceeded boundary_time + omega * T_I.  C2=False, reward=0.
        Valid DRL training data (paper-defined failure mode, PAPER_EXACT Eq.8).

    RUNTIME_GUARD_FAILURE — Implementation guard hit before the C2 deadline:
        event queue exhausted (OUR_RECONSTRUCTION: PBFT view-change cap) or max_events
        exceeded.  NOT a paper-defined failure.  Do NOT use as DRL training data without
        verifying it is equivalent to C2_DEADLINE_EXCEEDED for the specific scenario.
    """
    FINALIZED_WITHIN_C2_LIMIT = "FINALIZED_WITHIN_C2_LIMIT"
    C2_DEADLINE_EXCEEDED = "C2_DEADLINE_EXCEEDED"
    RUNTIME_GUARD_FAILURE = "RUNTIME_GUARD_FAILURE"


@dataclass
class StepResult:
    """Outcome of one decision epoch."""
    step: int
    action: LiuAction
    reward: float
    des_c1: bool
    des_c2: bool
    des_c3: bool
    des_t_c_s: float                    # float('inf') when C2 deadline exceeded or runtime guard
    stake_gini: float
    geo_gini: float
    link_rate_mean_mbps: float
    link_rate_min_mbps: float
    link_rate_max_mbps: float
    paper_c2: bool
    paper_reward: float
    failure_reason: DESFailureReason = DESFailureReason.FINALIZED_WITHIN_C2_LIMIT
    stalled: bool = False               # True when des_c2=False due to any failure reason


class LiuDynamicEpochEnv:
    """Dynamic DES decision-epoch environment with FSMC link-rate evolution.

    Each call to `step(action)` runs one real DES epoch, then advances the
    FSMC link-rate matrix for the next epoch.  The validator set, protocol,
    S_B, and T_I are all part of the action and can change each step.

    Population (fixed throughout an episode):
      chi, Upsilon, x, c — fixed at construction.
    Dynamic component:
      R(t) — evolves via FSMC transition each epoch.
    """

    def __init__(
        self,
        *,
        node_count: int,
        transaction_size_bytes: float,
        stakes_tokens: tuple[float, ...],
        capabilities_ghz: tuple[float, ...],
        positions_km: tuple[tuple[float, float], ...],
        initial_link_rows_mbps: tuple[tuple[float | None, ...], ...],
        faulty_node_ids: tuple[int, ...] = (),
        reference_params: LiuReferenceParameters,
        geo_model: ContinuousSpatialIntensityModel,
        offered_workload_tx: int = 30_000,
        max_events: int = 2_000_000,
        seed: int = 42,
    ) -> None:
        self._N = node_count
        self._chi = transaction_size_bytes
        self._stakes = stakes_tokens
        self._caps = capabilities_ghz
        self._positions = positions_km
        self._initial_link_rows = initial_link_rows_mbps
        self._faulty = faulty_node_ids
        self._ref_params = reference_params
        self._geo = geo_model
        self._offered = offered_workload_tx
        self._max_events = max_events
        self._tx_size_mb = transaction_size_bytes / 1_000_000.0

        self._rng = random.Random(seed)
        self._fsmc: LinkFSMCState = _make_fsmc_state(node_count, initial_link_rows_mbps)
        self._step_count: int = 0

    def reset(self, seed: int | None = None) -> dict:
        """Reset FSMC to initial state; return initial observation."""
        if seed is not None:
            self._rng = random.Random(seed)
        self._fsmc = _make_fsmc_state(self._N, self._initial_link_rows)
        self._step_count = 0
        return self._observe()

    def _link_stats(self) -> tuple[float, float, float]:
        rates = [
            self._fsmc.current_links.rate(i, j)
            for i in range(self._N) for j in range(self._N) if i != j
        ]
        return sum(rates) / len(rates), min(rates), max(rates)

    def _observe(self) -> dict:
        mean_r, min_r, max_r = self._link_stats()
        return {
            "step": self._step_count,
            "link_rate_mean_mbps": mean_r,
            "link_rate_min_mbps": min_r,
            "link_rate_max_mbps": max_r,
            "link_state_hash": self._fsmc.current_links.deterministic_hash(),
        }

    def _current_link_rows(self) -> tuple:
        links = self._fsmc.current_links
        return tuple(
            tuple(None if j == i else links.rate(i, j) for j in range(self._N))
            for i in range(self._N)
        )

    def step(self, action: LiuAction) -> StepResult:
        """Run one DES epoch with current FSMC state, then advance FSMC."""
        link_rows = self._current_link_rows()
        snapshot = LiuSingleActionSnapshot(
            node_count=self._N,
            transaction_size_bytes=self._chi,
            stakes_tokens=self._stakes,
            capabilities_ghz=self._caps,
            positions_km=self._positions,
            link_rows_mbps=link_rows,
            faulty_node_ids=self._faulty,
            epoch0_validator_ids=action.validator_ids,
        )

        mean_r, min_r, max_r = self._link_stats()

        try:
            result = run_single_action(
                snapshot,
                action,
                self._geo,
                reference_params=self._ref_params,
                stake_gini_threshold=self._ref_params.stake_gini_threshold_eta_s,
                geographic_gini_threshold=self._ref_params.geographic_gini_threshold_eta_l,
                finality_multiplier_omega=self._ref_params.finality_multiplier_omega,
                signature_cycles_alpha=self._ref_params.signature_verification_cycles_alpha,
                mac_cycles_beta=self._ref_params.mac_operation_cycles_beta,
                offered_workload_tx=self._offered,
                workload_tx_size_mb=self._tx_size_mb,
                max_events=self._max_events,
            )

            d = result.des_observed
            p = result.paper_reference
            self._fsmc = self._fsmc.advance(self._rng)
            self._step_count += 1

            return StepResult(
                step=self._step_count,
                action=action,
                reward=d.reward_des_liu_objective,
                des_c1=d.c1_decentralization_passed,
                des_c2=d.c2_finality_passed,
                des_c3=d.c3_security_passed,
                des_t_c_s=d.t_c_des_s,
                stake_gini=p.stake_gini,
                geo_gini=p.geographic_gini,
                link_rate_mean_mbps=mean_r,
                link_rate_min_mbps=min_r,
                link_rate_max_mbps=max_r,
                paper_c2=p.c2_finality_passed,
                paper_reward=p.reward_paper_reference,
                failure_reason=DESFailureReason.FINALIZED_WITHIN_C2_LIMIT,
                stalled=False,
            )

        except (C2DeadlineExceeded, DESQueueExhausted, RuntimeError) as exc:
            # Determine failure reason:
            #   C2DeadlineExceeded  → authoritative Liu C2 failure (PAPER_EXACT condition)
            #   DESQueueExhausted   → IMPLEMENTATION_GUARD (view-change cap, OUR_RECONSTRUCTION)
            #   RuntimeError        → IMPLEMENTATION_GUARD (max_events cap, OUR_RECONSTRUCTION)
            # All three map to des_c2=False, reward=0 and advance FSMC normally.
            # The caller must check failure_reason before using as DRL training data.
            if isinstance(exc, C2DeadlineExceeded):
                reason = DESFailureReason.C2_DEADLINE_EXCEEDED
            else:
                reason = DESFailureReason.RUNTIME_GUARD_FAILURE

            links = LinkStateMatrix.from_rows(link_rows)
            mask = tuple(1 if i in set(action.validator_ids) else 0 for i in range(self._N))
            malicious = sum(1 for f in self._faulty if f in set(action.validator_ids))
            oracle = evaluate_reference(
                node_stakes=self._stakes,
                node_capabilities_ghz=self._caps,
                link_rates=links,
                geographic_gini=self._geo.geographic_gini(),
                producer_mask=mask,
                protocol=action.consensus_protocol,
                block_size_mb=action.block_size_mb,
                block_interval_s=action.block_interval_s,
                malicious_count=malicious,
                params=self._ref_params,
            )
            paper = LiuPaperReferenceRecord.from_reference(oracle)
            self._fsmc = self._fsmc.advance(self._rng)
            self._step_count += 1

            return StepResult(
                step=self._step_count,
                action=action,
                reward=0.0,
                des_c1=paper.c1_decentralization_passed,
                des_c2=False,
                des_c3=paper.c3_security_passed,
                des_t_c_s=float("inf"),
                stake_gini=paper.stake_gini,
                geo_gini=paper.geographic_gini,
                link_rate_mean_mbps=mean_r,
                link_rate_min_mbps=min_r,
                link_rate_max_mbps=max_r,
                paper_c2=paper.c2_finality_passed,
                paper_reward=paper.reward_paper_reference,
                failure_reason=reason,
                stalled=True,
            )
