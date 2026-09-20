# DES Reward Semantics — Preserving Liu's Objective

## Liu's optimization problem

Liu solves:

```
maximize   Ω(S_B, T_I) = floor(S_B / χ) / T_I         (Eq.1, Eq.13 numerator)
subject to C1 decentralization    (Eq.4/5)
           C2 finality: T_F = T_I + T_C ≤ ω · T_I    (Eq.6, Eq.8)
           C3 security:  f ≤ F^δ                      (Eq.9)
```

The objective is **nominal analytical throughput Ω**; consensus latency `T_C` affects the
optimization *only* through the finality constraint C2, never through the reward numerator.

## The (fixed) issue

The previous DES-side reward was:

```
throughput_des_single_epoch_finalized = finalized_tx / (T_I + T_C_DES)
reward_des_realized                    = throughput_des_single_epoch_finalized
                                          if (C1_DES ∧ C2_DES ∧ C3_DES) else 0
```

Under this definition `T_C_DES` entered the objective **twice**: once as the denominator of
the throughput numerator, and again as the gate through C2. That is not Liu's objective and
would bias any optimization built on top of it.

## New DES-side Liu-compatible reward

```
throughput_paper_nominal   = floor(S_B / χ) / T_I            (Liu Ω; identical on both sides)
reward_des_liu_objective   = throughput_paper_nominal
                              if (C1_DES ∧ C2_DES ∧ C3_DES) else 0
```

`T_C_DES` now enters **only** through `C2_DES` — the same role Liu gives it. The scientific
comparison isolates a single degree of freedom: **paper vs DES *feasibility evaluation***
for the same objective Ω.

## Three named quantities (never conflate them)

| Name | Formula | Role |
|------|---------|------|
| `throughput_paper_nominal` | `floor(S_B / χ) / T_I` | Liu's objective Ω; identical on both sides |
| `reward_paper_reference` | Ω if paper C1/C2/C3 pass else 0 | Liu solved by the paper equations (analytical) |
| `reward_des_liu_objective` | Ω if DES C1/C2/C3 pass else 0 | Liu solved with DES-evaluated constraints (Liu-compatible) |
| `throughput_des_single_epoch_finalized` | `finalized_tx / (T_I + T_C_DES)` | **DES_OPERATIONAL_METRIC** — diagnostic |
| `reward_des_realized` | previous single-epoch reward | historical diagnostic; preserved unchanged for backward compatibility |

## Backward compatibility

- **Historical golden JSONs** (PBFT / Zyzzyva fast + recovery / Quorum) were not modified.
- **`reward_des_realized`** and its policy label `LIU_EQ13_DES_REALIZATION_finalized_tps_v1`
  keep their historical meaning (single-epoch finalized-TPS gated by DES C1/C2/C3). Loaders
  built before this milestone see identical values for the fields they consume.
- New additive fields on `LiuDesObservedRecord`:
  - `reward_des_liu_objective: float = 0.0`
  - `throughput_paper_nominal: float = 0.0`
  - `reward_des_liu_objective_policy_version: str = "liu_omega_gated_by_des_constraints_v1"`
- New additive field on `LiuDigitalTwinDelta`:
  - `reward_liu_objective_delta: float = 0.0` (equals `reward_des_liu_objective − reward_paper_reference`;
    zero when paper and DES agree on feasibility, exactly ±Ω when they disagree).
- Controlled-sweep CSV gains four columns: `reward_des_liu_objective`,
  `throughput_paper_nominal`, `reward_liu_objective_delta`, plus the existing diagnostics.

## Key regression: PBFT at S_B = 2 MB, T_I = 1 s (χ = 200 B)

`Ω = floor(2 000 000 / 200) / 1 = 10 000 TPS`. From the golden PBFT-2/1 case:

| Field | Value | Note |
|-------|-------|------|
| `paper_c2` | **False** | Paper Eq.15 T_F = 6.19 s > ω·T_I = 6 s |
| `des_c2` | **True** | DES T_F = 1.31 s ≤ 6 s (concurrent delivery + advance-at-quorum) |
| `throughput_paper_nominal` | 10 000 | Ω |
| `reward_paper_reference` | **0** | paper C2 gates it to zero |
| `reward_des_liu_objective` | **10 000** | Ω gated by DES C1/C2/C3 (all pass) |
| `throughput_des_single_epoch_finalized` | ~1 930 | diagnostic; separately reported |
| `reward_des_realized` | ~1 930 | historical field; preserved unchanged |
| `reward_liu_objective_delta` | **+10 000** | paper says infeasible, DES says feasible |

This action is the acceptance regression: enforced by
`test_liu_reward_semantics.py::SingleActionIntegration::test_pbft_2mb_1s_regression_paper_c2_fail_des_c2_pass`.

## What this enables

The DES twin now answers Liu's own optimization question: *under the same Ω objective,
does event-level operational feasibility rank the (S_B, T_I, δ) action space differently
from analytical feasibility?* The controlled 18-action smoke already shows this:

- **Best paper-objective set (Ω = 10 000, 2 tied):** `{ZYZZYVA 2/1, LIU_QUORUM 2/1}`.
- **Best DES-Liu-objective set (Ω = 10 000, 3 tied):** `{PBFT 2/1, ZYZZYVA 2/1, LIU_QUORUM 2/1}`.
- **Intersection:** `{ZYZZYVA 2/1, LIU_QUORUM 2/1}` — PBFT 2/1 is admitted **only** by the DES.

The single-epoch diagnostic still ranks Quorum 2/1 alone as the highest **operational** TPS
observed. That ranking is meaningful in its own right and is now clearly separated from Liu's
objective.

## Future work — NOT this milestone

A genuine multi-block steady-state TPS
`total finalized tx / total simulated observation time` may later be defined as an alternative
DRL objective (with its own constraints), but that is a separate design and is not the
Liu-compatible reward.
