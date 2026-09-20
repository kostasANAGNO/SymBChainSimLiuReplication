# N30 Seed-42 Clean Policy Evaluation

**Gate: N30_SEED42_POLICY_VERIFIED — NOT AWARDED**

## 1. Evaluation Design

- Eval seeds: [1000, 1001, 1002, 1003] (outside training range [42, 441])
- Initial states: ALL_HIGH_100, ALL_MID_55, ALL_LOW_10, TRAINING_MIXED
- Steps per trajectory: 20
- Trajectories: 16
- Total steps per policy: 320
- FSMC trajectory: **identical** for all policies per trajectory (same `random.Random(eval_seed)` initial state)

## 2. C1 Validity

- `geo_gini = 0.1000` (constant for all K=28 validator sets, ≤ 0.30 threshold)
- C1-valid validator sets: **384 / 435**  (stake gini ≤ 0.20)
- C1-invalid validator sets: 51

## 3. Strong Fixed Baseline

- Protocol: LIU_QUORUM  |  S_B = 6.8 MB  |  T_I = 2.0 s
- action_id = 7  |  stake_gini = 0.1925  |  C1-valid = True

## 4. Global Policy Metrics
*(all policies, 320 steps each)*

| Metric | DQN ep400 | Random | Strong Fixed | DQN ep350 |
|--------|-----------|--------|--------------|-----------|
| Cumulative reward | 2716500.0 | 1594633.3333 | 476000.0 | 2627966.6667 |
| Mean reward/step | 8489.0625 | 4983.2292 | 1487.5 | 8212.3958 |
| Median reward/step | 11666.666666666666 | 0.0 | 0.0 | 11666.666666666666 |
| Feasible steps | 225 | 147 | 28 | 218 |
| Feasibility rate | 0.7031 | 0.4594 | 0.0875 | 0.6813 |
| C1 pass | 319 | 281 | 320 | 320 |
| C2 pass | 225 | 147 | 28 | 218 |
| C3 pass | 320 | 320 | 320 | 320 |
| C2_DEADLINE_EXCEEDED | 86 | 157 | 281 | 87 |
| RUNTIME_GUARD_FAILURE | 9 | 16 | 11 | 15 |
| Mean Ω feasible | 12126.6667 | 12061.2245 | 17000.0 | 12054.893 |
| Valid validator sel. | 319 | 281 | 320 | 320 |
| Invalid validator | 1 | 39 | 0 | 0 |
| Valid rate | 0.9969 | 0.8781 | 1.0 | 1.0 |

## 5. DQN ep400 Action Profile

- Protocol distribution: {'LIU_QUORUM': 17, 'PBFT': 118, 'ZYZZYVA': 185}
- Block-size distribution: {'6.8': 8, '3.4': 9, '4.6': 28, '7.0': 85, '7.4': 87, '6.2': 18, '5.8': 5, '1.2': 80}
- Block-interval distribution: {'2.0': 36, '1.0': 9, '3.0': 172, '2.5': 23, '0.5': 80}
- Distinct actions selected: 124 / 3915
- Top-10 action ids: [(553, 18), (1638, 15), (1184, 11), (1179, 7), (455, 7), (1183, 6), (1148, 6), (1593, 6), (1220, 5), (1854, 5)]

### 5a. Adaptation by Network Condition (DQN ep400)

| Bucket | n | feasible | mean_reward | protocol_dist |
|--------|---|----------|-------------|---------------|
| NEAR_HIGH_100MBPS | 4 | 4 | 17000.0 | {'LIU_QUORUM': 4} |
| CONTAINS_LOW_10MBPS | 312 | 217 | 8330.66 | {'LIU_QUORUM': 13, 'PBFT': 118, 'ZYZZYVA': 181} |
| MEDIUM_RANGE | 4 | 4 | 12333.33 | {'ZYZZYVA': 4} |

## 6. Paired Comparisons

DQN ep400 vs Random (16 trajectories):
- Mean paired diff (ep400 − random): **70117**
- ep400 wins: 16  |  tied: 0  |  random wins: 0

DQN ep400 vs Strong Fixed (16 trajectories):
- Mean paired diff (ep400 − fixed): **140031**
- ep400 wins: 16  |  tied: 0  |  fixed wins: 0

DQN ep400 vs DQN ep350 diagnostic (16 trajectories):
- Mean paired diff (ep400 − ep350): **5533**
- ep400 wins: 10  |  tied: 0  |  ep350 wins: 6
- Verdict: **NOT_DETECTED**

## 7. Q-Value Health (ep400)

- Samples: 8
- Any NaN: False
- Any Inf: False
- Any degenerate (std<1e-4): False
- Global Q range: [-6.9377, 5.9996]
- Mean Q std across samples: 0.1986

## 8. Concrete Trajectory Example (DQN ep400)

Eval seed 1000, initial state ALL_HIGH_100

| Step | Network | Bucket | Protocol | S_B | T_I | C1-valid | C1 | C2 | C3 | Reward | Failure |
|------|---------|--------|----------|-----|-----|----------|----|----|----|--------|---------|
| 0 | mean=100 min=100 Mbps | NEAR_HIGH_10 | LIU_QUORUM | 6.8 | 2.0 | True | 1 | 1 | 1 | 17000 | FINALIZED_WI |
| 1 | mean=82 min=10 Mbps | CONTAINS_LOW | LIU_QUORUM | 3.4 | 1.0 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 2 | mean=73 min=10 Mbps | CONTAINS_LOW | LIU_QUORUM | 3.4 | 1.0 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 3 | mean=66 min=10 Mbps | CONTAINS_LOW | LIU_QUORUM | 3.4 | 1.0 | True | 1 | 1 | 1 | 17000 | FINALIZED_WI |
| 4 | mean=60 min=10 Mbps | CONTAINS_LOW | PBFT | 4.6 | 2.0 | True | 1 | 1 | 1 | 11500 | FINALIZED_WI |
| 5 | mean=58 min=10 Mbps | CONTAINS_LOW | PBFT | 7.0 | 3.0 | True | 1 | 1 | 1 | 11667 | FINALIZED_WI |
| 6 | mean=57 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 7.4 | 3.0 | True | 1 | 1 | 1 | 12333 | FINALIZED_WI |
| 7 | mean=56 min=10 Mbps | CONTAINS_LOW | LIU_QUORUM | 6.8 | 2.0 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 8 | mean=57 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 6.2 | 2.5 | True | 1 | 1 | 1 | 12400 | FINALIZED_WI |
| 9 | mean=56 min=10 Mbps | CONTAINS_LOW | PBFT | 5.8 | 2.5 | True | 1 | 1 | 1 | 11600 | FINALIZED_WI |
| 10 | mean=55 min=10 Mbps | CONTAINS_LOW | PBFT | 7.0 | 3.0 | True | 1 | 0 | 1 | 0 | RUNTIME_GUAR |
| 11 | mean=53 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 7.4 | 3.0 | True | 1 | 1 | 1 | 12333 | FINALIZED_WI |
| 12 | mean=55 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 7.4 | 3.0 | True | 1 | 1 | 1 | 12333 | FINALIZED_WI |
| 13 | mean=56 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 1.2 | 0.5 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 14 | mean=57 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 6.2 | 2.5 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 15 | mean=56 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 6.2 | 2.5 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 16 | mean=56 min=10 Mbps | CONTAINS_LOW | ZYZZYVA | 1.2 | 0.5 | True | 1 | 0 | 1 | 0 | C2_DEADLINE_ |
| 17 | mean=58 min=10 Mbps | CONTAINS_LOW | PBFT | 7.0 | 3.0 | True | 1 | 1 | 1 | 11667 | FINALIZED_WI |
| 18 | mean=59 min=10 Mbps | CONTAINS_LOW | PBFT | 7.0 | 3.0 | True | 1 | 1 | 1 | 11667 | FINALIZED_WI |
| 19 | mean=58 min=10 Mbps | CONTAINS_LOW | PBFT | 7.0 | 3.0 | True | 1 | 1 | 1 | 11667 | FINALIZED_WI |

## 9. Gate Criteria

- **A_stable_finite_q**: PASS
- **B_near_zero_c1_invalid_selection**: PASS
- **C_meaningful_feasible_behavior**: PASS
- **D_better_than_random_paired**: PASS
- **E_no_runtime_guard_pathology**: FAIL
- **F_policy_adaptation_evidence**: PASS

## Result: N30_SEED42_POLICY_VERIFIED — NOT AWARDED