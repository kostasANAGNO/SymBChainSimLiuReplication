# Full 2400-action controlled sweep — summary

Attempted: **2400 / 2400**  
SUCCESS: **2400** · ERROR: 0 · TIMEOUT: 0 · INVALID: 0

Total wall: **73900.8 s (1231.68 min)**; mean 30.792 s; median 4.120 s; max 63043.948 s

## C2 paper-vs-DES 2×2

| | DES PASS | DES FAIL |
|---|---|---|
| **Paper PASS** | 2013 | 0 |
| **Paper FAIL** | 79 | 308 |

paper_c2 pass: 2013, fail: 387  
des_c2 pass: 2092, fail: 308  
paper_fail_des_pass count: 79  
paper_pass_des_fail count: 0

## Best-reward tied action sets

**Best paper reward:** 17000.0  
Tied set size: 2  
Actions: ['LIU_QUORUM|S_B=3.4|T_I=1.0', 'LIU_QUORUM|S_B=6.8|T_I=2.0']


**Best DES-Liu-objective reward:** 17000.0  
Tied set size: 2  
Actions: ['LIU_QUORUM|S_B=3.4|T_I=1.0', 'LIU_QUORUM|S_B=6.8|T_I=2.0']


**Intersection:** 2 action(s): ['LIU_QUORUM|S_B=3.4|T_I=1.0', 'LIU_QUORUM|S_B=6.8|T_I=2.0']

## Per protocol

| protocol | attempted | des_feasible | paper_feasible | best_des_action | best_des_liu_reward | best_paper_reward |
|---|---|---|---|---|---|---|
| PBFT | 800 | 680 | 603 | PBFT|S_B=7.0|T_I=3.0 | 11666.666666666666 | 7555.555555555556 |
| ZYZZYVA | 800 | 688 | 686 | ZYZZYVA|S_B=6.2|T_I=2.5 | 12400.0 | 12000.0 |
| LIU_QUORUM | 800 | 724 | 724 | LIU_QUORUM|S_B=3.4|T_I=1.0 | 17000.0 | 17000.0 |

## Largest paper↔DES discrepancies

- **abs_delta_t_c**: {'action': 'PBFT|S_B=8.0|T_I=0.5', 'value': -10.131147320418194}
- **abs_delta_t_f**: {'action': 'PBFT|S_B=8.0|T_I=0.5', 'value': -10.131147320418194}

## Invariants

- seeds observed: ['0']
- C1 violations: 0
- C3 violations: 0

## Sanity

- non-monotonic T_C_DES dips: 0
- NaN/negative metrics: 0
