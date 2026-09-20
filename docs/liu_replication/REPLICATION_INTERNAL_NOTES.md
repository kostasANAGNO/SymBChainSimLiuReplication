# Liu 2019 Replication — Internal Knowledge File

**Purpose**: Preserve facts needed for final documentation. Not the final thesis doc.  
**Paper**: "Performance Optimization for Blockchain-Enabled IIoT Systems: A DRL Approach"  
Liu et al., IEEE TII Vol.15 No.6, June 2019. DOI: 10.1109/TII.2019.2897805  
**Last updated**: 2026-09-16

---

## 1. Completed Gates

| Gate | Status | Notes |
|---|---|---|
| `N30_ANALYTICAL_BASELINE_VERIFIED` | AWARDED | Paper-formula oracle exhaustive sweep |
| `N30_DES_FIDELITY_VERIFIED` | AWARDED | PBFT/Zyzzyva/Quorum DES vs analytical |
| `N30_VALIDATOR_SELECTION_LEARNING_SIGNAL_VERIFIED` | AWARDED | 384/435 C1-valid, 51 C1-invalid; binary reward confirmed |
| `N30_DRL_PILOT_VERIFIED` | AWARDED | 48/300 feasible with corrected 100k transactions |
| `DRL_MICRO_GROUND_TRUTH_VERIFIED` | AWARDED | N=6/K=4, zero-regret, 3 seeds, 150 eps each |
| `N30_SEED42_POLICY_VERIFIED` | PENDING | Awaiting ep400 checkpoint (~1.5h remaining as of 2026-09-16) |

---

## 2. Population Parameters (OUR_RECONSTRUCTION)

Liu paper specifies ranges, not distributions. Our reconstruction for N=30/K=28:

```python
N = 30; K = 28; CHI_BYTES = 200
STAKES  = tuple(float(4 + (i % 10)) for i in range(30))      # cycling 4-13
CAPS    = tuple(float(10 + (i % 21)) for i in range(30))     # cycling 10-30 GHz
POSITIONS = tuple((0.2*(i%5), 0.2*(i//5)) for i in range(30))  # 5×6 grid, 0-0.8 km
```

N=100/K=21 paper baseline: uniform stakes=25, cycling caps 10+(i%21), 10×10 regular grid.  
C1 threshold: stake_gini <= 0.2 (eta_s=0.2), geo_gini <= 0.3 (eta_l=0.3).  
Population Gini ≈ 0.194 for N=30 fixture → 384/435 sets pass C1, 51 fail.

---

## 3. Action Domain (N=30/K=28)

- **Validator sets**: C(30,28) = 435 sets
- **Protocol×S_B×T_I configs**: 9
  - LIU_QUORUM: (3.4/1.0), (6.8/2.0), (5.0/1.5)
  - PBFT: (7.0/3.0), (5.8/2.5), (4.6/2.0)
  - ZYZZYVA: (6.2/2.5), (7.4/3.0), (1.2/0.5)
- **Total actions**: 435 × 9 = **3915**
- **Optimal actions** (both GOOD and MEDIUM): 768 (= 384 valid sets × 2 max-omega configs)
- **Max omega**: floor(6.8e6/200)/2.0 = **17000** (LIU_QUORUM 6.8MB/2.0s)
- **REWARD_SCALE**: 17000.0

---

## 4. FSMC Reconstruction (OUR_RECONSTRUCTION)

```
L = 3 levels: LOW=10 Mbps, MID=55 Mbps, HIGH=100 Mbps
Transition matrix: diagonal=0.70, off-diagonal=0.15 each
Training uses GOOD (100 Mbps) and MEDIUM (55 Mbps) only.
POOR (10 Mbps) excluded from training: oracle shows 0/3915 feasible actions.
```

P(at least one link hits 10 Mbps from ANY state in 1 step | 435 links) = 1 - 0.85^435 > 0.9999.  
This means every training episode encounters POOR links. ~10.7% observed DES feasibility  
is a consequence of our reconstructed FSMC, NOT a Liu paper prediction.

---

## 5. Key Numerical Results

### 2400-Action Controlled Sweep (N=100/K=21, deterministic)
- 2400/2400 SUCCESS
- Paper C2 pass: 2013 (83.9%), DES C2 pass: 2092 (87.2%)
- **79 actions fail paper formula but pass DES** — all PBFT large-S_B cases; source: M=1 in DES vs M=3 in paper
- **0 paper-pass DES-fail** — DES strictly covers paper feasibility
- Best paper reward: 17000 (LIU_QUORUM 3.4/1.0 and 6.8/2.0)
- Known mismatch: M_DES=1 vs M_paper=3 (batch processing rounds); documented as KNOWN_MISMATCH

### N=30 Oracle (DES-verified, deterministic)
- GOOD_100Mbps: oracle_best=17000, n_optimal=768, feasible=3456/3915
- MEDIUM_55Mbps: oracle_best=17000, n_optimal=768, feasible=3456/3915
- POOR_10Mbps: oracle_best=0, feasible=0/3915

### N=30 Validator Sensitivity (paper formula)
- GOOD/MEDIUM all configs: 384/435 feasible (88.3% C1 pass rate)
- POOR all configs: 0/435 feasible
- Distinct reward values per config: {0.0, max_omega} — binary signal only
- No state-dependent validator adaptation: all 384 valid sets are reward-equivalent per config

### Micro DRL (N=6/K=4, 180 actions)
- Ground truth: GOOD 180/180 feasible, MEDIUM/POOR 30/180 optimal
- 3 seeds (42/123/7), 150 eps each, 2120s total runtime
- Evaluation: zero regret at epsilon=0 for both GOOD and MEDIUM states
- Key finding: 0.0 regret = policy achieved true optimal reward in each state

---

## 6. Protocol Fidelity Status

| Protocol | Analytical | DES | Status |
|---|---|---|---|
| PBFT | Eq.15 App.B | LiuPBFT runtime | PAPER_EXACT (with M=1 known mismatch) |
| Zyzzyva | Eq.16/17 App.B | LiuZyzzyva fast+recovery | PAPER_EXACT |
| LIU_QUORUM | Eq.Quorum | LiuQuorum runtime | PAPER_EXACT |

- **M_DES=1 vs M_paper=3**: DES simulates single-block rounds (M=1). Paper uses M=3 batch.  
  Effect: DES T_C shorter for PBFT → 79 paper-fail DES-pass cases.  
  Documented, not changed: our DES is physically correct for 1 block/round.
- **Transmission ×8 factor**: MB→Mb conversion required; paper silent. Applied consistently.
- **Alpha/beta units**: treated as megacycles (MHz notation). T_V ~ 0.7-2 ms per request.

---

## 7. Important Bug Fixes

### offered_workload_tx = 100_000 (NOT 30_000)
- **Root cause**: train_n30.py initially used 30_000. For S_B >= 6.0 MB (capacity >= 30k tx),
  block 1 exhausted all transactions, block 2 had 0 tx and stalled until C2 deadline → 98% C2 failures.
- **Fix**: Changed to 100_000 (matches controlled sweep). Covers 2 full blocks at max S_B=7.4 MB
  (capacity=37k tx × 2 = 74k < 100k).
- **File**: `experiments/liu_replication/drl/n30/train_n30.py:169`

### Block Capacity Exact Formula
- `block_capacity = floor(S_B_bytes / chi_bytes)` transactions per block.
- S_B in MB × 10^6 to bytes, then integer division by chi_bytes=200.
- This exact formula is required; any rounding error causes capacity errors.

### C2 Timing Semantics (height=2 DES)
- `run_single_action` bootstraps to height=1 (no deadline), applies action at boundary_time,
  then runs to height=2 with deadline = `boundary_time + omega * T_I` where omega=6.
- C2 deadline is finality deadline, NOT epoch interval deadline.

---

## 8. Reconstruction Decisions

### Geographic Lambda
- Paper specifies G(lambda) = "geographic decentralization" with lambda(x) unstated.
- Implementation: grid positions, pairwise Euclidean gini. Matches App.A Eq.14 endorsement.
- Classification: OUR_RECONSTRUCTION (not PAPER_EXACT — continuous form unstated).

### FSMC Levels and Transitions
- Paper states "network state R(t) changes over time" but gives no numeric levels.
- Levels {10, 55, 100} Mbps chosen based on realistic IIoT link quality categories.
- Transition probability 0.70 diagonal is our reconstruction.
- All FSMC behavior labeled: OUR_RECONSTRUCTION.

### N=30/K=28 Reasoning
- Paper: N=100, K=21. Liu DRL exhaustive action domain impractical at N=100.
- N=30/K=28 chosen to keep C(N,K) tractable: C(30,28)=435, C(30,28)×9=3915 total actions.
- This is NOT a scaled Liu population — it is a reduced exhaustive validation environment.
- K=28 rationale: high K/N ratio (93%) mirrors paper's K/N=21% ratio (reversed: high validator fraction).
- K=21 vs K=28 analytical T_C difference: ≤0.001s (negligible).

### DQN Architecture
- `Q(S,A)`: candidate-conditioned scalar. Feature = cat(state, action_embedding).
- Hidden: [256,128] for N=30. Exact Liu paper architecture not specified.
- Epsilon: linear decay 1.0→0.05 over 90% of total steps.
- Gamma: 0.9 (OUR_RECONSTRUCTION).
- All hyperparameters: OUR_RECONSTRUCTION (paper does not publish DRL hyperparameters).

---

## 9. Known Limitations

1. **Validator learning is static C1 feasibility only**: DQN can learn to avoid 51 C1-invalid
   sets. Cannot learn state-dependent validator adaptation because all 384 valid sets are
   reward-equivalent at GOOD/MEDIUM. This is a scientific limitation of the N=30 reward landscape.

2. **FSMC reconstruction uncertainty**: True Liu FSMC parameters unknown. Our 3-level
   reconstruction yields ~10.7% DES feasibility during training, which is lower than the
   deterministic GOOD/MEDIUM oracle (88.3%). This gap is fundamental, not a bug.

3. **M=1 vs M=3**: Our DES models single-block epochs. Liu paper uses M=3 batch blocks.
   Impact: 79 PBFT actions that paper classifies as infeasible are DES-feasible. Not corrected:
   our DES faithfully implements the reward formula with M=1.

4. **N=30 is not N=100**: Results from N=30/K=28 cannot directly reproduce Liu's Figure 5
   (which uses N=100/K=21 with 64-candidate sampling and offline pre-training).

5. **No offline pre-training**: Liu's Algorithm 1 includes an offline data collection phase.
   Our implementation uses online epsilon-greedy only. Classification: OUR_RECONSTRUCTION.

---

## 10. Current N=30 DRL State (as of 2026-09-16)

- **Active training**: Seed 42, ep ~277/399, epsilon ~0.269
- **Latest checkpoint**: `ckpt_seed42_ep250.pt` (18:56:34)
- **Expected completion**: ~1.5h from this writing
- **Training config**: 400 episodes × 20 steps, GOOD/MEDIUM alternating env
- **Seeds**: [42, 123, 7] sequentially in one process
- **Step log**: `experiments/liu_replication/drl/n30/step_log_seed42.csv`
- **Process**: Background task `btf0wyz0w`

**Intermediate gate findings (ep 0-210)**:
- 0 C1 failures in 1139 greedy steps (validator learning complete)
- Greedy feasibility 13.5% vs explore 11.1% (+2.4pp positive signal)
- 100% of C2 failures coincide with link_min=10 Mbps (FSMC-driven)
- DQN favoring ZYZZYVA 7.4/3.0 (easier C2 deadline) over optimal LIU_QUORUM 6.8/2.0

**Post-evaluation plan**: After ep400 checkpoint:
1. Stop process before seed 123 starts
2. Epsilon=0 evaluation: DQN vs random vs STRONG_FIXED_BASELINE (LIU_QUORUM 6.8/2.0)
3. Paired comparison on identical FSMC trajectory seeds
4. Award/deny `N30_SEED42_POLICY_VERIFIED`
5. If pass: proceed with seeds 123/7

---

## 11. File Locations After Cleanup

| Artifact | Location |
|---|---|
| Core Liu DES runtime | `src/Simulator/Chain/Consensus/LiuRuntime/` |
| Liu analytical reference | `src/Simulator/Liu/` |
| Action domain builder | `experiments/liu_replication/drl/build_action_domain.py` |
| DQN agent | `experiments/liu_replication/drl/dqn_agent.py` |
| N=30 trainer | `experiments/liu_replication/drl/n30/train_n30.py` |
| N=30 oracle | `experiments/liu_replication/drl/n30/oracle_paper_reference.json` |
| N=30 validator sensitivity | `experiments/liu_replication/drl/n30/validator_sensitivity.json` |
| Micro DRL trainer | `experiments/liu_replication/drl/micro/train_micro.py` |
| Micro evaluation | `experiments/liu_replication/drl/micro/evaluation_summary.json` |
| Controlled sweep runner | `experiments/liu_replication/controlled_sweep/run_controlled_full.py` |
| Controlled sweep final summary | `experiments/liu_replication/controlled_sweep/controlled_full_summary.json` |
| DES methodology docs | `experiments/liu_replication/des_digital_twin/` |
| Paper fidelity audit | `experiments/liu_replication/paper_fidelity_audit/` |
| Reconstruction decisions | `experiments/liu_replication/reference_population/` |
| DRL unit tests | `experiments/liu_replication/drl/tests/` |
| Core Liu tests | `tests/test_liu_*.py` |

---

## 12. Unresolved Items

- N30 seed-42 final evaluation (N30_SEED42_POLICY_VERIFIED): PENDING
- Seeds 123/7: will proceed only if seed-42 gate passes
- Final comparison figure (DQN vs random vs STRONG_FIXED_BASELINE): PENDING
- Final documentation milestone (Phase 13-14): PENDING after gate
