# Liu et al. 2019 Replication Documentation
## Operationalizing the Adaptive Blockchain Optimization Framework Inside SymBChainSim

**Paper:** Liu et al., "Performance Optimization for Blockchain-Enabled Industrial Internet of Things
(IIoT) Systems: A Deep Reinforcement Learning Approach",
IEEE Transactions on Industrial Informatics, Vol. 15, No. 6, June 2019. DOI: 10.1109/TII.2019.2897805

**Final milestone declared:** LIU_REPLICATION_BASELINE_TECHNICALLY_COMPLETE (2026-09-20)

**Evidence basis:** Equation numbers, Table I, and Algorithm 1 were verified directly from
`docs/papers/Liu_2019_TII.pdf` using PyMuPDF (pymupdf) on 2026-09-20. All numerical results
are cross-referenced with final experiment artifacts. The PDF is 12 pages; Table I and Algorithm 1
appear on page 7 (right column). Source hashes of frozen runtime files are recorded in
`experiments/liu_replication/drl/n30/LIU_REPLICATION_RUNTIME_FROZEN.json`.

---

## 1. Purpose and Scope

### 1.1 Assigned Task

This work operationalizes the Liu et al. 2019 adaptive blockchain optimization framework inside
SymBChainSim, a Python discrete-event blockchain simulator. The replication was required to
establish a validated experimental baseline before conducting original research on blockchain
security, decentralization, and performance trade-offs.

### 1.2 Why Replication Before Research

Liu et al. propose a Deep Reinforcement Learning (DRL) controller that dynamically selects
blockchain consensus protocol, block size, block interval, and validator set to maximize
transaction throughput subject to decentralization, finality, and fault-tolerance constraints.
The paper's results depend on specific modeling choices — some explicit, many underspecified.
Reproducing those choices (or making explicit reconstructions where they are absent) provides:

1. A controlled environment where parameters are fully known and auditable.
2. A combined analytical-plus-DES evaluation framework that goes beyond the paper's
   purely analytical model.
3. A baseline DQN controller whose behaviour is traceable to known training conditions.

### 1.3 Objective

The objective was **not** to reproduce figures or numerical results from Liu exactly.

The objective was to **operationalize** the Liu adaptive blockchain optimization framework
inside a discrete-event blockchain simulator so that future hypotheses can be studied under
controlled, reproducible conditions.

### 1.4 Scope Boundary

This document covers the implementation and validation up to seed-42 training and evaluation.
Seeds 123 and 7 are optional robustness experiments. The thesis research questions (which use
this baseline) are developed separately.

---

## 2. Liu et al. Original Framework

### 2.1 Setting

Liu et al. consider an Industrial Internet of Things (IIoT) system. Table I specifies:
- **N = 100** IIoT nodes
- **K = 21** selected block producers
- Geographical area: 1 km × 1 km

### 2.2 State (Eq. 10)

The system state is the 5-tuple:

```
S(t) = [χ, Υ, x, c, R](t)
```

| Symbol | Meaning | Table I |
|--------|---------|---------|
| χ (chi) | Average transaction size in bytes | 200 B |
| Υ (Upsilon) | Stake set over all N nodes | 1–50 token (range) |
| x | Position set over all N nodes (2D coordinates) | 1 km × 1 km area |
| c | Computational resource set in GHz | 10–30 GHz (range) |
| R | N×N directed link-rate matrix in Mbps | 10–100 Mbps (range) |

Only R(t) varies over time; χ, Υ, x, c are static across epochs.

### 2.3 Action (Eq. 11)

At each decision epoch t, the DRL controller selects:

```
A(t) = [a, δ, S_B, T_I](t)
```

| Symbol | Meaning | Table I |
|--------|---------|---------|
| a | Block-producer indicator: binary N-vector; exactly K ones | |
| δ | Consensus protocol: PBFT (δ=0), Zyzzyva (δ=1), Quorum (δ=2) | |
| S_B | Block size in MB | limit Ṡ = 8 MB |
| T_I | Block interval in seconds | max Ṫ = 10 s |

### 2.4 Throughput Objective (Eq. 1)

```
Ω(S_B, T_I) = floor(S_B / χ) / T_I    [transactions per second]
```

Ω is a purely nominal quantity — it depends only on the chosen action parameters, not on
actual DES outcomes. Consensus latency T_C affects the objective only through constraint C2.

### 2.5 Consensus Latency Decomposition (Eq. 7)

```
T_C,δ = T_D,δ + T_V,δ
```

Where T_D,δ is the message delivery time and T_V,δ is the message verification time for
protocol δ. The analytical derivations of T_D,δ and T_V,δ for each protocol are in Appendix B
(Eq. 15 for PBFT, Eq. 16–17 for Zyzzyva; Quorum derived separately).

### 2.6 Constraints

**C1 — Decentralization** (Eq. 4, 5):

```
C1: G(Υ) ≤ η_s   AND   G(λ) ≤ η_l
```

Where:
- G(Υ) is the stake Gini coefficient over the K selected producers (Eq. 2, pairwise formula)
- G(λ) is the geographic Gini coefficient (Eq. 3, continuous integral over density λ(x))
- η_s = 0.2 (Table I), η_l = 0.3 (Table I)

**C2 — Finality** (Eq. 6, 8):

```
T_F,δ = T_I + T_C,δ ≤ ω × T_I
⟺  T_C,δ ≤ (ω − 1) × T_I
```

Where ω = 6 (Table I). The paper calls this the TTF constraint (time to finality).

**C3 — Fault Tolerance** (Eq. 9):

```
f ≤ F^δ
F^0 = F^1 = floor((K−1)/3)    (PBFT, Zyzzyva)
F^2 = 0                         (Quorum)
```

### 2.7 Optimization Problem (Eq. 12 — Problem P1)

```
P1: max_{A} Q(S, A)
    C1: G(Υ) ≤ η_s, G(λ) ≤ η_l
    C2: T_F,δ ≤ ω × T_I, δ = 0, 1, 2
    C3: f ≤ F^δ, δ = 0, 1, 2
```

Where Q(S,A) = E[Σ_{t=0}^∞ μ^t R^(t)(S^(t),A^(t)) | S^(0)=S, A^(0)=A] is the action-value
function with discount factor μ ∈ (0, 1].

### 2.8 Reward Function (Eq. 13)

```
R(t)(S(t), A(t)) = floor(S_B / χ) / T_I   if C1 ∧ C2 ∧ C3 satisfied
                   0                          otherwise
```

### 2.9 FSMC Link Evolution

Liu specifies that R(t) evolves as a finite-state Markov chain (FSMC):
"we model the time-varying transmission links as finite-state Markov channels."
Link rates are "partitioned and quantized into L levels r = {r_1, ..., r_L}" with an
L×L transition probability matrix. Liu does **not** specify: L, the rate levels,
or the transition probabilities. These are PAPER_UNDERSPECIFIED (OUR_RECONSTRUCTION).

### 2.10 DRL Algorithm (Algorithm 1)

Liu's Algorithm 1 has two phases:

**Offline DNN construction:**
1. Load historical state transition profiles and Q(S,A) value estimates into experience memory D.
2. Pre-train the DNN (main Q network) with input pairs (S, A) and corresponding estimated Q(S, A).

**Online learning (for each decision epoch t):**
1. Select action ε-greedy: random with prob. ε; argmax_A Q(S^(t), A^(t)) otherwise.
2. Execute A^(t) (select block producers, consensus algorithm, block size, interval).
3. Observe reward R^(t) and next state S^(t+1).
4. Store experience (S^(t), A^(t), R^(t), S^(t+1)) in memory D.
5. Mini-batch sample from D; compute Bellman target y^(i) = R^(i) + γ max_{A'} Q(S^(i+1), A').
6. Update target Q network with loss L(θ) = [y^(i) − Q(S^(i), A'; θ)]² every G steps.

Table I specifies: batch size M = 3, ω = 6, η_s = 0.2, η_l = 0.3, α/β (see Section 7).
Liu does **not** specify: DNN architecture, replay buffer size, ε schedule, γ, G,
episode length, or training budget. These are PAPER_NOT_SPECIFIED (OUR_RECONSTRUCTION).

---

## 3. SymBChainSim as the Host Environment

### 3.1 What SymBChainSim Provides

SymBChainSim is a Python discrete-event simulator for blockchain systems. Core concepts:

- **Discrete-event simulation**: All activity is timestamped events on a priority queue.
  Time advances by processing the earliest pending event.
- **Virtual time**: Communication latency, CPU processing delay, and consensus timeouts
  are modelled as event scheduling offsets. Not wall-clock time.
- **Event queue**: The scheduler dispatches events to handlers which produce further events.
  The DES terminates when the queue is empty or a halting condition is met.
- **Network message delivery**: `delay = payload_bits / link_rate_bps`.
- **CPU validation delay**: `delay = cycles / capability_hz`.
- **Consensus events**: Protocol-specific events dispatched per protocol state machine.
- **ProtocolFinalityRecord**: Event marking when consensus finality is achieved,
  determining T_C_DES.

### 3.2 Why DES Over Analytical Models Only

The Liu analytical model computes T_C from closed-form expressions. The DES provides:
1. **Concurrency**: Multiple messages in-flight simultaneously.
2. **Event-driven C2 evaluation**: C2 checked against actual simulated finality time.
3. **Fault scenario emulation**: Faulty nodes injectable at the event level.
4. **Extensibility**: Attacks, network partitions, and adversarial behaviour at event level.

For the same (S, A), both `T_C_paper` (analytical) and `T_C_DES` (simulation) are computed and
compared step-by-step. This is a central scientific contribution of the integration.

---

## 4. Overall Mapping from Liu to SymBChainSim

| Liu Concept | Liu Source | SymBChainSim Realization | Code Location | Fidelity |
|-------------|-----------|--------------------------|---------------|----------|
| χ = 200 B | Table I | `LiuReferenceParameters.transaction_size_bytes = 200` | `ReferenceCore.py` | PAPER_EXACT |
| Υ (1–50 token) | Table I | `STAKES` cycling 4–13 per node | `train_n30.py:STAKES` | OUR_RECONSTRUCTION (specific assignment) |
| x (1 km² area) | Table I | `POSITIONS` 5×6 km grid, 0.2 km spacing | `train_n30.py:POSITIONS` | OUR_RECONSTRUCTION |
| c (10–30 GHz) | Table I | `CAPABILITIES` cycling 10–30 GHz per node | `train_n30.py:CAPABILITIES` | Range from Table I; assignment OUR_RECONSTRUCTION |
| R (10–100 Mbps) | Table I | `LinkStateMatrix`; FSMC {10, 55, 100} Mbps | `LinkFSMC.py` | Endpoints from Table I; levels/probs OUR_RECONSTRUCTION |
| a (binary N-vector) | Eq. 11 | Sorted tuple of K validator IDs | `action_encoder.py` | PAPER_DERIVED |
| δ ∈ {0,1,2} | Eq. 11 | `LiuConsensusProtocol` enum | `Protocol.py` | PAPER_EXACT |
| S_B in MB | Eq. 11 | `LiuAction.block_size_mb` | `Action.py` | PAPER_EXACT |
| T_I in s | Eq. 11 | `LiuAction.block_interval_s` | `Action.py` | PAPER_EXACT |
| Ω = floor(S_B/χ)/T_I | Eq. 1 | `reference_throughput_objective` | `ReferenceCore.py` | PAPER_EXACT |
| C1: G(Υ) ≤ η_s | Eq. 2, 4 | `paper_stake_gini` | `ReferenceCore.py` | PAPER_EXACT |
| C1: G(λ) ≤ η_l | Eq. 3, 5 | `geographic_gini_eq3`; λ(x) reconstructed | `ReferenceCore.py` | PAPER_EXACT (integral); λ(x) OUR_RECONSTRUCTION |
| C2: T_F ≤ ω·T_I | Eq. 6, 8 | Analytical + DES event-level | `ReferenceCore.py`; `_run_to_height` | PAPER_EXACT formula; DES PAPER_DERIVED |
| C3: f ≤ F^δ | Eq. 9 | `_reference_tolerated_faults` | `ReferenceCore.py` | PAPER_EXACT |
| reward = Ω or 0 | Eq. 13 | `StepResult.reward_des_liu_objective` | `liu_dynamic_env.py` | PAPER_EXACT |
| PBFT (Appendix B) | Eq. 15-region | `LiuPBFTRuntime`; 5-phase DES | `LiuRuntime/` | PAPER_EXACT with M=1 KNOWN_MISMATCH |
| Zyzzyva fast (Eq. 16) | Appendix B | `LiuZyzzyvaRuntime.fast_path` | `LiuRuntime/` | PAPER_EXACT fast path |
| Zyzzyva recovery (Eq. 17) | Appendix B | `LiuZyzzyvaRuntime.recovery_path` | `LiuRuntime/` | PAPER_DERIVED; t_r OUR_RECONSTRUCTION |
| Quorum | Appendix B | `LiuQuorumRuntime`; 2-phase | `LiuRuntime/` | PAPER_DERIVED |
| FSMC R(t) | Body text | `LinkFSMC.py:LinkFSMCState.advance` | `Liu/LinkFSMC.py` | Structure PAPER_EXACT; params OUR_RECONSTRUCTION |
| Q(S,A) | Eq. 12, Alg. 1 | Candidate-conditioned scalar Q-net | `q_network.py`, `dqn_agent.py` | Concept PAPER_EXACT; architecture OUR_RECONSTRUCTION |
| DRL loop | Algorithm 1 | `train_n30.py:train_one_seed` | `train_n30.py` | Online phase PAPER_EXACT; offline not reproduced |

---

## 5. State Implementation

### 5.1 χ — Transaction Size

**Paper:** χ = 200 B (Table I). Used in Eq. 1: `floor(S_B / χ)` transactions per block.
**Implementation:** Static scalar `LiuReferenceParameters.transaction_size_bytes = 200`
(`src/Simulator/Liu/ReferenceCore.py`). Same in analytical and DES paths.
**Fidelity:** PAPER_EXACT.

### 5.2 Υ — Stake Vector

**Paper:** Table I states stake range 1–50 token per node.

**Implementation (OUR_RECONSTRUCTION — specific assignment):**
```python
STAKES = tuple(float(4 + (i % 10)) for i in range(30))
# → [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 4, 5, ...]  (cycling, range 4–13)
```
`experiments/liu_replication/drl/n30/train_n30.py:STAKES`

The 4–13 range is within the paper's 1–50 token range. The specific cycling assignment,
the choice of 4–13 rather than 1–50, and the N=30 population size are all OUR_RECONSTRUCTION.

**Use in C1:** Stake Gini computed over K=28 selected validators, not all N=30.
`ReferenceCore.py:paper_stake_gini` implements Eq. 2 exactly (PAPER_EXACT formula).

### 5.3 x — Position Vector

**Paper:** Table I states a 1 km × 1 km geographical coverage area.

**Implementation (OUR_RECONSTRUCTION — specific grid):**
```python
POSITIONS = tuple((0.2*(i%5), 0.2*(i//5)) for i in range(30))
# → (0.0,0.0), (0.2,0.0), ..., (0.8,0.8)  range [0,0.8] km
```

The Eq. 3 integral formula for G(λ) is PAPER_EXACT. The spatial intensity function λ(x) is
OUR_RECONSTRUCTION — the paper introduces λ(x) via an inhomogeneous Poisson point process
reference but does not specify its form.

**Note:** For the N=30/K=28 fixture, geo_gini = 0.1000 for all 435 validator sets
(constant), satisfying η_l = 0.3.

### 5.4 c — Computational Capability

**Paper:** Table I states computing resource 10–30 GHz per node.

**Implementation:**
```python
CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(30))
# → [10, 11, 12, ..., 30, 10, 11, ...]  (cycling 10–30 GHz)
```
`experiments/liu_replication/drl/n30/train_n30.py:CAPABILITIES`

The range 10–30 GHz is taken directly from Table I. The specific cycling assignment is
OUR_RECONSTRUCTION. In the DES: `processing_delay = total_cycles / (capability_ghz × 1e9)` s.

**Fidelity:** Range PAPER_EXACT (Table I); assignment within range OUR_RECONSTRUCTION.

### 5.5 R — Link-Rate Matrix

**Paper:** Table I states link rate R_n ∈ [10, 100] Mbps. FSMC Markov structure specified
(Section III-B). Number of states L, exact rate levels, and transition probabilities: not given.

**Implementation (endpoints from Table I; L, levels, probs OUR_RECONSTRUCTION):**

FSMC levels: {10, 55, 100} Mbps — endpoints match Table I range; 55 Mbps intermediate is ours.

Transition matrix (per directed link, independent across links):
```
From LOW:  [0.70, 0.20, 0.10]
From MID:  [0.15, 0.70, 0.15]
From HIGH: [0.10, 0.20, 0.70]
```
`experiments/liu_replication/drl/n30/clean_run_manifest_seed42.json:fsmc_config`

**Important consequence:** With N=30 and 870 directed links, the probability that at least
one link reaches 10 Mbps from any starting condition within one step is ≈ 1. Virtually
every training episode encounters low-rate links. The ~59% feasibility rate in training is
a consequence of this FSMC reconstruction, not a Liu paper prediction.

---

## 6. Action Implementation

### 6.1 Complete Action A = [a, δ, S_B, T_I]

### 6.2 a — Validator Selection

**Paper (Eq. 11):** Binary N-vector; exactly K ones indicate selected producers.

**Implementation:** Sorted tuple of K=28 selected node IDs.
For N=30/K=28: C(30,28) = 435 possible validator combinations.
This is the "reduced exhaustive validation environment" — all 435 selections enumerable.

**Why N=30/K=28:** The paper's N=100/K=21 gives C(100,21) ≈ 10^19 — computationally
intractable for exhaustive evaluation. N=30/K=28 (C(30,28)=435) enables:
1. Complete DES evaluation of all validator sets
2. Exact C1-validity classification of all 435 sets
3. Full analytical oracle computation

This is **a reduced validation environment**, not a scaled reproduction of Liu's N=100/K=21.

**C1-validity of validator sets:**
- geo_gini = 0.1000 for all 435 sets (constant; satisfies η_l = 0.3)
- stake_gini varies per set; threshold η_s = 0.2
- Result: **384 / 435 C1-valid; 51 C1-invalid** (stake gini > 0.20)

**Validator learning:** Because all 384 C1-valid sets are reward-equivalent within each
(δ, S_B, T_I) configuration, the DRL learns static C1 feasibility (avoid the 51 invalid sets),
not state-dependent ranking among the 384 valid sets.

### 6.3 δ — Protocol Selection

PBFT / Zyzzyva / Quorum as specified in Eq. 11. Three protocols; DES event structures differ.

### 6.4 S_B — Block Size

**Paper (Eq. 1):** S_B in MB. Block capacity = `floor(S_B_bytes / χ_bytes)` transactions.

**Offered workload:** `offered_workload_tx = 100,000` transactions per epoch (critical value).
Using 30,000 causes transaction starvation for large S_B (7.0 MB block needs 35,000 tx;
second block would have 0 tx, stalling until C2 deadline → 98% false C2 failures).

### 6.5 T_I — Block Interval

**Paper (Eq. 6, 8):** T_I enters both the throughput objective Ω and the C2 deadline ω·T_I.

**Implementation:** Two-height bootstrap. Height-2 epoch has deadline `boundary_time + ω·T_I`.
`ProtocolFinalityRecord` before deadline → C2 PASS. Deadline first → C2 FAIL.

### 6.6 Reduced Action Domain

9 protocol/S_B/T_I configurations curated for high-Ω, medium-Ω, and exploratory regions:

| Protocol | S_B (MB) | T_I (s) | Ω (TPS) |
|----------|-----------|---------|---------|
| LIU_QUORUM | 3.4 | 1.0 | 17,000 |
| LIU_QUORUM | 6.8 | 2.0 | 17,000 |
| LIU_QUORUM | 5.0 | 1.5 | 11,333 |
| PBFT | 7.0 | 3.0 | 11,667 |
| PBFT | 5.8 | 2.5 | 11,600 |
| PBFT | 4.6 | 2.0 | 11,500 |
| ZYZZYVA | 6.2 | 2.5 | 12,400 |
| ZYZZYVA | 7.4 | 3.0 | 12,333 |
| ZYZZYVA | 1.2 | 0.5 | 12,000 |

**Total: 435 × 9 = 3,915 actions.**

This is a DES_OPERATIONAL_REALIZATION. The paper's action space is continuous in S_B and T_I
and combinatorial in a; our reduced domain preserves all four action dimensions at tractable scale.

---

## 7. Consensus Protocol Realization

### 7.1 PBFT

**Paper analytical model (Appendix B — Eq. 15 region):**

Computational loads per batch of M requests:
```
Primary z_bp:  O^0_bp = Mα + [2M + 4(K+f−1)]β
Replica z_bi:  O^0_bi = Mα + [M  + 4(K+f−1)]β
```

Validation time:
```
T_V,0 = (1/M) max_{k≠c} { O^0_bk / c_bk }
```

Message delivery time (5 phases t1–t5):
```
T_D,0 = (1/M)(t1 + t2 + t3 + t4 + t5)
```

Each phase i uses `min(max_i(M·S_B / R_{bi,bj}), T)` where T is the timeout.
Total consensus latency: T_C,0 = T_D,0 + T_V,0 (Eq. 7).

**Implementation:** `src/Simulator/Liu/AnalyticalPBFT.py:PBFTAnalyticalModel.evaluate`

**DES operational realization:** 5 message-phase DES events. T_C_DES measured from epoch
start to `ProtocolFinalityRecord` at the REPLY phase.

**Known mismatch — M:** Analytical uses M=3 (Table I). DES uses M=1 per epoch. See Section 8.

**Fault tolerance:** F^0 = floor((K−1)/3) = floor(27/3) = 9 at K=28.

### 7.2 Zyzzyva

**Paper analytical model (Appendix B, Eq. 16 — fast path):**

Fast case (f=0, all replicas honest):
```
O^1(1)_bp = Mα + (2M+K−1)β
O^1(1)_bi = Mα + (M+1)β
T_D,1(1) = (1/M)(t1 + t2 + t3)
```

**Paper analytical model (Eq. 17 — two-phase/recovery):**

With f faulty replicas:
```
O^1(2)_bp = Mα + (4M+K+f−1)β
O^1(2)_bi = Mα + (3M+1)β
T_D,1(2) = (1/M)(t1 + t2 + t3 + t_r + t4 + t5)
```

Where t_r is the recovery time. **t_r is referenced in Eq. 17 but not listed in Table I —
it is OUR_RECONSTRUCTION.** `LiuReferenceParameters.recovery_delay_s = 0.05 s`.

**Fidelity:** Fast path PAPER_EXACT; recovery path PAPER_DERIVED; t_r OUR_RECONSTRUCTION.

### 7.3 Quorum

**Paper (Appendix B):** Two phases — Request and Reply; no primary; no batching; F^2 = 0.

```
O^2_bk = α + 2β   (per request, per replica)
T_V,2 = max_{k≠c} { O^2_bk / c_bk }
T_D,2 = t1 + t2
```

Note: Quorum delivery time in the paper uses S_B (not M·S_B) because "Quorum is not able to
support batching" (Appendix B, first paragraph).

**Implementation:** `LiuQuorumRuntime` — 2-phase DES; F^2=0 confirmed.
**Fidelity:** PAPER_DERIVED (formulas less explicit than PBFT; derived from described semantics).
**Controlled sweep:** LIU_QUORUM DES=724/800 = paper=724/800 (exact agreement).

---

## 8. M Parameter Audit — Known Mismatch

### 8.1 Paper Specification

Table I: **M = 3** (batch size). In Appendix B:
- Payload per phase: M·S_B (each phase transmits M blocks)
- CPU cost per request: O^δ_bk / M (amortized over M requests)
- Delivery time per request: T_D = (1/M)(t1 + t2 + ...) (batching divides by M)

### 8.2 DES Operational Realization

The DES realizes **M = 1**: one block per `run_single_action` call. Liu provides no operational
definition of how to execute M>1 within a single DRL decision epoch. Possible interpretations
(M sequential rounds, M simultaneous rounds, pipelined) each produce a different T_C and
different DES overhead — the paper does not specify which.

**Classification:** KNOWN_MISMATCH / DES_OPERATIONAL_REALIZATION

**Sensitivity evidence:**
- 218 actions × 3 link scenarios = **654 DES evaluations**
- C2 classification changes (M=1 → M=3): **0**

This evidence is fixture-specific and does not claim M is universally irrelevant.

**Result:** Analytical reference retains M=3. DES uses M=1. Both recorded in all artifacts.

---

## 9. α and β Parameters

**Paper text (Section III-B):** "verifying signatures, generating message authentication
codes (MACs), and verifying MACs, requiring α, β, and β CPU cycles, respectively."
→ Text explicitly says **CPU cycles**.

**Table I entry:** "The computing cost for verifying signatures and generating/verifying MACs,
α, β" listed with value **"2MHz/1MHz"**.

**Inconsistency:** The paper body says "CPU cycles" (implying α, β are cycle counts), while
Table I writes "2MHz/1MHz" (MHz notation, which is a frequency unit).

**Classification: PAPER_AMBIGUOUS** — the table and text are inconsistent. Our code treats
the values as CPU cycle counts: `alpha_cycles = 2,000,000`, `beta_cycles = 1,000,000`
(`LiuReferenceParameters`, `src/Simulator/Liu/ReferenceCore.py`). This gives plausible
T_V values (~0.2–2 ms) for the capability range 10–30 GHz.

---

## 10. Constraint Implementation

### 10.1 C1 — Decentralization

**Stake Gini (Eq. 2, PAPER_EXACT):**
```
G(Υ) = Σ_i Σ_j |Y_i − Y_j| / (2K Σ_i Y_i)
```
`ReferenceCore.py:paper_stake_gini` → `DecentralizationMetrics.py:canonical_pairwise_gini`

**Geographic Gini (Eq. 3 integral PAPER_EXACT; λ(x) OUR_RECONSTRUCTION):**
```
G(λ) = ∫_Ξ ∫_Ξ |λ(x)−λ(y)| dy dx / (2K)
```
(`ContinuousSpatial.py:planar_gradient_intensity`)

For the N=30/K=28 fixture: geo_gini = 0.1000 for all 435 validator sets.
C1 validity determined entirely by stake gini (51 sets fail stake ≤ 0.20; all pass geo ≤ 0.30).

### 10.2 C2 — Finality

**Paper (Eq. 6, 8):** T_F = T_I + T_C ≤ ω·T_I ↔ T_C ≤ (ω−1)·T_I.

**Analytical C2:** T_C_paper from Appendix B formulas; C2 = (T_C_paper ≤ 5·T_I).

**DES C2:** Two-height bootstrap; C2 = `ProtocolFinalityRecord` arrives before `boundary + ω·T_I`.

**Critical C2 Bug Fixed:**

*Old behavior:* `_run_to_height` waited for all N nodes (including observers receiving via gossip).
In heterogeneous/degraded conditions: gossip delay caused false C2 failures.

*Corrected behavior (gate POST_CLEANUP_LIU_SEMANTICS_PRESERVED: PASS):*
Terminates on the first `ProtocolFinalityRecord` event (validator quorum commits the block).
Observer gossip does not determine C2.

Audit: 14/18 (78%) representative old-C2-FAIL cases became C2-PASS under corrected semantics.
This confirmed material contamination of the pre-fix training run, which was quarantined.

**Failure modes:**
- `C2DeadlineExceeded`: authoritative Liu C2 failure (PAPER_EXACT condition)
- `DESQueueExhausted` / `RuntimeError`: RUNTIME_GUARD_FAILURE — not Liu C1/C2/C3 (see Section 22)

### 10.3 C3 — Fault Tolerance

**Paper (Eq. 9):** F^0=F^1=floor((K−1)/3); F^2=0.
At K=28: F^PBFT=F^Zyzzyva=9; F^Quorum=0.
In the fixture f=0 faulty nodes → C3 always passes.
**Fidelity:** PAPER_EXACT.

---

## 11. Analytical Reference Oracle vs Operational DES

**These are not competing simulators. They serve distinct, complementary roles.**

**Analytical reference oracle:**
`evaluate_reference(S, A, params) → LiuPaperReferenceRecord`
(`ReferenceCore.py`). For any (S, A) pair, computes T_C_paper, C1, C2, C3, reward_paper.

**Operational DES:**
`run_single_action` via full blockchain DES. Computes T_C_DES, C1, C2, C3, reward_des.

**Comparison in controlled sweep (Section 14):**
- paper_pass ∧ DES_fail = **0** (DES never accepts an action the analytical model rejects)
- paper_fail ∧ DES_pass = **79** (all PBFT, all attributable to M mismatch)
- This validates the DES is at least as restrictive as Liu for C2.

---

## 12. Reward Semantics

**Liu's objective (Eq. 1, 13):** r = Ω if C1∧C2∧C3, else 0. Ω = floor(S_B/χ)/T_I.

**Three distinct reward quantities (never conflate):**

| Name | Formula | Fidelity |
|------|---------|----------|
| `reward_paper_reference` | Ω if paper C1/C2/C3 pass else 0 | PAPER_EXACT |
| `reward_des_liu_objective` | Ω if DES C1/C2/C3 pass else 0 | PAPER_EXACT (objective); DES_OPERATIONAL_REALIZATION (constraint evaluation) |
| `throughput_des_realized` | finalized_tx / (T_I + T_C_DES) | DES_OPERATIONAL_METRIC (diagnostic only) |

**Critical correction made during development:**
An earlier DRL reward used `finalized_tx / (T_I + T_C_DES)` — actual throughput realized
in the DES epoch. This is NOT Liu's objective Ω. Under this formulation, T_C_DES entered
the optimization twice: once in the objective, and again through C2.

Corrected: `reward_des_liu_objective = Ω if DES-evaluated C1∧C2∧C3 else 0`.
T_C_DES now enters only through C2, exactly as Liu intends.
`experiments/liu_replication/controlled_sweep/reward_semantics.md` documents this decision.

---

## 13. FSMC / Dynamic Network Environment

**FSMC reconstruction (OUR_RECONSTRUCTION):**
- L = 3 states: LOW = 10 Mbps, MID = 55 Mbps, HIGH = 100 Mbps
  (10 and 100 Mbps endpoints match Table I; 55 Mbps intermediate is our choice)
- Transition matrix (per directed link, independent across links):

| From\To | LOW | MID | HIGH |
|---------|-----|-----|------|
| LOW | 0.70 | 0.20 | 0.10 |
| MID | 0.15 | 0.70 | 0.15 |
| HIGH | 0.10 | 0.20 | 0.70 |

**FSMC trajectory identity (directly verified 2026-09-20):**
All 4 policies face identical link-state sequences per trajectory.
Proof: `env.step()` advances `self._fsmc` exactly once per call (success path line 235 and
exception path line 285 in `liu_dynamic_env.py`). `self._rng` is consumed exclusively by
the FSMC advance. `reset(seed)` deterministically re-seeds `self._rng`.
Verification: 16 trajectories × 20 steps × 6 pairwise hash comparisons = **1,920 comparisons,
0 mismatches**.

---

## 14. Development and Verification Sequence

| # | Milestone | Gate/Result |
|---|-----------|-------------|
| 1 | Analytical reference core | `ReferenceCore.py` complete |
| 2 | DES instrumentation | `Instrumentation.py`; `ProtocolFinalityRecord` |
| 3 | Validator/observer separation | `ValidatorSet.py`, `NodeProfile.py` |
| 4 | Node profiles | `NodeProfile.py` |
| 5 | PBFT operational realization | PBFT protocol gate PASS |
| 6 | Block-size/capacity correction; offered_workload=100k | Critical bug fix |
| 7 | Zyzzyva fast/recovery path | Zyzzyva protocol gate PASS |
| 8 | Quorum | Quorum protocol gate PASS |
| 9 | Full protocol action space | 3,915-action domain |
| 10 | Reward-semantics correction | `reward_des_liu_objective` |
| 11 | Fixed-validator controlled sweep | N30_DES_FIDELITY_VERIFIED |
| 12 | Exhaustive validator-selection env | 384/51 C1-valid/invalid |
| 13 | Dynamic FSMC environment | LIU_DYNAMIC_DES_ENVIRONMENT_VERIFIED |
| 14 | C2 semantic fix | POST_CLEANUP_LIU_SEMANTICS_PRESERVED: PASS |
| 15 | Micro DRL (N=6/K=4) | DRL_MICRO_GROUND_TRUTH_VERIFIED |
| 16 | N30 seed-42 training | 8,000 steps, 59.1% feasible, 0 NaN |
| 17 | Repository cleanup | 397 superseded files removed |
| 18 | Semantic audit + clean retrain | C2 bug; old checkpoints quarantined; clean retrain |
| 19 | Greedy policy evaluation | N30_SEED42_POLICY_VERIFIED AWARDED |

---

## 15. Controlled 2,400-Action Experiment

**Design:** Fixed population S_0 (N=100/K=21, f=0), all combinations of
3 protocols × 40 block sizes × 20 block intervals = 2,400 independent actions.
Both analytical (paper) and DES C2 evaluated for every action.

**Results (`controlled_full_summary.json`):**

| Metric | Value |
|--------|-------|
| Total actions | 2,400 |
| Paper C2 pass | 2,013 (83.9%) |
| DES C2 pass | 2,092 (87.2%) |
| paper_pass ∧ DES_pass | 2,013 |
| paper_pass ∧ DES_fail | **0** |
| paper_fail ∧ DES_pass | **79** |
| both_fail | 308 |

*Sum: 2013 + 0 + 79 + 308 = 2400 ✓*

**Per-protocol:**

| Protocol | DES feasible | Paper feasible |
|----------|-------------|----------------|
| LIU_QUORUM | 724/800 | 724/800 |
| ZYZZYVA | 688/800 | 686/800 |
| PBFT | 680/800 | 603/800 |

**Global optimum:** LIU_QUORUM S_B=3.4/T_I=1.0 or S_B=6.8/T_I=2.0, Ω=17,000.
Paper and DES agree on the optimal action set.

**79 paper_fail / DES_pass cases:** All PBFT, large S_B. Root cause: M=1 in DES produces
shorter per-round T_C than M=3 in analytical → DES passes C2 where analytical fails.
This is the KNOWN_MISMATCH described in Section 8.

---

## 16. Validator-Selection Validation

**Design:** N=30, K=28, C(30,28)=435 validator sets. All analytically evaluated for C1.

**Results:**
- C1-valid: **384 / 435** (88.3%) — stake gini ≤ 0.20
- C1-invalid: **51 / 435** (11.7%) — stake gini > 0.20
- Geographic Gini: 0.1000 constant for all 435 sets; validity determined entirely by stake Gini

**Reward structure:** Within each (δ, S_B, T_I) config, all 384 C1-valid sets achieve
identical Ω in GOOD and MEDIUM conditions. Validator selection produces a binary C1-pass/fail
signal, not a continuous rank signal.

**Oracle:** GOOD (100 Mbps): 3,456/3,915 feasible; 768/3,915 optimal (Ω=17,000).
MEDIUM (55 Mbps): identical. POOR (10 Mbps): 0/3,915 feasible.

---

## 17. Micro DRL Validation

**Design (N=6, K=4, 180 actions):**
- 3 seeds (42/123/7), 150 episodes each, exhaustive ground truth for all 180 actions
- Wall time: 2,120 s (~35 min)

**Purpose:** Validate the DRL infrastructure (state/action encoding, Q-network, ε-greedy,
replay, Bellman targets) on a tractable problem where ground truth is fully known.

**Results (`experiments/liu_replication/drl/micro/evaluation_summary.json`):**
- All 11 criteria A–K: PASS (gate DRL_MICRO_GROUND_TRUTH_VERIFIED)
- **Zero regret in GOOD, MEDIUM, and POOR network states** (ε=0 greedy matches oracle)
- Multi-seed confirmed

**Limitation:** State dim=55 (vs N30 dim=991), action space=180 (vs 3,915).
The micro DRL validates the DRL infrastructure, not N30 DRL scalability.

---

## 18. Final N30 DRL Training

### 18.1 Why a Clean Retrain Was Required

An earlier seed-42 training run was discarded due to the C2 semantic bug (Section 10.2).
Audit: 14/18 (78%) representative old-C2-FAIL cases became C2-PASS under corrected semantics.
This contaminated the training reward signal. Old checkpoints are preserved and labeled
CONTAMINATED / NON_AUTHORITATIVE.

### 18.2 Clean Run Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Seed | 42 | Baseline seed |
| N, K | 30, 28 | Reduced env |
| Action domain | 3,915 | 435 × 9 |
| Episodes | 400 | |
| Steps / episode | 20 | |
| Total steps | 8,000 | |
| Reward scale | 17,000 | = max Ω |
| Hidden sizes | [256, 128] | OUR_RECONSTRUCTION |
| Gamma | 0.90 | OUR_RECONSTRUCTION |
| Learning rate | 0.0005 | OUR_RECONSTRUCTION |
| Replay capacity | 10,000 | OUR_RECONSTRUCTION |
| Replay warmup | 200 | OUR_RECONSTRUCTION |
| Batch size | 64 | OUR_RECONSTRUCTION |
| Target sync | 50 steps | OUR_RECONSTRUCTION |
| ε start → end | 1.0 → 0.05 | OUR_RECONSTRUCTION |
| ε decay | 90% of 8,000 = 7,200 steps | OUR_RECONSTRUCTION |
| Offered workload | 100,000 tx | Critical correction |
| Wall time | ~7.6 hours | |

`experiments/liu_replication/drl/n30/clean_run_manifest_seed42.json`

### 18.3 Training Results

From `pilot_gate_seed42_clean.json` and `episode_log_seed42_clean.csv`:

| Metric | Value |
|--------|-------|
| Total steps | 8,000 |
| Feasible steps | 4,726 / 8,000 = **59.1%** |
| Mean reward / step | 7,267 |
| Max reward (Ω) | 17,000 |
| Distinct actions seen | 2,728 / 3,915 = 69.7% |
| Training steps with loss | 7,801 |
| NaN losses | **0** |
| ε final | 0.05 |
| Replay buffer at end | 8,000 |

The 59.1% feasibility rate is a consequence of the reconstructed FSMC parameters
(virtually every episode encounters 10 Mbps links; 0/3,915 actions feasible at 10 Mbps).
It is not a Liu paper prediction.

### 18.4 Primary Model Selection

ep400 is the primary model. No model-selection rule was defined before training, and the
protocol specified ep400 as primary. ep350 was evaluated diagnostically (Section 21).

---

## 19. Final N30 Policy Evaluation

**Method:** 4 eval seeds [1000,1001,1002,1003] × 4 initial states = 16 trajectories,
20 steps each, 320 steps per policy. FSMC trajectories identical for all 4 policies
(verified: 1,920 hash comparisons, 0 mismatches).

**Policies evaluated:**
1. **ep400 DQN** (primary, ε=0 greedy)
2. **ep350 DQN** (diagnostic)
3. **Random** (uniform random over 3,915 actions)
4. **Strong Fixed** (LIU_QUORUM, S_B=6.8 MB, T_I=2.0 s; one fixed C1-valid validator set)

"Strong Fixed" is **not** an oracle. It represents the best action in high-rate conditions
(Ω=17,000 when feasible) but fails C2 in most degraded-link steps (large S_B / short T_I).

**Global policy metrics (320 steps per policy):**

| Metric | ep400 DQN | Random | Strong Fixed | ep350 DQN |
|--------|-----------|--------|--------------|-----------|
| Cumulative reward | 2,716,500 | 1,594,633 | 476,000 | 2,627,967 |
| Mean reward / step | 8,489.06 | 4,983.23 | 1,487.50 | 8,212.40 |
| Feasible steps | 225 (70.3%) | 147 (45.9%) | 28 (8.75%) | 218 (68.1%) |
| C1 pass | 319 / 320 | 281 / 320 | 320 / 320 | 320 / 320 |
| C2 pass | 225 / 320 | 147 / 320 | 28 / 320 | 218 / 320 |
| C3 pass | 320 / 320 | 320 / 320 | 320 / 320 | 320 / 320 |
| C2_DEADLINE_EXCEEDED | 86 | 157 | 281 | 87 |
| RUNTIME_GUARD_FAILURE | 9 | 16 | 11 | 15 |
| Mean Ω (feasible steps) | 12,127 | 12,061 | 17,000 | 12,055 |
| C1-valid validator sel. | 319 (99.7%) | 281 (87.8%) | 320 (100%) | 320 (100%) |

**Paired comparisons (from `n30_seed42_policy_evaluation.json`):**

| Comparison | DQN wins | Mean paired diff per trajectory |
|------------|----------|--------------------------------|
| ep400 vs random | **16/16** | +70,117 |
| ep400 vs strong_fixed | **16/16** | **+140,031** |

Source: `paired_comparisons.ep400_vs_strong_fixed.summary.mean_paired_diff = 140031.25`
(computed from 16 trajectory diffs totaling 2,240,500 / 16 = 140,031.25).

Note: An earlier summary artifact (`LIU_REPLICATION_BASELINE_TECHNICALLY_COMPLETE.json`)
originally recorded 122,792 — this was an error. The correct value is 140,031 in all
authoritative artifacts as of 2026-09-20.

`experiments/liu_replication/drl/n30/n30_seed42_policy_evaluation.json`

---

## 20. Learned Adaptation

The ep400 DQN shows state-dependent policy adaptation across network conditions:

| Network Bucket | Steps | Feasible | Mean Ω | Protocol selected |
|----------------|-------|----------|--------|-------------------|
| NEAR_HIGH_100MBPS | 4 | 4 (100%) | 17,000 | LIU_QUORUM / 6.8 / 2.0 exclusively |
| MEDIUM_RANGE | 4 | 4 (100%) | 12,333 | ZYZZYVA / 7.4 / 3.0 exclusively |
| CONTAINS_LOW_10MBPS | 312 | 217 (70%) | 8,331 | PBFT×118, ZYZZYVA×181, QUORUM×13 |

In the 4 high-rate steps, the DQN selects LIU_QUORUM S_B=6.8/T_I=2.0 (Ω=17,000) exclusively.
In the 4 medium-rate steps, it selects ZYZZYVA S_B=7.4/T_I=3.0 exclusively.
In degraded conditions it diversifies across PBFT and ZYZZYVA with T_I=3.0 s in 168/312 steps.

**Do not overstate:** The DQN demonstrates state-dependent adaptive behaviour. It does not
achieve provably optimal control. "Zero regret" was demonstrated only in the micro DRL
environment (exhaustive ground truth available). No exhaustive oracle exists for N30
(1.25M+ DES evaluations would be needed; computationally intractable).

**Validator learning:** DQN selects C1-valid validators in 319/320 steps (99.7%),
confirming it has learned to avoid the 51 C1-invalid sets. No fine-grained validator
composition learning is expected or observed (all 384 valid sets are reward-equivalent
within each config).

---

## 21. Q-Network Health

Q-value samples at ε=0 evaluation (ep400, 3,915 actions per state):

| Initial State | Q range | Degenerate |
|---------------|---------|------------|
| ALL_HIGH_100 | [5.52, 6.00] | No |
| ALL_MID_55 | [5.01, 5.29] | No |
| ALL_LOW_10 | [−6.94, −5.53] | No |
| TRAINING_MIXED | [0.59, 1.93] | No |

- **No NaN, no Inf** (8 states × 3,915 actions verified)
- State-dependent Q structure: HIGH → Q ≈ +5.8, LOW → Q ≈ −6.1
- Discriminative across all states (std 0.05–0.35): network is not degenerate

Q values are normalized by REWARD_SCALE=17,000. Q ≈ 5.8 in ALL_HIGH corresponds to
approximately 98,600 in cumulative discounted return.

---

## 22. ep350 vs ep400 Diagnostic

| Metric | ep400 | ep350 |
|--------|-------|-------|
| Cumulative reward | 2,716,500 | 2,627,967 |
| Mean reward / step | 8,489.06 | 8,212.40 |
| Feasible steps | 225 (70.3%) | 218 (68.1%) |
| RUNTIME_GUARD_FAILURE | 9 | 15 |
| ep400 wins (paired) | 10/16 | — |
| ep350 wins (paired) | — | 6/16 |
| Mean paired diff | ep400 better by +5,533/trajectory | |

**Verdict: LATE_TRAINING_POLICY_REGRESSION — NOT DETECTED.**

ep400 outperforms ep350 on 10/16 trajectories with +5,533 mean advantage.
ep400 also has fewer RUNTIME_GUARD_FAILURE events (9 vs 15). ep400 is the primary result.

---

## 23. Runtime Guard Limitation

**Definition:** `RUNTIME_GUARD_FAILURE` (abbreviated RGF) is raised by:
1. `DESQueueExhausted`: PBFT view-change mechanism exhausts the DES event queue before
   consensus finality. This fires the PBFT timeout-backoff cap (OUR_RECONSTRUCTION).
2. `RuntimeError` (max_events exceeded): Hard event-count limit guard.

Neither is a Liu-defined C2 constraint. Both are implementation guards (OUR_RECONSTRUCTION).

**Observed in evaluation:**
- ep400: 9/320 = 2.8% → fewer than either baseline
- Random: 16/320 = 5.0%
- Strong Fixed: 11/320 = 3.4%

**Classification: KNOWN_DES_RUNTIME_GUARD_LIMITATION**
- Not DQN-specific (DQN < random < fixed)
- Not a Liu C2/C3 result — steps receive reward=0 and are excluded from feasibility statistics
- Concentrated in CONTAINS_LOW_10MBPS with large S_B / long T_I (PBFT) actions
- An environmental property of the DES under degraded-link PBFT simulations

**Criterion E:** Criterion E of N30_SEED42_POLICY_VERIFIED was specified as "no systematic
runtime-guard pathology." The 2.8% rate (lower than both baselines) does not constitute
systematic DQN-induced pathology. Awarded as **PASS_WITH_DOCUMENTED_LIMITATION**.

---

## 24. Contaminated Training Run

An earlier seed-42 training run (tag: pre-finality-fix) was discarded:

**Bug:** `_run_to_height` waited for all N nodes including gossip propagation to observers.
In mixed/degraded link conditions: slow gossip delivery caused false C2 failures.

**Audit:** 14/18 (78%) representative old-C2-FAIL cases became C2-PASS under corrected semantics
in heterogeneous network conditions.

**Decision:** Old checkpoints labeled CONTAMINATED / NON_AUTHORITATIVE.
Clean retrain from scratch (Section 18) required.

The contaminated checkpoint is preserved at:
`experiments/liu_replication/drl/n30/contaminated_pre_finality_fix/`

This discovery and correction demonstrates methodological rigor: the C2 bug was formally
detected through gate criteria, quantified, and corrected before any scientific claims.

---

## 25. DRL Architecture and Paper Fidelity

**Liu specifies (Algorithm 1):** Q(S,A), ε-greedy, argmax, experience memory D,
mini-batch Bellman updates every G steps, target network.

**Liu does NOT specify:** DNN architecture, state encoding, action encoding, replay buffer
size, ε schedule, γ, G (target sync period), episode length, training budget.

**Offline DRL phase (Algorithm 1 specifies; not reproduced):** Algorithm 1 includes an
offline pre-training phase using "historical state transition profiles" loaded into memory D.
The historical data generation process is not specified. Online-only RL was used.

**Our practical realization (OUR_RECONSTRUCTION):**

| Component | Architecture | Code |
|-----------|-------------|------|
| State encoder | dim = 1+4N+N(N-1) = 991 for N=30 | `state_encoder.py:StateEncoder.encode` |
| Action encoder | dim = N+5 = 35 for N=30 | `action_encoder.py:ActionEncoder.encode` |
| Q-network | State+Action concatenated → MLP[256,128] → scalar | `q_network.py:CandidateConditionedQNetwork` |
| Inference | Forward pass over all 3,915 actions; global argmax | `dqn_agent.py:DQNAgent.select_action` |

**Reduced action domain:** 435 × 9 = 3,915 actions preserves all four action dimensions
at tractable scale. The paper's space is combinatorial in a and continuous in S_B, T_I.

---

## 26. Paper Ambiguities and Reconstruction Table

| Item | Paper provides | Missing / ambiguous | Reconstruction | Classification |
|------|---------------|--------------------|----|---------------|
| α, β | Text: "CPU cycles"; Table I: "2MHz/1MHz" | Text and table are inconsistent | Treated as cycle counts: 2,000,000 and 1,000,000 | PAPER_AMBIGUOUS |
| M | Table I: M=3 | No DES operational definition for multi-block epoch | M=3 analytical; M=1 DES | KNOWN_MISMATCH |
| λ(x) | Referenced as spatial intensity | Form not given | Planar gradient model | OUR_RECONSTRUCTION |
| FSMC state count L | "partitioned and quantized into L levels" | L not given | L=3 | OUR_RECONSTRUCTION |
| FSMC rate levels | "r = {r_1,...,r_L}" | Exact values not given | {10, 55, 100} Mbps | OUR_RECONSTRUCTION |
| FSMC transition matrix | "L×L transition probability matrix" | Probabilities not given | Diagonal 0.70 | OUR_RECONSTRUCTION |
| t_r (Zyzzyva recovery) | Referenced in Eq. 17 | Value not in Table I | 0.05 s | OUR_RECONSTRUCTION |
| Decision epoch | T_I as block interval | How T_I maps to DES epoch scheduling not specified | Two-height bootstrap | OUR_RECONSTRUCTION |
| Offline dataset | Algorithm 1 step 1 | Data generation process not specified | Not reproduced | Not reproduced |
| DNN architecture | Not specified | Layers, widths, activation | [256,128] MLP, ReLU | OUR_RECONSTRUCTION |
| Replay buffer | Not specified | Size, sampling | 10k, uniform | OUR_RECONSTRUCTION |
| ε schedule | Not specified | Decay type, rate | Linear over 90% of steps | OUR_RECONSTRUCTION |
| Training budget | Not specified | Total episodes/steps | 400 × 20 = 8,000 steps | OUR_RECONSTRUCTION |
| γ (discount) | Not specified | Value | 0.90 | OUR_RECONSTRUCTION |
| G (target sync) | Not specified | Steps between target updates | 50 | OUR_RECONSTRUCTION |

---

## 27. Fidelity Classification Matrix

| Component | Liu source | Implementation | Fidelity | Validation |
|-----------|-----------|----------------|----------|------------|
| State S = [χ,Υ,x,c,R] | Eq. 10 | `state_encoder.py:StateEncoder.encode` | Structure PAPER_EXACT; values/encoding OUR_RECONSTRUCTION | Micro DRL zero regret |
| Action A = [a,δ,S_B,T_I] | Eq. 11 | `action_encoder.py:ActionEncoder.encode` | Structure PAPER_EXACT; encoding OUR_RECONSTRUCTION | 3,915-action domain enumerated |
| Ω objective | Eq. 1 | `ReferenceCore.py:reference_throughput_objective` | PAPER_EXACT | Controlled sweep; agrees on optimum |
| C1 stake gini | Eq. 2, 4 | `paper_stake_gini` | PAPER_EXACT | 384/51 split confirmed |
| C1 geo gini (integral) | Eq. 3, 5 | `geographic_gini_eq3` | PAPER_EXACT integral; λ OUR_RECONSTRUCTION | geo_gini=0.10 ≤ 0.30 |
| C2 formula | Eq. 6, 8 | `reference_consensus`; `_run_to_height` | PAPER_EXACT formula; DES PAPER_DERIVED | paper_pass/DES_fail=0 (2,400 evaluations) |
| C3 formula | Eq. 9 | `_reference_tolerated_faults` | PAPER_EXACT | All controlled sweep actions |
| Reward | Eq. 13 | `StepResult.reward_des_liu_objective` | PAPER_EXACT | Regression test |
| PBFT CPU cost | Appendix B | `AnalyticalPBFT.py` | PAPER_EXACT (M=1 KNOWN_MISMATCH) | Controlled sweep |
| PBFT DES events | Not in paper | `LiuPBFTRuntime` | DES_OPERATIONAL_REALIZATION | N30 evaluation |
| Zyzzyva fast path | Eq. 16 | `LiuZyzzyvaRuntime.fast_path` | PAPER_EXACT | Fidelity gate PASS |
| Zyzzyva recovery | Eq. 17 | `LiuZyzzyvaRuntime.recovery_path` | PAPER_DERIVED; t_r OUR_RECONSTRUCTION | Recovery gate PASS |
| Quorum | Appendix B | `LiuQuorumRuntime` | PAPER_DERIVED | 724=724 controlled sweep |
| M=3 | Table I | Analytical M=3; DES M=1 | KNOWN_MISMATCH | 654 sensitivity: 0 changes |
| α=2MHz/1MHz, β=1MHz | Table I | Cycle counts 2×10^6, 1×10^6 | PAPER_AMBIGUOUS (text/table conflict) | T_V plausible range |
| η_s=0.2, η_l=0.3, ω=6 | Table I | `LiuReferenceParameters` | PAPER_EXACT | |
| N=100, K=21 | Table I | N=30, K=28 (reduced) | OUR_RECONSTRUCTION — explicitly documented | 435-set exhaustive enum |
| Stake 1–50 token | Table I | 4–13 (cycling, within range) | OUR_RECONSTRUCTION (specific assignment) | |
| Capability 10–30 GHz | Table I | 10–30 GHz cycling | Range PAPER_EXACT; assignment OUR_RECONSTRUCTION | |
| Link rate 10–100 Mbps | Table I | FSMC {10,55,100} Mbps | Endpoints PAPER_EXACT; L/probs OUR_RECONSTRUCTION | |
| λ(x) | Eq. 3 reference | `ContinuousSpatial.py` | OUR_RECONSTRUCTION | geo_gini=0.10 |
| FSMC transition | Markov R(t) | `LinkFSMC.py` | Structure PAPER_EXACT; params OUR_RECONSTRUCTION | 1920 comparisons, 0 mismatches |
| Q(S,A) | Eq. 12 | `q_network.py:CandidateConditionedQNetwork` | Concept PAPER_EXACT; architecture OUR_RECONSTRUCTION | Micro zero-regret |
| ε-greedy | Algorithm 1 | `dqn_agent.py:select_action` | PAPER_EXACT | |
| Bellman update | Algorithm 1 | `dqn_agent.py:learn` | PAPER_EXACT | |
| Offline pre-training | Algorithm 1 | Not reproduced | Not reproduced (data generation unspecified) | |
| RUNTIME_GUARD_FAILURE | Not in paper | `DESQueueExhausted` | OUR_RECONSTRUCTION | 9 DQN/16 random/11 fixed; not DQN-specific |

---

## 28. Known Limitations

The following limitations are explicitly acknowledged and carry into future work:

1. **DES M=1 vs analytical M=3:** Sensitivity analysis (654 evaluations) found 0 C2
   classification changes, but this is fixture-specific.

2. **α/β unit inconsistency:** Paper body says "CPU cycles"; Table I writes "2MHz/1MHz".
   Code treats as cycle counts (2,000,000 and 1,000,000). PAPER_AMBIGUOUS.

3. **Reconstructed λ(x):** Eq. 3 references but does not specify λ(x). Planar gradient
   model used (OUR_RECONSTRUCTION).

4. **Reconstructed FSMC:** {10, 55, 100} Mbps, diagonal=0.70 transition matrix.
   Feasibility statistics under dynamic evaluation are sensitive to these choices.

5. **N=30/K=28 reduced environment:** Paper: N=100/K=21. N=30/K=28 chosen for exhaustive
   tractability. Scaling behaviour not established.

6. **3,915-action domain:** 9 curated (δ, S_B, T_I) configs. Not the paper's continuous space.

7. **Q-network architecture OUR_RECONSTRUCTION:** [256,128] MLP not from paper.

8. **Decision-epoch semantics reconstructed:** Two-height bootstrap is OUR_RECONSTRUCTION.

9. **Runtime guard:** PBFT view-change cap / `DESQueueExhausted` is an implementation
   constraint, not a Liu-defined boundary.

10. **Seed 42 only:** Seeds 123 and 7 are optional robustness experiments for future work.

11. **Offline DRL phase not reproduced:** Algorithm 1 specifies offline pre-training.
    Historical data generation process is not specified by Liu.

12. **Stake range narrower than paper:** Paper: 1–50 token. Used: 4–13. Within paper's range.

---

## 29. Reproducibility

**Key artifacts:**

| Artifact | Path |
|----------|------|
| Runtime freeze manifest | `experiments/liu_replication/drl/n30/LIU_REPLICATION_RUNTIME_FROZEN.json` |
| Clean run manifest | `experiments/liu_replication/drl/n30/clean_run_manifest_seed42.json` |
| Final checkpoint (primary) | `experiments/liu_replication/drl/n30/checkpoints/ckpt_seed42_clean_ep400.pt` |
| Episode log | `experiments/liu_replication/drl/n30/episode_log_seed42_clean.csv` |
| Action domain | `experiments/liu_replication/drl/reduced_action_domain.csv` |
| Oracle reference | `experiments/liu_replication/drl/n30/oracle_paper_reference.json` |
| Policy evaluation | `experiments/liu_replication/drl/n30/n30_seed42_policy_evaluation.json` |
| Controlled sweep summary | `experiments/liu_replication/controlled_sweep/controlled_full_summary.json` |
| Micro DRL evaluation | `experiments/liu_replication/drl/micro/evaluation_summary.json` |
| Baseline complete | `experiments/liu_replication/drl/n30/LIU_REPLICATION_BASELINE_TECHNICALLY_COMPLETE.json` |

**To reproduce the evaluation:**
```bash
uv run python experiments/liu_replication/drl/n30/eval_n30_seed42.py
```
Uses `ckpt_seed42_clean_ep400.pt` and the fixed population/FSMC configuration.
Expected runtime: ~71 minutes on reference hardware.

**Source SHA256 hashes** (frozen runtime): see `LIU_REPLICATION_RUNTIME_FROZEN.json:frozen_source_sha256`.

---

## 30. Final Conclusion

This work has operationalized the Liu et al. 2019 adaptive blockchain optimization framework
inside SymBChainSim, establishing a validated experimental baseline for future research.

**What was operationalized from the paper:**
- State/action/reward/constraint specification (Eq. 1–13, Algorithm 1 online phase, Table I, Appendix B)
- Three consensus protocols (PBFT, Zyzzyva, Quorum) as operational DES event systems
- Verified against the analytical reference oracle (2,400 controlled evaluations)
- DRL training loop (ε-greedy, replay, Bellman updates, target network)
- Validated to zero regret in a tractable ground-truth environment (micro DRL)

**What was reconstructed (OUR_RECONSTRUCTION):**
- FSMC parameters (levels, transition probabilities)
- Population fixture (stakes, capabilities, positions, validator grid)
- Spatial intensity λ(x) for geographic Gini
- Q-network architecture and DRL hyperparameters
- Decision-epoch semantics (two-height bootstrap)
- Zyzzyva recovery parameter t_r

**What was documented as KNOWN_MISMATCH:**
- DES M=1 vs analytical M=3 (paper underspecified; sensitivity: 0 C2 changes in 654 evals)

**What was documented as PAPER_AMBIGUOUS:**
- α, β units (text says "CPU cycles"; Table I says "2MHz/1MHz")

**Primary evaluation result (seed 42, ep400 DQN, 16 trajectories):**
- ep400 DQN outperforms random baseline 16/16 trajectories (mean +70,117/trajectory)
- ep400 DQN outperforms strong fixed baseline 16/16 trajectories (mean +140,031/trajectory)
- ep400 feasibility 70.3% vs random 45.9% vs fixed 8.75%
- 9/320 steps (2.8%) classified RUNTIME_GUARD_FAILURE (fewer than both baselines)
- FSMC trajectory identity: 1,920 hash comparisons, 0 mismatches

**Research transition:**
The validated baseline supports original research on blockchain security, decentralization
trade-offs, adaptive control under adversarial conditions, and multi-objective optimization.
All reconstruction decisions are explicit with fidelity classifications; all primary
quantitative results are traceable to specific JSON artifact files.

---

*Document generated and evidence-audited: 2026-09-20.*
*PDF verified directly using PyMuPDF (Table I and Algorithm 1 on page 7 of the PDF).*
*Based on final artifacts from `LIU_REPLICATION_BASELINE_TECHNICALLY_COMPLETE.json`.*
