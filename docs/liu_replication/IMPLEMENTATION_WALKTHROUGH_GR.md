# SymBChainSim — Αναλυτική Περιγραφή Υλοποίησης

> **Γλώσσα:** Ελληνική | **Κοινό:** Τελειόφοιτος / Εξεταστική επιτροπή  
> **Αναφορά:** Liu et al., "Adaptive Blockchain Optimization via Deep Reinforcement Learning", IEEE TII 2019  
> **Περιβάλλον:** Python 3.14, PyTorch 2.14, N=30 κόμβοι, K=28 validators

---

## Περιεχόμενα

1. [Χάρτης έργου](#1-χάρτης-έργου)
2. [DES — Πυρήνας προσομοίωσης διακριτών γεγονότων](#2-des--πυρήνας-προσομοίωσης-διακριτών-γεγονότων)
3. [Μοντέλο δικτύου και καθυστερήσεις](#3-μοντέλο-δικτύου-και-καθυστερήσεις)
4. [FSMC — Δυναμικό κανάλι σύνδεσης](#4-fsmc--δυναμικό-κανάλι-σύνδεσης)
5. [Κατάσταση S(t) — Κωδικοποίηση](#5-κατάσταση-st--κωδικοποίηση)
6. [Ενέργεια A(t) — Κωδικοποίηση και πεδίο ενεργειών](#6-ενέργεια-at--κωδικοποίηση-και-πεδίο-ενεργειών)
7. [Πρωτόκολλα συναίνεσης — PBFT, Zyzzyva, Quorum](#7-πρωτόκολλα-συναίνεσης--pbft-zyzzyva-quorum)
8. [Εκτέλεση μιας ενέργειας στο DES](#8-εκτέλεση-μιας-ενέργειας-στο-des)
9. [Δείκτης C1 — Αποκέντρωση](#9-δείκτης-c1--αποκέντρωση)
10. [Δείκτης C2 — Τελεσιδικία (Finality)](#10-δείκτης-c2--τελεσιδικία-finality)
11. [Δείκτης C3 — Ανοχή σφαλμάτων](#11-δείκτης-c3--ανοχή-σφαλμάτων)
12. [Ανταμοιβή Ω — Liu Throughput](#12-ανταμοιβή-ω--liu-throughput)
13. [DQN Αρχιτεκτονική](#13-dqn-αρχιτεκτονική)
14. [DQN Agent — Επιλογή ενέργειας και εκπαίδευση](#14-dqn-agent--επιλογή-ενέργειας-και-εκπαίδευση)
15. [Ρυθμίσεις εκπαίδευσης (DQNConfig)](#15-ρυθμίσεις-εκπαίδευσης-dqnconfig)
16. [Σενάριο εκπαίδευσης — train_n30.py](#16-σενάριο-εκπαίδευσης--train_n30py)
17. [Σενάριο αξιολόγησης — eval_n30_seed42.py](#17-σενάριο-αξιολόγησης--eval_n30_seed42py)
18. [Demo μιας απόφασης — demo_single_decision.py](#18-demo-μιας-απόφασης--demo_single_decisionpy)
19. [Οδηγός εκτέλεσης](#19-οδηγός-εκτέλεσης)
20. [Αφήγηση end-to-end](#20-αφήγηση-end-to-end)

---

## 1. Χάρτης έργου

```
SymBChainSim/
├── src/Simulator/                 ← Simulator packages (installed via pip install -e .)
│   ├── Engine/                    ← DES πυρήνας
│   │   ├── Simulation.py          ← Κεντρική κλάση Simulation (ρολόι + ουρά)
│   │   ├── EventQueue.py          ← PrioQueue / Queue (heapq)
│   │   └── Event.py               ← Event, MessageEvent, SystemEvent
│   ├── Chain/
│   │   ├── Network.py             ← Καθυστέρηση μετάδοσης μηνυμάτων
│   │   └── Consensus/LiuRuntime/  ← Υλοποίηση PBFT / Zyzzyva / Quorum
│   │       ├── Common/
│   │       │   ├── SingleActionRuntime.py   ← run_single_action()
│   │       │   └── RuntimeEvaluation.py     ← Αξιολόγηση C1/C2/C3 + ανταμοιβή
│   │       ├── PBFT/              ← LIU_PBFT
│   │       ├── Zyzzyva/           ← LIU_ZYZZYVA
│   │       └── Quorum/            ← LIU_QUORUM
│   ├── Liu/
│   │   ├── ReferenceCore.py       ← paper_stake_gini(), GeographicGiniConfig
│   │   ├── LinkFSMC.py            ← LinkFSMCState, LinkRateLevels, LinkTransitionMatrix
│   │   └── LinkState.py           ← LinkStateMatrix
│   └── Utils/
│       ├── ComputationalDelay.py  ← scale(base_delay, ghz)
│       └── DecentralizationMetrics.py  ← canonical_pairwise_gini()
│
└── experiments/liu_replication/
    ├── dynamic_epoch/
    │   └── liu_dynamic_env.py     ← LiuDynamicEpochEnv (RL environment)
    ├── drl/
    │   ├── state_encoder.py       ← StateEncoder (dim=991 για N=30)
    │   ├── action_encoder.py      ← ActionEncoder (dim=35 για N=30)
    │   ├── q_network.py           ← QNetwork (MLP)
    │   ├── dqn_agent.py           ← DQNAgent (ε-greedy, replay, Bellman)
    │   ├── config.py              ← DQNConfig
    │   └── n30/
    │       ├── train_n30.py       ← Κύριο σενάριο εκπαίδευσης
    │       ├── eval_n30_seed42.py ← Αξιολόγηση πολιτικής
    │       └── checkpoints/
    │           └── ckpt_seed42_clean_ep400.pt  ← Τελικό checkpoint
    ├── reduced_action_domain.csv  ← 3915 ενέργειες A(t)
    └── demo_single_decision.py    ← Demo για παρουσίαση
```

Το πακέτο εγκαθίσταται με `pip install -e . --no-deps`, το οποίο δημιουργεί ένα αρχείο `.pth` στο `site-packages` που βάζει τον κατάλογο `src/Simulator` στο `sys.path`. Έτσι οι εισαγωγές `from Liu.LinkFSMC import ...`, `from Chain.Consensus...` κ.λπ. λειτουργούν σε οποιοδήποτε script ή IDE χωρίς χειροκίνητη ρύθμιση.

---

## 2. DES — Πυρήνας προσομοίωσης διακριτών γεγονότων

Το SymBChainSim χρησιμοποιεί Discrete-Event Simulation (DES) για να μοντελοποιεί τη χρονική εξέλιξη ενός blockchain δικτύου. Η αρχή είναι απλή: αντί να "τρέχει" ο χρόνος συνεχώς, η προσομοίωση μεταπηδά από γεγονός σε γεγονός.

### 2.1 Δομή γεγονότων — `Engine/Event.py`

```python
# Engine/Event.py
class Event:
    time: float       # Εικονική ώρα εκτέλεσης (δευτερόλεπτα)
    payload: Any      # Ωφέλιμο φορτίο (π.χ. μήνυμα πρωτοκόλλου)
    handler: Callable # Συνάρτηση που καλείται όταν εκτελεστεί το γεγονός
    creator: Any      # Κόμβος που δημιούργησε το γεγονός
    actor: Any        # Κόμβος που εκτελεί το γεγονός

class MessageEvent(Event):
    receiver: Any         # Παραλήπτης μηνύματος
    forwarded_by: Any     # Κόμβος αναμετάδοσης (για gossip)

class SystemEvent(Event):
    pass                  # Εσωτερικά γεγονότα (timeouts, timers)
```

### 2.2 Ουρά γεγονότων — `Engine/EventQueue.py`

```python
# Engine/EventQueue.py
class PrioQueue:          # Υλοποίηση με heapq (min-heap)
    def add_event(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.time, self._counter, event))

class Queue:              # Wrapper γύρω από PrioQueue
    pass
```

Η ουρά είναι min-heap πάνω στο `event.time`, άρα το επόμενο γεγονός είναι πάντα αυτό με τον μικρότερο χρόνο. Αυτό εγγυάται ότι τα γεγονότα εκτελούνται σε αιτιώδη σειρά.

### 2.3 Κύριος βρόχος — `Engine/Simulation.py`

```python
# Engine/Simulation.py
class Simulation:
    clock: float = 0.0    # Εικονικό ρολόι
    q: Queue              # Ουρά γεγονότων

    def sim_next_event(self) -> Event:
        event = self.q.pop()       # Εξάγει το γεγονός με τον μικρότερο χρόνο
        self.clock = event.time    # Προωθεί το ρολόι
        event.handler(event)       # Εκτελεί τον χειριστή
        return event
```

Κάθε φορά που καλείται η `sim_next_event()`:
1. Εξάγεται το γεγονός με τον μικρότερο `time` από τον σωρό.
2. Το εικονικό ρολόι `self.clock` ανανεώνεται στον χρόνο αυτού του γεγονότος.
3. Καλείται ο `handler` του γεγονότος, ο οποίος μπορεί να εισάγει νέα γεγονότα στην ουρά.

Αυτός ο βρόχος συνεχίζεται μέχρι η ουρά να αδειάσει ή να πληρωθεί κάποια συνθήκη τερματισμού (π.χ. επίτευξη τελεσιδικίας μπλοκ ή λήξη προθεσμίας C2).

---

## 3. Μοντέλο δικτύου και καθυστερήσεις

### 3.1 Καθυστέρηση μετάδοσης — `Chain/Network.py`

Κάθε μήνυμα που στέλνεται από κόμβο `s` σε κόμβο `r` υπόκειται σε:

```
T_msg = T_transmission + T_latency + T_queueing + T_processing
```

- **T_transmission**: `message_size / bandwidth`, όπου `bandwidth = min(rate_s→r, rate_r→s)` (bottleneck)
- **T_latency**: καθυστέρηση διάδοσης βάσει γεωγραφικής απόστασης (km)
- **T_queueing**: μοντέλο ουράς (queue congestion)
- **T_processing**: υπολογιστική καθυστέρηση παραλήπτη

Η γεωγραφική απόσταση υπολογίζεται από τον πίνακα `positions_km` (N×2 συντεταγμένες) και χρησιμοποιείται στο `GeographicGiniConfig` για τον υπολογισμό του Gini αποκέντρωσης θέσεων.

### 3.2 Υπολογιστική καθυστέρηση — `Utils/ComputationalDelay.py`

```python
# Utils/ComputationalDelay.py
REFERENCE_CAPABILITY_GHZ = 20.0

def scale(base_delay: float, ghz: float) -> float:
    return base_delay * 20.0 / ghz
```

Ένας κόμβος με δυνατότητα `c` GHz εκτελεί μια λειτουργία αναφοράς (calibrated σε 20 GHz) σε χρόνο `base_delay * 20 / c`. Έτσι ισχυρότεροι κόμβοι (υψηλό GHz) επεξεργάζονται συναλλαγές και μηνύματα γρηγορότερα.

---

## 4. FSMC — Δυναμικό κανάλι σύνδεσης

### 4.1 Θεωρία

Το Finite-State Markov Chain (FSMC) μοντελοποιεί τη χρονική μεταβλητότητα του εύρους ζώνης κάθε κατευθυνόμενης σύνδεσης i→j. Σε κάθε epoch (χρονικό βήμα), κάθε σύνδεση βρίσκεται σε ένα από L=3 επίπεδα ρυθμού:

```
FSMC_LEVELS = (10 Mbps,  55 Mbps,  100 Mbps)
              ΚΑΚΟ       ΜΕΣΑΙΟ    ΚΑΛΟ
```

Η μετάβαση από επίπεδο i σε επίπεδο j ακολουθεί πίνακα μεταβάσεων με διαγώνιο πιθανότητα 0.70:

```
P = [[0.70, 0.15, 0.15],   ← επίπεδο 0 (10 Mbps)
     [0.15, 0.70, 0.15],   ← επίπεδο 1 (55 Mbps)
     [0.15, 0.15, 0.70]]   ← επίπεδο 2 (100 Mbps)
```

Δηλαδή: με πιθανότητα 70% η σύνδεση παραμένει στο ίδιο επίπεδο, και με 30% μεταβαίνει σε ένα από τα άλλα δύο.

### 4.2 Υλοποίηση — `Liu/LinkFSMC.py`

```python
# Liu/LinkFSMC.py

@dataclass(frozen=True, slots=True)
class LinkRateLevels(CanonicalSerializable):
    rates_mbps: tuple[float, ...]    # (10.0, 55.0, 100.0) για Liu

@dataclass(frozen=True, slots=True)
class LinkTransitionMatrix(CanonicalSerializable):
    probabilities: tuple[tuple[float, ...], ...]  # LxL πίνακας

    def sample_next_level(self, current_level: int, rng: RandomSource) -> int:
        # Δειγματοληψία CDF: επιλέγει επίπεδο βάσει συσσωρευτικής πιθανότητας
        draw = rng.random()
        cumulative = 0.0
        for next_level, probability in enumerate(self.probabilities[current_level]):
            cumulative += probability
            if draw < cumulative:
                return next_level
        return self.level_count - 1

@dataclass(frozen=True, slots=True)
class LinkFSMCState(CanonicalSerializable):
    rate_levels: LinkRateLevels
    transition_tensor: LinkTransitionTensor   # N×N πίνακες μεταβάσεων
    current_links: LinkStateMatrix            # Τρέχοντες ρυθμοί NxN

    def advance(self, rng: RandomSource) -> "LinkFSMCState":
        return self.with_current_links(self.transition(rng))
```

Σημαντικό: το `LinkTransitionTensor.shared()` δημιουργεί έναν κοινό πίνακα μεταβάσεων για όλες τις N(N-1) κατευθυνόμενες συνδέσεις — αυτό αντιστοιχεί στο μοντέλο του Liu et al.

### 4.3 Ενσωμάτωση στο RL Environment — `dynamic_epoch/liu_dynamic_env.py`

```python
# dynamic_epoch/liu_dynamic_env.py
class LiuDynamicEpochEnv:
    def step(self, action: dict) -> StepResult:
        result = run_single_action(...)   # Εκτελεί DES για αυτή την ενέργεια
        self._fsmc = self._fsmc.advance(self._rng)  # Μεταβαίνει σε νέο epoch
        return result
```

Μετά από κάθε βήμα-ενέργεια, ο πίνακας συνδέσεων μεταβαίνει στοχαστικά. Η κατάσταση του επόμενου βήματος περιλαμβάνει τους νέους ρυθμούς. Αυτό δημιουργεί ένα μη-στατικό (non-stationary) MDP, ακριβώς όπως στο Liu et al.

---

## 5. Κατάσταση S(t) — Κωδικοποίηση

### 5.1 Ορισμός κατάστασης από Liu et al.

Η κατάσταση `S(t)` αντιπροσωπεύει το πλήρες πληροφοριακό πλαίσιο στον χρόνο `t`:

```
S(t) = [χ, Υ, x, c, R]
```

| Συστατικό | Σύμβολο | Τύπος | Περιγραφή |
|-----------|---------|-------|-----------|
| Μέγεθος συναλλαγής | χ | scalar | bytes ανά συναλλαγή |
| Στόχοι (stakes) | Υ | N-διάνυσμα | tokens ανά κόμβο |
| Θέσεις | x | N×2 πίνακας | συντεταγμένες km |
| Δυνατότητες | c | N-διάνυσμα | GHz ανά κόμβο |
| Πίνακας συνδέσεων | R | N×N πίνακας | ρυθμός Mbps i→j |

### 5.2 Υλοποίηση — `drl/state_encoder.py`

```python
# drl/state_encoder.py
class StateEncoder:
    # dim = 1 + N + 2N + N + N*(N-1) = 1 + N*(N+3)
    # Για N=30: 1 + 30*33 = 991

    def encode(self, state_dict: dict) -> np.ndarray:
        # [0]        chi_bytes / CHI_MAX          ← κανονικοποίηση
        # [1..N]     stakes / STAKE_MAX
        # [N+1..3N]  positions_x/y / POS_MAX      ← 2N features
        # [3N+1..4N] capabilities / CAP_MAX
        # [4N+1..]   OffDiag(R) / LINK_MAX        ← N*(N-1) features
```

Για N=30: `dim = 1 + 30*(30+3) = 991`.

Η κανονικοποίηση χρησιμοποιεί σταθερές από το `config.py`:
- `CHI_MAX`: μέγιστο μέγεθος συναλλαγής
- `STAKE_MAX`: μέγιστα stakes
- `POS_MAX`: μέγιστη απόσταση km
- `CAP_MAX`: μέγιστο GHz
- `LINK_MAX = 100.0` Mbps

Ο off-diagonal πίνακας R (N×N χωρίς διαγώνιο) δίνει N*(N-1) = 870 χαρακτηριστικά μόνο για τον ρυθμό σύνδεσης.

---

## 6. Ενέργεια A(t) — Κωδικοποίηση και πεδίο ενεργειών

### 6.1 Ορισμός ενέργειας από Liu et al.

```
A(t) = [a, δ, S_B, T_I]
```

| Συστατικό | Σύμβολο | Επιλογές | Περιγραφή |
|-----------|---------|----------|-----------|
| Σύνολο validators | a | C(30,28) = 435 | K=28 από N=30 |
| Πρωτόκολλο | δ | {PBFT, Zyzzyva, Quorum} | 3 επιλογές |
| Μέγεθος μπλοκ | S_B | {1, 2, 4} MB | 3 επιλογές |
| Διάστημα μπλοκ | T_I | {2, 4, 8} s | 3 επιλογές |

Το συνολικό πεδίο ενεργειών: **3915 = 435 × 3 × 3** (πρωτόκολλο × S_B × T_I = 9 συνδυασμοί ανά σύνολο validators).

Όλες οι ενέργειες είναι προ-υπολογισμένες σε: `experiments/liu_replication/drl/reduced_action_domain.csv`

### 6.2 Κωδικοποίηση ενέργειας — `drl/action_encoder.py`

```python
# drl/action_encoder.py
class ActionEncoder:
    # dim = N + 5
    # Για N=30: 35

    def encode(self, action: dict) -> np.ndarray:
        # [0..N-1]   validator_mask  ← binary N-vector (1 αν κόμβος είναι validator)
        # [N..N+2]   protocol        ← one-hot (PBFT=100, Zyzzyva=010, Quorum=001)
        # [N+3]      S_B_norm        ← block_size_mb / S_B_MAX
        # [N+4]      T_I_norm        ← block_interval_s / T_I_MAX
```

Για N=30: `dim = 30 + 3 + 1 + 1 = 35`.

Η `encode_batch()` κωδικοποιεί ταυτόχρονα και τις 3915 ενέργειες σε έναν πίνακα `(3915, 35)` — αυτός ο πίνακας φορτώνεται μία φορά στην αρχή και χρησιμοποιείται σε κάθε βήμα για την αξιολόγηση όλων των Q-τιμών.

---

## 7. Πρωτόκολλα συναίνεσης — PBFT, Zyzzyva, Quorum

Τα τρία πρωτόκολλα υλοποιούνται στο `src/Simulator/Chain/Consensus/LiuRuntime/`.

### 7.1 LIU_PBFT

Το Practical Byzantine Fault Tolerance απαιτεί 3f+1 validators για ανοχή f σφαλμάτων. Η διαδικασία έχει τρεις φάσεις:
- **Pre-prepare**: Ο primary στέλνει την πρόταση μπλοκ σε όλους τους validators.
- **Prepare**: Κάθε validator εκπέμπει μήνυμα prepare· μετά από 2f+1 prepares, περνά στο commit.
- **Commit**: Κάθε validator εκπέμπει commit· μετά από 2f+1 commits, το μπλοκ οριστικοποιείται.

Σε περίπτωση timeout, ενεργοποιείται view-change. Η υλοποίηση ορίζει ένα cap στον αριθμό των view-changes (το `DESQueueExhausted` guard) για να αποφύγει άπειρες επαναλήψεις.

### 7.2 LIU_ZYZZYVA

Η Zyzzyva επιχειρεί ταχεία διαδρομή (fast path) σε 1.5 round-trips αντί για 3:
- **Speculative execution**: Ο primary στέλνει πρόταση· όλοι οι validators εκτελούν και απαντούν κατευθείαν στον client.
- **Fast path** (f=0): Αν ο client λάβει K=2f+1 ταυτόσημες απαντήσεις, το μπλοκ οριστικοποιείται χωρίς επιπλέον φάσεις.
- **Recovery path** (f>0): Αν η ταχεία διαδρομή αποτύχει, ακολουθεί μια επιπλέον φάση commit.

### 7.3 LIU_QUORUM

Το Quorum (Stellar-style) απαιτεί ομόφωνη συμφωνία — δεν ανέχεται **κανένα** σφάλμα (f=0). Είναι το απλούστερο πρωτόκολλο, αλλά χωρίς ανοχή σφαλμάτων:

```python
# RuntimeEvaluation.py
def tolerated_faults(protocol, K) -> int:
    if protocol == LIU_QUORUM:
        return 0
    else:  # PBFT, Zyzzyva
        return (K - 1) // 3
```

Για K=28: PBFT/Zyzzyva ανέχονται f=9 σφάλματα, ενώ το Quorum ανέχεται f=0.

---

## 8. Εκτέλεση μιας ενέργειας στο DES

### 8.1 Σημείο εισόδου — `Chain/Consensus/LiuRuntime/Common/SingleActionRuntime.py`

```python
# SingleActionRuntime.py

def run_single_action(
    action: LiuAction,
    network: ...,
    sim_time_limit: float,
    omega: float = 6.0,
    ...
) -> ProtocolFinalityRecord:
    ...
    result = _run_to_height(sim, target_height, sim_time_limit)
    ...
```

Η `run_single_action()` είναι η βασική συνάρτηση που:
1. Δημιουργεί μια νέα `Simulation` για αυτή την ενέργεια.
2. Εισάγει γεγονότα εκκίνησης (request για νέο μπλοκ στον primary).
3. Τρέχει τον βρόχο DES μέχρι:
   - Επίτευξη τελεσιδικίας μπλοκ (`ProtocolFinalityRecord`), ή
   - Λήξη της προθεσμίας C2 (`C2DeadlineExceeded`), ή
   - Εξάντληση της ουράς γεγονότων (`DESQueueExhausted`).

### 8.2 Μέτρηση χρόνου τελεσιδικίας

```python
# SingleActionRuntime.py

# Η προθεσμία C2 υπολογίζεται ως:
c2_deadline_s = boundary_time + omega * T_I

# Ο χρόνος τελεσιδικίας DES:
t_c_des = finality_time - request_sent_at
```

Η `finality_time` είναι η στιγμή που ο **πρώτος** validator κάνει commit — όχι όλοι οι N κόμβοι. Αυτό αντικατοπτρίζει τον ορισμό τελεσιδικίας του Liu et al.: το μπλοκ θεωρείται οριστικοποιημένο μόλις ένα quorum commits.

---

## 9. Δείκτης C1 — Αποκέντρωση

### 9.1 Ορισμός Liu et al.

Ο C1 ελέγχει ότι η επιλογή validators δεν συγκεντρώνει υπερβολικά ισχύ:

```
C1: G(Υ_selected) ≤ η_s = 0.2  AND  G(x_selected) ≤ η_l = 0.3
```

Όπου G() είναι ο συντελεστής Gini. Η πρώτη συνθήκη ελέγχει την ισότητα stakes, η δεύτερη την γεωγραφική κατανομή.

### 9.2 Υλοποίηση — `Utils/DecentralizationMetrics.py`

```python
# Utils/DecentralizationMetrics.py

def canonical_pairwise_gini(values: list[float]) -> float:
    """
    Gini = Σ_i Σ_j |x_i - x_j| / (2 * n * Σ_i x_i)
    """
    n = len(values)
    total = sum(values)
    if total == 0:
        return 0.0
    numerator = sum(abs(xi - xj) for xi in values for xj in values)
    return numerator / (2 * n * total)
```

Αυτή η "pairwise" φόρμουλα είναι η τυπική ορισμός Gini που χρησιμοποιείται στο paper. Για ομοιόμορφη κατανομή G=0, για πλήρη συγκέντρωση G→1.

---

## 10. Δείκτης C2 — Τελεσιδικία (Finality)

### 10.1 Ορισμός Liu et al.

```
C2: T_F = T_I + T_C ≤ ω · T_I,   ω = 6.0
```

Όπου:
- `T_I`: block interval (διάστημα μπλοκ, σε δευτερόλεπτα)
- `T_C`: χρόνος consensus (πόσο διαρκεί η διαδικασία συναίνεσης)
- `T_F = T_I + T_C`: συνολικός χρόνος τελεσιδικίας

Απλοποιώντας: `T_I + T_C ≤ 6·T_I` → `T_C ≤ 5·T_I`.

### 10.2 Αποτυχία C2

Υπάρχουν δύο τρόποι αποτυχίας C2 στην υλοποίηση:

| Αιτία | Τύπος | Σημασία |
|-------|-------|---------|
| `C2DeadlineExceeded` | Λήξη προθεσμίας DES | Αυθεντική αποτυχία C2 από Liu |
| `DESQueueExhausted` | Εξάντληση ουράς | Guard υλοποίησης (PBFT view-change cap) |

Η `C2DeadlineExceeded` είναι η κανονική: ο χρόνος `t_c_des > ω·T_I`. Η `DESQueueExhausted` είναι ένα ασφαλές τέλος όταν το PBFT κολλήσει σε view-changes χωρίς πρόοδο — σε αυτή την περίπτωση η ανταμοιβή είναι επίσης 0 (αποτυχία C2).

---

## 11. Δείκτης C3 — Ανοχή σφαλμάτων

### 11.1 Ορισμός Liu et al.

```
C3: f ≤ F^δ
```

Όπου `f` είναι ο αριθμός πραγματικών σφαλμάτων (faulty κόμβοι) και `F^δ` η ικανότητα ανοχής του πρωτοκόλλου δ:

```python
# RuntimeEvaluation.py
def tolerated_faults(protocol, K) -> int:
    if protocol == LIU_QUORUM:
        return 0                  # Quorum: καθόλου σφάλματα
    else:
        return (K - 1) // 3      # PBFT/Zyzzyva: floor((K-1)/3)
```

Για K=28: `(28-1)//3 = 9` για PBFT/Zyzzyva, `0` για Quorum.

Στην αξιολόγηση, το `f` (faulty_node_ids) καθορίζεται από το περιβάλλον εκπαίδευσης/αξιολόγησης. Κατά την κανονική εκπαίδευση `f=0`, άρα C3 πάντα ικανοποιείται (ακόμα και για Quorum).

---

## 12. Ανταμοιβή Ω — Liu Throughput

### 12.1 Ορισμός Liu et al.

```
Ω(S, A) = floor(S_B · 10^6 / χ) / T_I,    αν C1 ∧ C2 ∧ C3
         = 0,                               αλλιώς
```

Όπου:
- `S_B`: μέγεθος μπλοκ σε MB
- `χ`: μέγεθος συναλλαγής σε bytes
- `T_I`: block interval σε δευτερόλεπτα
- `floor(S_B·10^6/χ)`: αριθμός συναλλαγών ανά μπλοκ (χωρητικότητα)

Η ανταμοιβή είναι ο **throughput** (συναλλαγές ανά δευτερόλεπτο) αν όλοι οι περιορισμοί ικανοποιούνται.

### 12.2 Υλοποίηση — `Chain/Consensus/LiuRuntime/Common/RuntimeEvaluation.py`

```python
# RuntimeEvaluation.py
class LiuRuntimeConstraintEvaluator:
    def evaluate(self, finality_record: ProtocolFinalityRecord) -> StepResult:
        c1 = self._check_c1(...)
        c2 = self._check_c2(t_c_des, omega, T_I)
        c3 = self._check_c3(f, protocol, K)

        if c1 and c2 and c3:
            liu_throughput = math.floor(S_B * 1e6 / chi) / T_I
            reward = liu_throughput
        else:
            reward = 0.0

        return StepResult(reward=reward, des_c1=c1, des_c2=c2, des_c3=c3, ...)
```

### 12.3 Κλιμάκωση ανταμοιβής

Κατά την εκπαίδευση, η ανταμοιβή κανονικοποιείται:

```python
# train_n30.py
REWARD_SCALE = 17_000
reward_norm = reward / REWARD_SCALE
```

Η τιμή 17000 επελέγη εμπειρικά ώστε η κανονικοποιημένη ανταμοιβή να βρίσκεται στο [0, 1]. Για S_B=4 MB, χ=200 bytes, T_I=2s: `floor(4·10^6/200)/2 = 10000` tx/s. Για μικρότερα T_I ή μεγαλύτερα S_B η τιμή ανεβαίνει, άρα 17000 καλύπτει το πεδίο τιμών.

---

## 13. DQN Αρχιτεκτονική

### 13.1 QNetwork — `drl/q_network.py`

```python
# drl/q_network.py
class QNetwork(nn.Module):
    # Αρχιτεκτονική: MLP concat(state_enc, action_enc) → Q(S, A)
    #
    # Είσοδος:  state_dim + action_dim = 991 + 35 = 1026
    # Κρυφές:   hidden_sizes = (256, 128)
    # Έξοδος:   1 scalar Q(S, A)

    def forward(self, state_enc, action_enc):
        x = torch.cat([state_enc, action_enc], dim=-1)
        x = F.relu(self.fc1(x))   # 1026 → 256
        x = F.relu(self.fc2(x))   # 256  → 128
        return self.fc3(x)        # 128  → 1

    def q_values_batch(self, state_enc: Tensor, action_batch: Tensor) -> Tensor:
        # state_enc: (state_dim,) ή (B, state_dim)
        # action_batch: (A, action_dim) για A=3915 ενέργειες
        # Επιστρέφει: (A,) ή (B, A) Q-τιμές
        state_exp = state_enc.unsqueeze(0).expand(A, -1)
        pairs = torch.cat([state_exp, action_batch], dim=-1)
        return self.forward(pairs).squeeze(-1)
```

Για κάθε κατάσταση, το δίκτυο υπολογίζει **ταυτόχρονα** και τις 3915 Q-τιμές με ένα batch forward pass.

### 13.2 Δίκτυο στόχου (Target Network)

Η DQN χρησιμοποιεί δύο πανομοιότυπα δίκτυα:
- **Online network** (`q_net`): εκπαιδεύεται σε κάθε βήμα.
- **Target network** (`target_net`): αντιγράφεται από το online κάθε `G=50` βήματα εκπαίδευσης.

Ο στόχος Bellman υπολογίζεται χρησιμοποιώντας το target network, το οποίο είναι "παγωμένο" και σταθερό για 50 βήματα. Αυτό σταθεροποιεί την εκπαίδευση αποφεύγοντας τη μη-στασιμότητα των στόχων.

---

## 14. DQN Agent — Επιλογή ενέργειας και εκπαίδευση

### 14.1 DQNAgent — `drl/dqn_agent.py`

```python
# drl/dqn_agent.py
class DQNAgent:
    def select_action(self, state_enc: np.ndarray, epsilon: float) -> tuple[int, str]:
        if random.random() < epsilon:
            # Τυχαία εξερεύνηση
            action_idx = random.randrange(len(self._action_tensors))
            return action_idx, "random"
        else:
            # Greedy: argmax Q(S, A) για όλα τα A
            return self.argmax(state_enc), "greedy"

    def argmax(self, state_enc: np.ndarray) -> int:
        # Υπολογίζει Q-τιμές για όλα τα 3915 A ταυτόχρονα
        s_t = torch.from_numpy(state_enc).float()
        q_vals = self.q_net.q_values_batch(s_t, self._action_tensors)
        return int(q_vals.argmax().item())
```

### 14.2 Bellman Update

```python
    def _train_step(self, batch: list[Transition]) -> float:
        # batch: τυχαίο δείγμα από το replay buffer
        s, a, r, s_next, done = unpack(batch)

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net.q_values_batch(s_next, self._action_tensors)
            target = r + self.gamma * q_next.max(dim=-1).values * (1 - done)

        # Online Q(s, a)
        q_pred = self.q_net.q_values_batch(s, self._action_tensors[a])

        # MSE loss
        loss = F.mse_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Συγχρονισμός target network κάθε G=50 βήματα
        if self._opt_steps % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
```

### 14.3 Replay Buffer

Ο agent διατηρεί ένα Experience Replay buffer χωρητικότητας `replay_capacity=10_000` transitions. Σε κάθε βήμα εκπαίδευσης:
1. Αποθηκεύεται η τωρινή εμπειρία (s, a, r, s', done).
2. Τυχαίο δείγμα `batch_size=64` μεταβάσεων χρησιμοποιείται για το gradient update.

Η τυχαία δειγματοληψία σπάει τις χρονικές συσχετίσεις μεταξύ διαδοχικών εμπειριών, σταθεροποιώντας την εκπαίδευση.

---

## 15. Ρυθμίσεις εκπαίδευσης (DQNConfig)

### 15.1 `drl/config.py`

```python
# drl/config.py
class DQNConfig:
    hidden_sizes: tuple = (256, 128)    # Αρχιτεκτονική MLP
    gamma: float = 0.9                   # Παράγοντας έκπτωσης
    lr: float = 5e-4                     # Learning rate (Adam)
    replay_capacity: int = 10_000        # Μέγεθος buffer
    warmup_steps: int = 200              # Βήματα πριν αρχίσει η εκπαίδευση
    batch_size: int = 64                 # Μέγεθος batch
    target_sync_steps: int = 50          # Συχνότητα ανανέωσης target
```

Ο `gamma=0.9` (αντί 0.99) αντικατοπτρίζει ότι κάθε βήμα αντιστοιχεί σε ένα ολόκληρο block interval (2-8 δευτερόλεπτα) — άρα το μέλλον "αποσβένεται" γρηγορότερα σε πραγματικό χρόνο.

---

## 16. Σενάριο εκπαίδευσης — train_n30.py

### 16.1 Παράμετροι εκπαίδευσης

```python
# experiments/liu_replication/drl/n30/train_n30.py
N = 30                      # Αριθμός κόμβων
K = 28                      # Validators
CHI_BYTES = 200             # Bytes ανά συναλλαγή
STEPS_PER_EPISODE = 20      # Βήματα ανά επεισόδιο
TRAIN_EPISODES = 400        # Συνολικά επεισόδια
SEEDS = [42, 123, 7]        # Seeds εκπαίδευσης
REWARD_SCALE = 17_000       # Κανονικοποίηση ανταμοιβής
offered_workload_tx = 100_000  # Συναλλαγές στη mempool
```

### 16.2 Δύο περιβάλλοντα εκπαίδευσης

```python
envs = {
    "GOOD_100Mbps":  make_env(_all_links_rows(100.0), seed=seed),
    "MEDIUM_55Mbps": make_env(_all_links_rows(55.0),  seed=seed),
}
env_cycle = list(envs.items())
```

Τα επεισόδια εναλλάσσονται μεταξύ GOOD (100 Mbps) και MEDIUM (55 Mbps). Το POOR (10 Mbps) εξαιρείται από την εκπαίδευση γιατί σε 10 Mbps **κανένα** από τα 3915 A δεν ικανοποιεί το C2 — άρα δεν υπάρχει gradient signal. Χρησιμοποιείται μόνο για αξιολόγηση.

### 16.3 Μείωση ε (epsilon decay)

```python
# Η ε μειώνεται γραμμικά από 1.0 → 0.05 κατά τη διάρκεια της εκπαίδευσης
epsilon = max(EPS_MIN, EPS_START - step * (EPS_START - EPS_MIN) / EPS_DECAY_STEPS)
```

Στην αρχή (ε=1.0) ο agent εξερευνά τυχαία. Σταδιακά η ε μειώνεται και ο agent εκμεταλλεύεται όλο και περισσότερο τη μαθημένη Q-συνάρτηση.

### 16.4 Checkpoints

```python
# Αποθήκευση checkpoint κάθε 50 επεισόδια
if (ep + 1) % 50 == 0:
    save_checkpoint(agent, ep + 1, global_step, ckpt_dir / f"ckpt_seed42_clean_ep{ep+1}.pt")
```

Το τελικό checkpoint είναι: `checkpoints/ckpt_seed42_clean_ep400.pt`

### 16.5 Καταγραφή

Κάθε βήμα καταγράφεται στο: `drl/n30/step_log_seed42_clean.csv`

Τα πεδία περιλαμβάνουν: `episode`, `global_step`, `action_id`, `protocol`, `block_size_mb`, `block_interval_s`, `epsilon`, `des_c1/c2/c3`, `des_t_c_s`, `reward`, `reward_norm`, `link_rate_mean_mbps`, `loss`.

---

## 17. Σενάριο αξιολόγησης — eval_n30_seed42.py

### 17.1 Παράμετροι αξιολόγησης

```python
# experiments/liu_replication/drl/n30/eval_n30_seed42.py
EVAL_SEEDS = [1000, 1001, 1002, 1003]   # 4 seeds αξιολόγησης
EVAL_STEPS = 20                           # 20 βήματα ανά τροχιά
PRIMARY_CKPT  = CKPT_DIR / "ckpt_seed42_clean_ep400.pt"  # Ep400 (τελικό)
DIAG_CKPT     = CKPT_DIR / "ckpt_seed42_clean_ep350.pt"  # Ep350 (διαγνωστικό)
```

### 17.2 Αρχικές καταστάσεις δικτύου

```python
INITIAL_STATES: dict[str, tuple] = {
    "ALL_HIGH_100":    _all_links(100.0),   # Όλες οι συνδέσεις 100 Mbps
    "ALL_MID_55":      _all_links(55.0),    # Όλες οι συνδέσεις 55 Mbps
    "ALL_LOW_10":      _all_links(10.0),    # Όλες οι συνδέσεις 10 Mbps
    "TRAINING_MIXED":  INITIAL_LINK_ROWS,   # Μικτές (από εκπαίδευση)
}
```

### 17.3 Πολιτικές σύγκρισης

Η αξιολόγηση συγκρίνει 4 πολιτικές:
1. **DQN ep400** (ε=0, greedy): Τελική εκπαιδευμένη πολιτική
2. **DQN ep350** (ε=0, greedy): Ενδιάμεσο checkpoint για σύγκριση
3. **Random**: Τυχαία επιλογή ενέργειας
4. **Strong-fixed**: Η καλύτερη σταθερή ενέργεια από το sweep (από `source_sweep_reward`)

Όλες οι πολιτικές μοιράζονται τις ίδιες ακολουθίες FSMC (ίδιοι seeds) για δίκαιη σύγκριση.

### 17.4 Αποτελέσματα

Τα αποτελέσματα αποθηκεύονται σε:
- `n30_seed42_policy_evaluation.json`: Πλήρη δεδομένα ανά βήμα
- `n30_seed42_policy_evaluation.md`: Συνοπτικός πίνακας σύγκρισης

---

## 18. Demo μιας απόφασης — demo_single_decision.py

### 18.1 Σκοπός

Το `experiments/liu_replication/demo_single_decision.py` είναι ένα demo παρουσίασης που δείχνει σε ένα βήμα:
1. Την τρέχουσα κατάσταση S(t)
2. Την απόφαση του εκπαιδευμένου DQN A(t)
3. Την εκτέλεση στο DES
4. Τον έλεγχο C1/C2/C3
5. Την υπολογισμένη ανταμοιβή Ω

### 18.2 Παράμετροι demo

```python
N = 30          # Κόμβοι
K = 28          # Validators
CHI_BYTES = 200 # Bytes/tx
REWARD_SCALE = 17_000
```

### 18.3 Ροή εκτέλεσης

```
1. Φόρτωση ep400 checkpoint (ε=0 greedy)
2. Φόρτωση 3915 ενεργειών από reduced_action_domain.csv
3. Κωδικοποίηση κατάστασης S(t) → state_enc (991 διαστάσεις)
4. DQN argmax → best action A(t) από 3915 επιλογές
5. run_single_action(A(t)) → DES simulation
6. Αξιολόγηση C1/C2/C3 → reward Ω
7. Εκτύπωση αποτελεσμάτων
```

### 18.4 Δείγμα εξόδου

```
═══════════════════════════════ CURRENT STATE  S(t) ═══
  Transaction size χ : 200 B
  Nodes N            : 30
  Validators K       : 28
  Link rate (mean)   : 100.0 Mbps
  Link rate (min)    : 100.0 Mbps
  Link rate (max)    : 100.0 Mbps

═══════════════════════════════ DQN DECISION  A(t)   [Trained DQN policy] ═══
  Validators         : 28 nodes selected
  Protocol δ         : Quorum
  Block size S_B     : 4.0 MB
  Block interval T_I : 2.0 s
  Estimated Q-value  : 0.5882  (normalised by 17,000)

═══════════════════════════════ DES EXECUTION ═══
  C1 (decentralization) : PASS
  C2 (finality)         : PASS
  C3 (fault tolerance)  : PASS
  Finality status       : Finalized within C2 limit
  Consensus time T_C    : 0.042 s  (limit 10.0 s)
  Finality time T_F     : 2.042 s

═══════════════════════════════ LIU REWARD ═══
  Throughput Ω       : 10000.0 tx/s
  Reward (norm.)     : 0.5882
```

### 18.5 Λειτουργία σύγκρισης

```bash
.venv\Scripts\python.exe experiments/liu_replication/demo_single_decision.py --compare
```

Στη λειτουργία `--compare`, εκτελούνται τρεις πολιτικές (DQN, τυχαία, fixed) για την ίδια αρχική κατάσταση και εκτυπώνεται πίνακας σύγκρισης.

---

## 19. Οδηγός εκτέλεσης

### 19.1 Εγκατάσταση

```bash
# Δημιουργία virtual environment
python -m venv .venv

# Εγκατάσταση εξαρτήσεων
.venv\Scripts\python.exe -m pip install -r requirements.txt

# Εγκατάσταση simulator packages (χωρίς επανεγκατάσταση εξαρτήσεων)
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

### 19.2 Επαλήθευση εγκατάστασης

```bash
# Εκτέλεση test suite
.venv\Scripts\python.exe -m pytest tests/ -q

# Αναμενόμενο: 62+ tests, all pass, < 30 s
```

### 19.3 Demo (παρουσίαση)

```bash
.venv\Scripts\python.exe experiments/liu_replication/demo_single_decision.py
```

### 19.4 Αξιολόγηση πολιτικής (70 λεπτά περίπου)

```bash
.venv\Scripts\python.exe experiments/liu_replication/drl/n30/eval_n30_seed42.py
```

### 19.5 Εκπαίδευση (ΠΡΟΣΟΧΗ: πολύωρο)

```bash
# ΠΡΟΕΙΔΟΠΟΙΗΣΗ: Η εκπαίδευση διαρκεί πολλές ώρες
# Μην εκτελείτε εκτός αν είναι απαραίτητο
.venv\Scripts\python.exe experiments/liu_replication/drl/n30/train_n30.py --seed 42
```

---

## 20. Αφήγηση end-to-end

Αυτή η ενότητα περιγράφει ολόκληρη τη ροή από τη διαμόρφωση του δικτύου έως την τελική απόφαση του DRL agent.

### 20.1 Αρχικοποίηση

Ο χρήστης εκκινεί το σύστημα με N=30 κόμβους. Κάθε κόμβος `i` έχει:
- **Stake** `Υ_i` (tokens, προκαθορισμένα)
- **Θέση** `x_i` (συντεταγμένες km)
- **Δυνατότητα** `c_i` (GHz)
- **Συνδέσεις** `R_{ij}` (Mbps, αρχικά 100 Mbps για GOOD)

### 20.2 Κωδικοποίηση κατάστασης

Τα παραπάνω συνδυάζονται σε ένα διάνυσμα `S(t)` 991 διαστάσεων. Κάθε συστατικό κανονικοποιείται στο [0,1] ώστε το νευρωνικό δίκτυο να λαμβάνει σταθερά κλιμακωμένες εισόδους.

### 20.3 Απόφαση DRL

Ο `DQNAgent` τρέχει ένα forward pass του `QNetwork` για **όλα** τα 3915 ζεύγη `(S(t), A_i)` ταυτόχρονα:

```
Q(S(t), A_i) = f_θ([enc(S(t)); enc(A_i)])   για i=1,...,3915
```

Επιλέγεται η ενέργεια `A* = argmax_i Q(S(t), A_i)`.

### 20.4 Εκτέλεση στο DES

Η επιλεγμένη ενέργεια `A* = (a*, δ*, S_B*, T_I*)` εκτελείται στον simulator:

1. Οι K=28 επιλεγμένοι validators εκκινούν τη διαδικασία consensus.
2. Ο primary validator εκπέμπει πρόταση μπλοκ.
3. Τα μηνύματα διαδίδονται με καθυστερήσεις βάσει `Chain/Network.py` (bandwidth, latency).
4. Κάθε validator επεξεργάζεται τα μηνύματα με καθυστέρηση βάσει `c_i` GHz.
5. Μετά από επαρκείς ψήφους (threshold ανά πρωτόκολλο), το μπλοκ οριστικοποιείται.
6. Ο `ProtocolFinalityRecord` καταγράφει την ακριβή στιγμή τελεσιδικίας.

### 20.5 Αξιολόγηση περιορισμών

```
C1: G(Υ_a*) ≤ 0.2  AND  G(x_a*) ≤ 0.3  →  PASS/FAIL
C2: t_c_des ≤ ω·T_I = 6·2 = 12s         →  PASS/FAIL
C3: f=0 ≤ F^δ*                           →  PASS
```

### 20.6 Υπολογισμός ανταμοιβής

```
Ω = floor(4·10^6 / 200) / 2 = 10000 tx/s   (αν C1∧C2∧C3)
```

### 20.7 Μετάβαση FSMC

Μετά την εκτέλεση, κάθε σύνδεση `R_{ij}` μεταβαίνει στοχαστικά:
- 70%: παραμένει στο ίδιο επίπεδο
- 30%: μεταβαίνει σε γειτονικό επίπεδο

Ο agent λαμβάνει τη νέα κατάσταση `S(t+1)` και η διαδικασία επαναλαμβάνεται.

### 20.8 Τι μαθαίνει ο agent

Μετά από 400 επεισόδια × 20 βήματα = 8000 βήματα εκπαίδευσης (ανά seed), ο agent μαθαίνει:
- Σε **καλές** συνδέσεις (100 Mbps): να επιλέγει μεγάλα μπλοκ (S_B=4 MB) με σύντομα διαστήματα (T_I=2s) για μέγιστο throughput.
- Σε **μεσαίες** συνδέσεις (55 Mbps): να επιλέγει μικρότερα μπλοκ ή μεγαλύτερα T_I για να πετύχει C2.
- Να **αποφεύγει** συνδυασμούς που αποτυγχάνουν σε C1 ή C2 (μηδενική ανταμοιβή).
- Να **προτιμά το Quorum** (γρηγορότερο consensus) όταν f=0, καθώς δεν χρειάζεται η ανοχή σφαλμάτων.

Αυτή η συμπεριφορά αντικατοπτρίζει άμεσα τους στόχους του Liu et al.: βελτιστοποίηση blockchain throughput υπό συνθήκες δυναμικού δικτύου.

---

*Τέλος εγγράφου. Για ερωτήσεις επικοινωνήστε με τον συγγραφέα.*
