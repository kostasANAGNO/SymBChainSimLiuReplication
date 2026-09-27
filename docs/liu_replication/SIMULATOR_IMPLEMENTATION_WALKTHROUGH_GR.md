# src/Simulator/ — Αναλυτικό Implementation Walkthrough

> **Εστίαση:** αποκλειστικά `src/Simulator/` · Αρχεία experiments/ εξαιρούνται.  
> **Μέθοδος κατάταξης:** git log ανά αρχείο — τεκμηριωμένο παρακάτω.  
> **Αναφορά:** Liu et al., IEEE TII 2019, DOI 10.1109/TII.2019.2897805

---

## Κατάταξη αρχείων από git history

| Commit | Ημερομηνία | Τίτλος | Περιεχόμενο |
|--------|-----------|--------|-------------|
| `6b264ed` | 2025-09-26 | Initial clean commit | Αρχικός κώδικας SymBChainSim (Engine, Chain, Manager, Utils βάση) |
| `dc9bc4d` | 2026-09-20 | Extend SymBChainSim core for Liu integration | Τροποποίηση 14 αρχείων κορμού για Liu |
| `8511e7e` | 2026-09-20 | Add validator profiles and DES instrumentation | 8 νέα αρχεία (Chain + Utils) |
| `db1113a` | 2026-09-20 | Add Liu analytical reference model | Όλο το `Liu/` package (29 αρχεία) |
| `07581cc` | 2026-09-20 | Implement Liu consensus DES runtimes | Όλο το `Chain/Consensus/LiuRuntime/` (46 αρχεία) |

**Σύμβολα κατάταξης:**
- `EXISTING_UNCHANGED` — Υπήρχε στο initial commit, δεν αλλάχτηκε για Liu
- `EXISTING_MODIFIED` — Υπήρχε στο initial commit, τροποποιήθηκε στο `dc9bc4d`
- `NEW_FOR_LIU` — Δημιουργήθηκε αποκλειστικά για το Liu replication

---

## Περιεχόμενα

1. [Engine/](#1-engine)
2. [Chain/ — Κορμός δικτύου](#2-chain--κορμός-δικτύου)
3. [Chain/Consensus/LiuRuntime/ — Πρωτόκολλα DES](#3-chainconsensusliuruntime--πρωτόκολλα-des)
4. [Liu/ — Domain/Reference layer](#4-liu--domainreference-layer)
5. [Utils/](#5-utils)
6. [End-to-end dependency/call graph](#6-end-to-end-dependencycall-graph)

---

## 1. Engine/

### Ρητή δήλωση για DES core

Ο DES core **τροποποιήθηκε ελάχιστα** στο `dc9bc4d` (Engine/Event.py, Engine/Handler.py, Engine/Scheduler.py, Engine/Simulation.py) για να υποστηρίξει ένα νέο attribute `liu_context` στα events και epoch-awareness στον Handler, **χωρίς καμία αλλαγή στη βασική DES σημασιολογία**: το min-heap, το virtual clock και ο βρόχος `sim_next_event()` παρέμειναν αναλλοίωτα. Όλη η Liu λογική ενσωματώθηκε ως **additional layer** επάνω στο αναλλοίωτο DES substrate.

---

### 1.1 `Engine/Event.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Engine/Event.py`

**Σκοπός:** Βασικός τύπος γεγονότος DES. Τα γεγονότα έχουν χρόνο εκτέλεσης, handler, creator/actor και προαιρετικό payload.

**Κλάσεις:**

| Κλάση | Πεδία | Σκοπός |
|-------|-------|--------|
| `Event` | `time, payload, handler, creator, actor` | Γενικό DES γεγονός |
| `MessageEvent(Event)` | `+ receiver, forwarded_by` | Μήνυμα node-to-node |
| `SystemEvent(Event)` | (κληρονομεί) | Εσωτερικό γεγονός (timeout, timer) |

**Inputs/Outputs:** Constructor λαμβάνει χρόνο εκτέλεσης και handler. Κατά την εκτέλεση (από `Simulation.sim_next_event()`), ο handler καλείται με το event ως μόνο argument.

**Liu addition (dc9bc4d):** Προστέθηκε δυνατότητα να φέρει το `event` arbitrary attributes όπως `liu_context` (EpochContext) και `recipient_scope`. Αυτό επιτρέπει στον `Handler` να ξεχωρίζει Liu events από κλασικά events.

**Καλείται από:** `EventQueue.Queue.add_event()`, `Engine.Handler.handle_event()`  
**Καλεί:** Τίποτα — παθητικός τύπος δεδομένων.

**Liu concept:** DES event substrate — όχι άμεσα Liu concept, αλλά φέρει `liu_context` που συνδέει κάθε γεγονός με τη δομή `LiuRuntimeEpochContext`.

**Paper vs reconstruction:** Pure infrastructure. Το paper δεν καθορίζει DES implementation — RECONSTRUCTION.

**Tests:** `tests/test_liu_pbft_runtime.py`, `tests/test_liu_quorum_runtime.py` (έμμεσα μέσω event dispatch)

---

### 1.2 `Engine/EventQueue.py` — `EXISTING_UNCHANGED`

**Path:** `src/Simulator/Engine/EventQueue.py`

**Σκοπός:** Min-heap priority queue πάνω στο `heapq`. Το `Queue` είναι wrapper, το `PrioQueue` είναι η πραγματική υλοποίηση.

**Κλάσεις:**

```python
class PrioQueue:
    def add_event(event):    # heapq.heappush με key=(event.time, counter)
    def pop() -> Event:      # heapq.heappop — εξάγει το μικρότερο time
    def is_empty() -> bool

class Queue:                 # Thin wrapper γύρω από PrioQueue
```

**Inputs:** `Event` objects.  
**Outputs:** `Event` objects σε αύξουσα σειρά χρόνου (deterministic — tie-break με counter).

**Δεν τροποποιήθηκε για Liu.** Χρησιμοποιείται απευθείας από `SingleActionRuntime` χωρίς τροποποίηση.

---

### 1.3 `Engine/Simulation.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Engine/Simulation.py`

**Σκοπός:** Κεντρικός ελεγκτής DES: διατηρεί το εικονικό ρολόι, τη Queue και εκτελεί τον κύριο βρόχο.

**Βασική μέθοδος:**

```python
class Simulation:
    clock: float = 0.0   # εικονικό ρολόι

    def sim_next_event(self) -> Event:
        event = self.q.pop()        # εξαγωγή από min-heap
        self.clock = event.time     # προώθηση ρολογιού
        result = handle_event(event) # εκτέλεση handler
        return event
```

**Liu addition (dc9bc4d):** Προστέθηκε tracking του `max_events` counter για τον `DESQueueExhausted` guard στο `SingleActionRuntime`. Επίσης προστέθηκε `simulated_time_limit` check. Η βασική DES σημασιολογία (virtual clock, FIFO-by-time) δεν άλλαξε.

**Καλείται από:** `SingleActionRuntime._run_to_height()`  
**Καλεί:** `EventQueue.Queue.pop()`, `Engine.Handler.handle_event()`

---

### 1.4 `Engine/Handler.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Engine/Handler.py`

**Σκοπός:** Κεντρικός dispatcher. Λαμβάνει κάθε event και αποφασίζει πώς να το χειριστεί: αν ο actor είναι ζωντανός, αν το event ανήκει στο σωστό epoch, αν πρέπει να μπει στο backlog.

**Liu addition (dc9bc4d):** Ο handler ελέγχει αν το `event.liu_context.epoch_id` ταιριάζει με το ενεργό `epoch_id` του κόμβου. Αν όχι, το event καταγράφεται ως stale και απορρίπτεται gracefully — αντί να τερματίζει το simulation.

```python
def handle_event(event, checking_backlog=False) -> str:
    if not event.actor.state.alive:
        return "dead_node"
    
    liu_context = getattr(event, "liu_context", None)
    if liu_context is not None and current_epoch is not None:
        if liu_context.epoch_id != current_epoch.epoch_id:
            # Stale event από προηγούμενο epoch → απόρριψη
            return "stale_epoch"
    
    # ... classic backlog, dispatch logic ...
```

**Liu concept:** Epoch isolation — κρίσιμο για το multi-epoch framework του Liu replication.

**Paper vs reconstruction:** RECONSTRUCTION — το paper δεν καθορίζει πώς γίνεται dispatch των γεγονότων σε multi-epoch setting.

---

### 1.5 `Engine/Scheduler.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Engine/Scheduler.py`

**Σκοπός:** Helper για να εισάγονται events σε μελλοντικό χρόνο (schedule). Χρησιμοποιείται από πρωτόκολλα για να προγραμματίσουν timeouts.

**Liu addition (dc9bc4d):** Minor tweaks για συμβατότητα με τον Liu context payload format.

---

## 2. Chain/ — Κορμός δικτύου

### 2.1 `Chain/Network.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Chain/Network.py`

**Κατάταξη:** EXISTING_MODIFIED (dc9bc4d προσέθεσε ~25 γραμμές)

**Σκοπός:** Singleton που διαχειρίζεται όλους τους κόμβους και υπολογίζει την καθυστέρηση μετάδοσης μηνυμάτων.

**Βασικές συναρτήσεις:**

```python
class Network:
    nodes: list[Node]  # class-level singleton

    @staticmethod
    def calculate_message_propagation_delay(sender, receiver, message) -> float:
        # T_msg = T_tx + T_latency + T_queueing + T_processing
        bandwidth = min(link_rate(sender, receiver), link_rate(receiver, sender))  # bottleneck
        T_tx = message.size / bandwidth
        T_latency = geographic_latency(sender, receiver)
        return T_tx + T_latency + T_queueing + T_processing

    @staticmethod
    def get_bandwidth(sender, receiver) -> float:
        return min(sender.link_rate, receiver.link_rate)
```

**Liu addition (dc9bc4d):** Προστέθηκε `NodeProfile`-aware path για Liu nodes: όταν ένας κόμβος έχει `NodeProfile`, χρησιμοποιείται ο `ComputationalDelay.scale()` για το processing delay αντί για το παλαιό σταθερό μοντέλο.

**Liu concept:** Μοντελοποίηση φυσικού layer — transmission delay + geographical latency. Το Liu paper χρησιμοποιεί `R_{ij}` (Mbps) για το transmission time στις εξισώσεις Appendix B.

**Paper vs reconstruction:** Transmission delay από `R_{ij}` = PAPER. Το latency model (γεωγραφική απόσταση) = RECONSTRUCTION — το paper δεν καθορίζει φυσικό layer μοντέλο.

**Παράδειγμα:** Μήνυμα 4 MB σε link 100 Mbps: `T_tx = 8*4/100 = 0.32 s`.

**Tests:** `tests/test_liu_pbft_runtime.py` (έμμεσα)

---

### 2.2 `Chain/Node.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Chain/Node.py`

**Κατάταξη:** EXISTING_MODIFIED (dc9bc4d, +67 γραμμές)

**Σκοπός:** Αναπαράσταση ενός blockchain node. Διατηρεί: blockchain (list of Blocks), mempool (transaction pool), consensus protocol (cp), reconfiguration state.

**Liu addition (dc9bc4d):** Προστέθηκε `node_profile: NodeProfile | None` attribute και ενσωμάτωση με τον `NodeProfileSet`. Επίσης προστέθηκε `liu_link_state` για την αποθήκευση του τρέχοντος FSMC state του κόμβου.

**Inputs:** Configuration dict από YAML (base.yaml κ.λπ.)  
**Outputs:** Αλλαγές κατάστασης μέσω events.

**Καλείται από:** `SingleActionRuntime` (δημιουργεί nodes), `LiuRuntimeActionApplicator` (ανανεώνει epoch context)

---

### 2.3 `Chain/NodeProfile.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/NodeProfile.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Αμετάβλητες ερευνητικές ιδιότητες κόμβου που αφορούν το Liu state vector: `stake_tokens` (Υ_i) και `computational_capability_ghz` (c_i).

**Dataclasses:**

```python
@dataclass(frozen=True, slots=True)
class NodeProfile:
    node_id: int
    stake_tokens: float | None
    computational_capability_ghz: float | None

@dataclass(frozen=True, slots=True)
class NodeProfileSet:
    profiles: tuple[NodeProfile, ...]

    @classmethod
    def from_config(cls, total_nodes, config) -> "NodeProfileSet":
        # Φορτώνει stakes και GHz από YAML
```

**Inputs:** `total_nodes: int`, `config: dict` με `stake_tokens[]` και `computational_capability_ghz[]`.  
**Outputs:** `NodeProfileSet` — αμετάβλητη συλλογή N profiles.

**Liu concept:** Αντιστοιχεί στα Υ (stakes) και c (capabilities) του Liu state S(t) = [χ, Υ, x, c, R].

**Paper:** Υ_i ∈ Υ (stakes) — PAPER. c_i ∈ c (GHz) — PAPER. Η δομή `NodeProfile/NodeProfileSet` = RECONSTRUCTION (implementation detail).

**Καλείται από:** `SingleActionRuntime` (αρχικοποίηση), `StateBuilder` (χτίζει LiuState)  
**Καλεί:** `Liu.Validation.require_finite_number`

**Tests:** `tests/test_liu_node_validator.py`

---

### 2.4 `Chain/ValidatorSet.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/ValidatorSet.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Αμετάβλητη, ordered λίστα validator IDs. Single source of truth για validator membership και proposer ordering.

```python
@dataclass(frozen=True, slots=True)
class ValidatorSet:
    ids: tuple[int, ...]
    _id_lookup: frozenset[int]   # O(1) membership check

    def proposer_for(self, selection_value: int) -> int:
        return self.ids[selection_value % self.count]  # round-robin

    def contains(self, node_id: int) -> bool: ...
```

**Inputs:** `tuple[int, ...]` validator IDs (sorted, unique).  
**Outputs:** Membership checks και proposer selection.

**Liu concept:** Αντιστοιχεί στο σύνολο validators a ⊆ {0,...,N-1} της Liu action A(t) = [a, δ, S_B, T_I].

**Paper vs reconstruction:** Το Liu ορίζει a = selected validators — PAPER. Το `proposer_for()` = RECONSTRUCTION (height-based round-robin για Liu client role).

**Καλείται από:** `Roles.LiuPBFTRoles.for_height_view()`, `Roles.LiuQuorumRoles.for_height()`, `Roles.LiuZyzzyvaRoles.for_height_view()`  
**Tests:** `tests/test_liu_node_validator.py`

---

### 2.5 `Chain/Block.py` — `EXISTING_UNCHANGED`

**Path:** `src/Simulator/Chain/Block.py`

**Κατάταξη:** EXISTING_UNCHANGED

**Σκοπός:** Αναπαράσταση ενός μπλοκ blockchain. Χρησιμοποιείται από LiuRuntime protocols για αποθήκευση finalized block data.

**Liu addition:** Τα LiuRuntime protocols προσθέτουν `extra_data["liu_identity"]` = `LiuBlockIdentity.to_dict()` και `extra_data["round"]` σε κάθε block για να το κάνουν identifiable.

---

### 2.6 `Chain/TransactionFactory.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Chain/TransactionFactory.py`

**Κατάταξη:** EXISTING_MODIFIED (dc9bc4d, +84 γραμμές)

**Σκοπός:** Δημιουργεί συναλλαγές για το mempool και τα μπλοκ. Ο `execute_transactions()` επιλέγει transactions για ένα μπλοκ και υπολογίζει το μέγεθος.

**Liu addition (dc9bc4d):** Προστέθηκε `offered_workload_tx` parameter — ο αριθμός των pre-loaded transactions στο mempool ώστε να αποφεύγεται transaction starvation σε μεγάλα block sizes. Επίσης προστέθηκε transaction size scaling βάσει `chi_bytes` (μέγεθος συναλλαγής Liu).

**Κρίσιμη απόφαση:** `offered_workload_tx = 100_000` (όχι 30_000) — με 30k transactions, μεγάλα block sizes (4 MB / 200 B = 20k tx) εξαντλούν το pool, προκαλώντας ψευδείς αποτυχίες C2.

**Καλείται από:** `LiuPBFT/Transitions.start_request()`, `LiuZyzzyva/Transitions.start_request()`, `LiuQuorum/Transitions.start_request()`

---

### 2.7 `Chain/Reconfiguration/` — `EXISTING_MODIFIED`

**Κατάταξη:** Τα αρχεία υπήρχαν από πριν, τροποποιήθηκαν ελάχιστα (dc9bc4d: +2-5 γραμμές ανά αρχείο) για συμβατότητα με Liu epoch context.

| Αρχείο | Ρόλος |
|--------|-------|
| `CentralisedReconfiguration.py` | Επεξεργασία ConfigurationBlock events |
| `ConfigurationBlock.py` | Αναπαράσταση μπλοκ αναδιαμόρφωσης |
| `ReconfigurationState.py` | Κατάσταση αναδιαμόρφωσης ανά κόμβο |

---

## 3. Chain/Consensus/LiuRuntime/ — Πρωτόκολλα DES

**Κατάταξη:** Όλα NEW_FOR_LIU (07581cc)

Αυτό είναι το πιο σύνθετο υποσύστημα: ένα **πλήρες DES implementation** των τριών πρωτοκόλλων consensus του Liu et al., οργανωμένο σε υποφακέλους.

### Αρχιτεκτονική LiuRuntime

```
LiuRuntime/
├── Common/             ← Κοινός κώδικας και types
│   ├── ActionApplicator.py   ← Εφαρμογή LiuAction στο runtime
│   ├── Certificates.py       ← Certificate types (PBFT, Zyzzyva, Quorum)
│   ├── Configuration.py      ← LiuEnvironmentConfig
│   ├── Environment.py        ← Abstract base environment
│   ├── EpochContext.py       ← Αμετάβλητο epoch snapshot
│   ├── EventContext.py       ← Per-event context (epoch, height, view)
│   ├── Identity.py           ← LiuBlockIdentity (SHA-256)
│   ├── ProtocolFactory.py    ← Factory για PBFT/Zyzzyva/Quorum instances
│   ├── Roles.py              ← Deterministic client/primary/replica roles
│   ├── RuntimeEvaluation.py  ← C1/C2/C3 + ανταμοιβή Ω
│   ├── RuntimeProtocol.py    ← Abstract base protocol
│   ├── SingleActionRuntime.py ← run_single_action() + C2Deadline/DESExhausted
│   ├── StateBuilder.py       ← LiuState από runtime snapshot
│   ├── StateEvolution.py     ← S_t → S_(t+1) με FSMC
│   └── TimeoutPolicy.py      ← Versioned timeout scope
├── LiuPBFT/            ← PBFT implementation
├── LiuZyzzyva/         ← Zyzzyva implementation
├── LiuQuorum/          ← Quorum implementation
├── Processing/         ← CPU cost model
└── Transport/          ← Message sending layer
```

---

### 3.1 Common/Identity.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/Identity.py`

**Σκοπός:** Collision-resistant block/message identity βάσει SHA-256.

**Κλάσεις:**

```python
@dataclass(frozen=True, slots=True)
class LiuBlockIdentity:
    epoch_id: int
    height: int
    parent_digest: str   # SHA-256 του parent block
    block_digest: str    # SHA-256 του (epoch, height, parent, transactions)

    @classmethod
    def create(cls, epoch_id, height, parent, client_id, transactions) -> "LiuBlockIdentity":
        digest = sha256({"client_id": ..., "epoch_id": ..., "height": ..., ...})
        return cls(epoch_id, height, parent, digest)

def logical_message_id(identity, message_type, sender_id, receiver_id, view) -> str:
    # SHA-256 της μοναδικής σύνθεσης
```

**Liu concept:** Block identity για finality tracking — δεν ορίζεται ρητά στο paper. RECONSTRUCTION.

**Χρησιμοποιείται από:** Όλα τα protocol transitions για block identification και certificate creation.

---

### 3.2 Common/Roles.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/Roles.py`

**Σκοπός:** Ντετερμινιστική ανάθεση ρόλων (client, primary, replicas, backups) από το `ValidatorSet` και το (height, view).

```python
class LiuQuorumRoles:   # Client = validators[height-1 % K], Replicas = rest
class LiuPBFTRoles:     # Client = validators[(height-1) % K], 
                         # Primary = replicas[view % (K-1)]
class LiuZyzzyvaRoles:  # Client = validators[height % K],
                         # Primary = backups[view % (K-1)]
```

**Liu concept:** Το Liu paper καθορίζει client ως τον "block producer" — PAPER. Η συγκεκριμένη height-based round-robin mapping = RECONSTRUCTION.

---

### 3.3 Common/Certificates.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/Certificates.py`

**Σκοπός:** Αμετάβλητα certificate evidence που απαιτούνται από τα πρωτόκολλα για πρόοδο.

**Certificate types:**

| Certificate | Πρωτόκολλο | Σκοπός | Threshold |
|-------------|-----------|--------|-----------|
| `PBFTPreparedCertificate` | PBFT | 2f+1 prepare votes | 2f+1 |
| `PBFTCommitCertificate` | PBFT | 2f+1 commit votes | 2f+1 |
| `PBFTFinalityCertificate` | PBFT | Finality evidence | 2f+1 |
| `PBFTViewChangeEvidence` | PBFT | f+1 view-change | f+1 |
| `PBFTNewViewCertificate` | PBFT | New-view proof | f+1 |
| `ZyzzyvaFastReplyCertificate` | Zyzzyva | K replies (fast path) | K |
| `ZyzzyvaCommitCertificate` | Zyzzyva | 2f+1 commits (recovery) | 2f+1 |
| `ZyzzyvaFinalityCertificate` | Zyzzyva | Finality evidence | — |
| `LiuQuorumReplyCertificate` | Quorum | K-1 replies (all replicas) | K-1 |

Κάθε certificate ελέγχει: `len(signer_ids) == threshold`, unique signers, epoch/height/view match.

**Paper vs reconstruction:** Τα certificates αντιστοιχούν στα "quorum" και "commit" stages των πρωτοκόλλων — PAPER semantics, RECONSTRUCTION implementation.

---

### 3.4 Common/EpochContext.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/EpochContext.py`

**Σκοπός:** Αμετάβλητο snapshot ενός Liu action epoch — περιέχει όλα όσα χρειάζεται ένα πρωτόκολλο κατά τη διάρκεια εκτέλεσης.

```python
@dataclass(frozen=True)
class LiuRuntimeEpochContext:
    epoch_id: int
    epoch_hash: str            # SHA-256 canonical hash
    epoch_configuration: EpochConfiguration  # validator_set, protocol, S_B, T_I
    link_state_matrix: LinkStateMatrix        # R (NxN Mbps)
    capabilities_ghz: dict[int, float]       # {node_id: GHz}
    validator_ids: frozenset[int]
    
    def capability_ghz(self, node_id) -> float: ...
    def link_rate(self, sender_id, receiver_id) -> float: ...
```

**Liu concept:** Ένα epoch αντιστοιχεί σε ένα "decision step" του Liu MDP — ο agent επιλέγει μία action, η οποία ορίζει ένα epoch configuration.

---

### 3.5 Common/EventContext.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/EventContext.py`

**Σκοπός:** Immutable per-event context. Βρίσκεται σε κάθε `event.liu_context` και επιτρέπει στον `Handler` να ελέγχει αν ένα event ανήκει στο τρέχον epoch.

```python
@dataclass(frozen=True)
class LiuEventContext:
    epoch_id: int
    epoch_hash: str
    height: int
    view: int
    block_identity: LiuBlockIdentity
    logical_message_id: str
    protocol: str
```

---

### 3.6 Common/TimeoutPolicy.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/TimeoutPolicy.py`

**Σκοπός:** Versioned timeout scope semantics. Τα timeouts δεσμεύονται σε συγκεκριμένο (epoch, height, view, phase) — stale timeouts αγνοούνται αυτόματα.

```python
TIMEOUT_POLICY_VERSION = "liu_runtime_phase_relative_timeout_v2"

def scoped_timeout_payload(event, phase):
    event.payload.update({"epoch_id": ..., "height": ..., "view": ..., "phase": ...})

def timeout_scope_matches(protocol, event) -> bool:
    # Ελέγχει αν το timeout ανήκει στην τρέχουσα φάση
```

**Liu concept:** Το Liu paper καθορίζει timeout T — PAPER. Το versioned scope matching = RECONSTRUCTION για safe view-change.

---

### 3.7 Transport/NetworkAdapter.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Transport/NetworkAdapter.py`

**Σκοπός:** Αποστολή directed messages μεταξύ Liu nodes. Χρησιμοποιεί τον πίνακα R_{ij} για υπολογισμό serialization delay.

```python
class LiuDirectedTransport:
    def send(sender, receiver_id, sent_at, payload, handler, context, payload_size_mb):
        rate = epoch_context.link_state_matrix.rate(sender.id, receiver_id)  # R_{ij}
        serialization = 8.0 * payload_size_mb / rate  # Mbps → s
        arrival = sent_at + serialization + propagation_delay_s
        # Δημιουργεί MessageEvent και το καταγράφει στο LiuRuntimeInstrumentationCollector
```

**Liu concept:** `T_transmission = S_B * 8 / R_{ij}` — PAPER (Appendix B, Eq. 15-17).

**Παράδειγμα:** 4 MB block, R_{ij}=55 Mbps: `T_tx = 8*4/55 ≈ 0.582 s`.

---

### 3.8 Transport/MessageSizePolicy.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Transport/MessageSizePolicy.py`

**Σκοπός:** Explict policy για το μέγεθος των Liu messages. Κάθε φάση (Request, Reply, Announcement) μεταδίδει S_B MB.

```python
class LiuPaperMessageSizePolicy:
    def request_size_mb(self, block_size_mb) -> float: return block_size_mb
    def reply_size_mb(self, block_size_mb) -> float: return block_size_mb
```

**Paper vs reconstruction:** Το Liu paper αναφέρει S_B per phase — PAPER. Το ότι κάθε φάση μεταδίδει ολόκληρο το S_B = PAPER (literal από Appendix B).

---

### 3.9 Processing/ProcessingWork.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Processing/ProcessingWork.py`

**Σκοπός:** Αναπαράσταση crypto work σε Liu α/β operations (α = signature cycles, β = MAC cycles).

```python
@dataclass(frozen=True, slots=True)
class LiuProcessingWork:
    signature_operations: int   # α
    mac_operations: int         # β

    # Factory methods ανά ρόλο/πρωτόκολλο:
    @classmethod def quorum_replica_work(cls) -> LiuProcessingWork:
        return cls(signature_operations=1, mac_operations=2)

    @classmethod def pbft_primary_request_work(cls) -> LiuProcessingWork:
        return cls(signature_operations=1, mac_operations=2)

    @classmethod def pbft_backup_preprepare_work(cls) -> LiuProcessingWork:
        return cls(signature_operations=1, mac_operations=1)

    @classmethod def pbft_prepare_quorum_work(cls, replica_count) -> LiuProcessingWork:
        return cls(signature_operations=0, mac_operations=2 * replica_count)
    
    # κ.λπ. για Zyzzyva primary/replica/recovery
```

**Liu concept:** Appendix B Eq.15-17: processing time = (α·sig + β·mac) / (c_i · 10^9). PAPER concept, RECONSTRUCTION counts (paper δεν καθορίζει exact operation counts).

---

### 3.10 Processing/ProcessingCostModel.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Processing/ProcessingCostModel.py`

**Σκοπός:** Μετατρέπει `LiuProcessingWork` σε δευτερόλεπτα χρησιμοποιώντας `c_i` GHz.

```python
class LiuProcessingCostModel:
    signature_cycles_alpha: float   # α
    mac_cycles_beta: float          # β

    def cycles(self, work: LiuProcessingWork) -> float:
        return work.signature_operations * alpha + work.mac_operations * beta

    def duration_s(self, work: LiuProcessingWork, capability_ghz: float) -> float:
        return self.cycles(work) / (capability_ghz * 1_000_000_000.0)
```

**Παράδειγμα:** Quorum replica work (1 sig, 2 MACs), α=10^7, β=5·10^6, c=2.5 GHz:
`cycles = 10^7 + 2*5·10^6 = 2·10^7`, `duration = 2·10^7 / (2.5·10^9) ≈ 0.008 s`.

**Liu concept:** Liu Eq. 15-17 processing term — PAPER formula, RECONSTRUCTION cycle counts.

---

### 3.11 Processing/NodeComputeQueue.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Processing/NodeComputeQueue.py`

**Σκοπός:** Per-node CPU queue — σειριοποιεί processing work (ένα task τη φορά). Αν ο κόμβος είναι ήδη απασχολημένος, το επόμενο task αρχίζει μετά το τέλος του προηγούμενου.

```python
class NodeComputeQueue:
    _next_available: float = 0.0

    def reserve(self, current_time: float, duration_s: float) -> Reservation:
        start = max(current_time, self._next_available)
        end = start + duration_s
        self._next_available = end
        return Reservation(start, end)
```

**Liu concept:** CPU serialization — RECONSTRUCTION. Το paper δεν ορίζει αν processing είναι parallel ή sequential.

---

### 3.12 Common/SingleActionRuntime.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/SingleActionRuntime.py`

**Κατάταξη:** NEW_FOR_LIU — ο κεντρικός orchestrator ολόκληρης της DES εκτέλεσης.

**Σκοπός:** Λαμβάνει ένα `LiuAction` και μια κατάσταση δικτύου και τρέχει το DES simulation μέχρι finality ή failure.

**Βασικές exception classes:**

```python
class C2DeadlineExceeded(Exception):
    """DES simulated time > omega * T_I. Αυθεντική αποτυχία C2 από Liu."""

class DESQueueExhausted(Exception):
    """Queue αδειάσε χωρίς finality. Guard υλοποίησης — PBFT view-change cap."""
```

**Inputs:** `LiuSingleActionSnapshot` (S0 snapshot), `LiuAction` A0, `omega=6.0`  
**Outputs:** `ProtocolFinalityRecord` (τελεσιδικία) ή exception

**Σημαντική λεπτομέρεια:** Η `finality_time` = στιγμή που ο **πρώτος** validator κάνει commit, όχι όλοι. Αυτό αντιστοιχεί στον Liu ορισμό τελεσιδικίας.

**Υπολογισμός C2:**

```python
c2_deadline_s = boundary_time + omega * T_I
t_c_des = finality_time - request_sent_at
c2_pass = t_c_des <= (omega - 1) * T_I
```

**Καλείται από:** `LiuDynamicEpochEnv.step()` (experiments/) — η μόνη entry point.  
**Καλεί:** `LiuRuntimeStateBuilder`, `LiuRuntimeActionApplicator`, `Network`, `Node`, `Engine.Simulation`

**Tests:** `tests/test_liu_single_action_closed_loop.py`, `tests/test_liu_runtime.py`

---

### 3.13 Common/RuntimeEvaluation.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/RuntimeEvaluation.py`

**Σκοπός:** Post-execution αξιολόγηση C1/C2/C3 και υπολογισμός ανταμοιβής Ω από DES evidence.

**Βασικές κλάσεις:**

```python
class LiuRewardStatus(Enum):
    FEASIBLE = "FEASIBLE"
    C1_FAILED = "C1_FAILED"
    C2_FAILED = "C2_FAILED"
    C3_FAILED = "C3_FAILED"

class LiuRuntimeConstraintEvaluator:
    def evaluate(state, action, measurement) -> LiuRuntimeEpochEvaluation:
        c1 = check_c1(selected_stakes, selected_positions)
        c2 = check_c2(t_c_des, omega, T_I)
        c3 = check_c3(faulty_validators, protocol, K)
        
        if c1 and c2 and c3:
            liu_throughput = floor(S_B * 1e6 / chi) / T_I
            reward = liu_throughput
        else:
            reward = 0.0
        
        return LiuRuntimeEpochEvaluation(reward=reward, status=status, ...)
```

**Fault tolerance per protocol:**

```python
def tolerated_faults(protocol, K) -> int:
    if protocol == LIU_QUORUM: return 0           # f_max = 0
    else: return (K - 1) // 3                     # f_max = floor((K-1)/3)
```

**Paper vs reconstruction:**
- C1: G(Υ) ≤ η_s=0.2 AND G(λ) ≤ η_l=0.3 — PAPER (Liu Eq. 2-5)
- C2: T_F ≤ ω·T_I — PAPER (Liu Eq. 6-8)
- C3: f ≤ F^δ — PAPER (Liu Eq. 9)
- Ανταμοιβή Ω = floor(S_B·10^6/χ)/T_I — PAPER (Liu Eq. 13)

**Tests:** `tests/test_liu_reward_semantics.py`

---

### 3.14 Common/ActionApplicator.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/ActionApplicator.py`

**Σκοπός:** Εφαρμογή ενός `LiuAction` στο runtime — αλλάζει validators, πρωτόκολλο, block size, block interval. Διαχειρίζεται epoch transitions.

```python
class LiuRuntimeActionApplicator:
    def apply_action(action: LiuAction, activation_time: float) -> LiuEpochActivationResult:
        # Δημιουργεί νέο EpochConfiguration, ValidatorSet, EpochContext
        # Ενημερώνει τους κόμβους για νέους validators
        # Καταγράφει EpochLifecycleRecord
```

**Liu concept:** Transition A_t → νέο epoch — κεντρικό στο Liu MDP framework.

---

### 3.15 Common/StateBuilder.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/StateBuilder.py`

**Σκοπός:** Χτίζει `LiuState` από runtime snapshot. Υπολογίζει transaction size χ από empirical mean (pool observations) ή configured value.

**Κύρια κλάση:**

```python
class LiuRuntimeStateBuilder:
    def build(epoch_context, instr_collector, previous_state=None) -> LiuRuntimeStateBuildResult:
        chi = _compute_chi(...)          # empirical mean ή configured
        stakes = epoch_context.capabilities_ghz  # (N,)
        link_matrix = epoch_context.link_state_matrix  # R
        stake_gini = canonical_pairwise_gini(...)
        geo_gini = GridSpatialIntensityModel.evaluate(...)
        state = LiuState(chi, stakes, spatial_profiles, capabilities, link_matrix)
        return LiuRuntimeStateBuildResult(state=state, state_hash=state.deterministic_hash(), ...)
```

---

### 3.16 Common/StateEvolution.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/StateEvolution.py`

**Σκοπός:** Συντονιστής S_t → S_{t+1} μετά από finalized epoch. Καλεί `link_fsmc.advance(rng)` για να μεταβεί σε νέο network state.

```python
class LiuRuntimeStateEvolution:
    def advance_after_epoch(action, finality_record):
        # 1. Εφαρμόζει action στο runtime (ActionApplicator)
        # 2. Υπολογίζει C1/C2/C3 + reward (RuntimeEvaluation)
        # 3. Κάνει FSMC transition για S_{t+1}
        # 4. Χτίζει LiuState S_{t+1} (StateBuilder)
        return (reward, next_state, evaluation)
```

**Liu concept:** Πλήρης S_t, A_t → r, S_{t+1} MDP transition — PAPER concept.

---

### 3.17 LiuPBFT/ — Call/Event Flow

**Φάσεις PBFT:**

```
CLIENT_WAITING → [start_request]
    ↓  Δημιουργεί LiuBlockIdentity, επιλέγει transactions
    ↓  Στέλνει REQUEST σε PRIMARY (S_B MB per message)
PRIMARY (WAITING_PREPREPARE → PROCESSING_PREPREPARE)
    ↓  reserve_processing(pbft_primary_request_work)
    ↓  [CPU delay] Στέλνει PRE-PREPARE σε όλα τα BACKUPs
BACKUPS (WAITING_PREPREPARE → PROCESSING_PREPARE)
    ↓  reserve_processing(pbft_backup_preprepare_work)
    ↓  [CPU delay] Στέλνουν PREPARE σε όλους τους validators
ALL_REPLICAS (counting PREPARE messages)
    ↓  Μόλις 2f+1 PREPAREs → PBFTPreparedCertificate
    ↓  reserve_processing(pbft_prepare_quorum_work)
    ↓  [CPU delay] Στέλνει COMMIT σε όλους
ALL_REPLICAS (counting COMMIT messages)
    ↓  Μόλις 2f+1 COMMITs → PBFTCommitCertificate
    ↓  FINALIZED: Δημιουργεί Block, καταγράφει ProtocolFinalityRecord
    ↓  Στέλνει FINALIZED_BLOCK announcement
CLIENT ← ProtocolFinalityRecord (finality_time = τώρα)
```

**View-change (αν timeout):**
```
Timeout → VIEW_CHANGE (f+1 view-changes) → NEW_VIEW → επανεκκίνηση από νέο primary
Max view-changes cap → DESQueueExhausted (RECONSTRUCTION guard)
```

**Αρχεία:**

| Αρχείο | Ρόλος |
|--------|-------|
| `LiuPBFT/Protocol.py` | `LiuPBFT` class, `init()`, `start()`, role assignment |
| `LiuPBFT/State.py` | `LiuPBFTState(phase, height, view, certificates)` |
| `LiuPBFT/Messages.py` | Message type constants + `send_*()` helpers |
| `LiuPBFT/Transitions.py` | `start_request()`, `receive_preprepare()`, `receive_prepare()`, `receive_commit()`, `request_timeout()`, view-change logic |
| `LiuPBFT/Configuration.py` | `LiuPBFTRuntimeConfiguration` (timeout, α, β) |

**Liu concept:** PBFT από Liu Appendix B, Eq. 15 — PAPER. View-change cap = RECONSTRUCTION.

**Tests:** `tests/test_liu_pbft_runtime.py`, `tests/test_liu_pbft_view_change.py`

---

### 3.18 LiuZyzzyva/ — Call/Event Flow

**Φάσεις Zyzzyva (fast path):**

```
CLIENT (CLIENT_READY) → [start_request]
    ↓  Στέλνει REQUEST σε PRIMARY
PRIMARY (PRIMARY_WAITING_REQUEST → PROCESSING_ORDER)
    ↓  reserve_processing(zyzzyva_primary_order_work)
    ↓  [CPU delay] Στέλνει ORDER σε όλα τα BACKUPs
BACKUPS (BACKUP_WAITING_ORDER → PROCESSING_SPECULATIVE)
    ↓  reserve_processing(zyzzyva_speculative_replica_work)
    ↓  [CPU delay] Στέλνει REPLY απευθείας στον CLIENT
CLIENT (collecting REPLYs)
    ↓  Μόλις K REPLYs (ΟΛΑ τα replicas) → ZyzzyvaFastReplyCertificate
    ↓  FINALIZED (fast path): ProtocolFinalityRecord
```

**Recovery path (αν missing replicas):**

```
CLIENT έχει 2f+1 ≤ replies < K
    ↓  COMMIT message σε BACKUPs (ZyzzyvaSpeculativeEvidence)
BACKUPs → RECOVERY_COMMIT σε CLIENT
    ↓  2f+1 COMMITs → ZyzzyvaCommitCertificate
    ↓  FINALIZED (recovery path): ProtocolFinalityRecord
```

**Αρχεία:**

| Αρχείο | Ρόλος |
|--------|-------|
| `LiuZyzzyva/Protocol.py` | `LiuZyzzyva` class, fast/recovery path routing |
| `LiuZyzzyva/State.py` | `LiuZyzzyvaState(phase, speculative_execution_state)` |
| `LiuZyzzyva/Messages.py` | Message constants + `send_*()` helpers |
| `LiuZyzzyva/Transitions.py` | Fast path + recovery path transitions |
| `LiuZyzzyva/Configuration.py` | `LiuZyzzyvaRuntimeConfiguration` |

**Liu concept:** Zyzzyva fast/recovery paths από Liu Appendix B, Eq. 16-17 — PAPER.

**Tests:** `tests/test_liu_zyzzyva_runtime.py`

---

### 3.19 LiuQuorum/ — Call/Event Flow

**Φάσεις Quorum:**

```
CLIENT → [START_REQUEST] (μετά από T_I δευτερόλεπτα delay)
    ↓  TransactionFactory.execute_transactions()
    ↓  Δημιουργεί LiuBlockIdentity + Block
    ↓  Υπολογίζει estimated_slowest_required_round_trip_s() → timeout
    ↓  Στέλνει REQUEST σε ΌΛΟΥΣ τους replicas (K-1)
    ↓  Schedules TIMEOUT = estimated_rtt * safety_factor
REPLICAS (WAITING_REQUEST → PROCESSING_REQUEST_COMPLETE)
    ↓  reserve_processing(quorum_replica_work) [1 sig, 2 MACs]
    ↓  [CPU delay] Στέλνει REPLY στον CLIENT
CLIENT (collecting REPLYs)
    ↓  Μόλις K-1 REPLYs (ΟΛΑ τα replicas, f=0) → LiuQuorumReplyCertificate
    ↓  FINALIZED: ProtocolFinalityRecord (finality_path="quorum_all_replies")
    ↓  Announcements σε observers → FINALIZED_BLOCK
```

**Timeout (αν missing reply):**
```
TIMEOUT fires → request_timeout() → DES event expires → C2DeadlineExceeded
```

**Αρχεία:**

| Αρχείο | Ρόλος |
|--------|-------|
| `LiuQuorum/Protocol.py` | `LiuQuorum` class, leaderless (no primary) |
| `LiuQuorum/State.py` | `LiuQuorumState(phase, replies dict)` |
| `LiuQuorum/Messages.py` | REQUEST, REPLY, TIMEOUT, FINALIZED_BLOCK constants |
| `LiuQuorum/Transitions.py` | `start_request()`, `receive_request()`, `receive_reply()`, `request_timeout()` |
| `LiuQuorum/Configuration.py` | `LiuQuorumRuntimeConfiguration` + `ReplicaFaultPolicy` (f=0) |
| `LiuQuorum/AdmissibleTimeout.py` | `estimated_slowest_required_round_trip_s()` (RECONSTRUCTION) |

**Liu concept:** Leaderless request/reply — PAPER. F=0 (unanimous quorum) — PAPER. State-action adaptive timeout = RECONSTRUCTION.

**Tests:** `tests/test_liu_quorum_runtime.py`

---

### 3.20 Common/ProtocolFactory.py — `NEW_FOR_LIU`

**Path:** `src/Simulator/Chain/Consensus/LiuRuntime/Common/ProtocolFactory.py`

**Σκοπός:** Factory που δημιουργεί το σωστό protocol instance (LiuPBFT / LiuZyzzyva / LiuQuorum) βάσει του `LiuConsensusProtocol` enum.

```python
class LiuRuntimeProtocolFactory:
    def create(node, epoch_context, settings) -> LiuRuntimeConsensusProtocol:
        if protocol == LIU_QUORUM: return LiuQuorum(node, config)
        if protocol == PBFT: return LiuPBFT(node, config)
        if protocol == ZYZZYVA: return LiuZyzzyva(node, config)
```

---

## 4. Liu/ — Domain/Reference layer

**Κατάταξη:** Όλα NEW_FOR_LIU (db1113a)

Το `Liu/` package είναι ο **domain/reference layer** — περιέχει immutable domain types, αναλυτικά μοντέλα (paper equations), FSMC, serialization. Δεν εκτελεί DES — αυτό γίνεται στο `Chain/Consensus/LiuRuntime/`.

---

### 4.1 `Liu/Validation.py` — `NEW_FOR_LIU`

**Σκοπός:** Κοινά validation primitives για όλο το Liu domain layer.

```python
def require_integer(value, field_name, minimum=0) -> int
def require_finite_number(value, field_name, positive=False, non_negative=False) -> float
```

**Χρησιμοποιείται από:** Σχεδόν όλα τα Liu dataclasses στο `__post_init__`.

---

### 4.2 `Liu/Serialization.py` — `NEW_FOR_LIU`

**Σκοπός:** Abstract base class για canonical JSON serialization και SHA-256 hashing.

```python
class CanonicalSerializable(ABC):
    @abstractmethod
    def to_dict(self) -> dict: ...

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ...)

    def deterministic_hash(self) -> str:
        return sha256(self.canonical_json())
```

**Σημασία:** Παρέχει process-independent SHA-256 για state hashing — κρίσιμο για reproducibility και debugging.

---

### 4.3 `Liu/Protocol.py` — `NEW_FOR_LIU`

**Σκοπός:** Enum με τα τρία πρωτόκολλα του Liu action domain.

```python
class LiuConsensusProtocol(str, Enum):
    PBFT = "PBFT"
    ZYZZYVA = "ZYZZYVA"
    LIU_QUORUM = "LIU_QUORUM"
```

**Paper vs reconstruction:** PAPER — Liu et al. ορίζουν ακριβώς αυτά τα τρία.

---

### 4.4 `Liu/Action.py` — `NEW_FOR_LIU`

**Σκοπός:** Immutable τύπος για Liu action A(t) = [a, δ, S_B, T_I].

```python
@dataclass(frozen=True, slots=True)
class LiuAction(CanonicalSerializable):
    node_count: int
    validator_count: int
    validator_ids: tuple[int, ...]   # a (sorted, unique)
    consensus_protocol: LiuConsensusProtocol  # δ
    block_size_mb: float             # S_B (0.2 MB step)
    block_interval_s: float          # T_I (0.5 s step)
```

**Validation:** Block size step = 0.2 MB, interval step = 0.5 s — αυτά τα βήματα αντιστοιχούν στο Liu action grid.

**Paper:** Eq. 11 ορίζει A(t) = [a, δ, S^B, T^I] — PAPER.

**Παράδειγμα:** `LiuAction(30, 28, (0,1,...,27), LiuConsensusProtocol.LIU_QUORUM, 4.0, 2.0)`

---

### 4.5 `Liu/State.py` — `NEW_FOR_LIU`

**Σκοπός:** Immutable τύπος για Liu state S(t) = [χ, Υ, x, c, R].

```python
@dataclass(frozen=True, slots=True)
class LiuState(CanonicalSerializable):
    transaction_size_bytes: float        # χ
    stakes_tokens: tuple[float, ...]     # Υ (N-vector)
    spatial_profiles: SpatialProfileSet  # x (N×2 km)
    computational_capabilities_ghz: tuple[float, ...]  # c (N-vector)
    link_state_matrix: LinkStateMatrix   # R (N×N Mbps)
```

**Paper:** Eq. 10 ορίζει S(t) = [χ, Υ, x, c, R] — PAPER.

---

### 4.6 `Liu/LinkState.py` — `NEW_FOR_LIU`

**Σκοπός:** Immutable N×N πίνακας link rates σε Mbps. Διαγώνιος = None (self-links δεν υπάρχουν).

```python
@dataclass(frozen=True, slots=True)
class LinkStateMatrix(CanonicalSerializable):
    node_count: int
    rates_mbps: tuple[tuple[float | None, ...], ...]

    def rate(self, sender_id, receiver_id) -> float: ...  # O(1)
    
    @classmethod
    def from_rows(cls, rates_mbps) -> "LinkStateMatrix": ...
```

**Paper:** R_{ij} = link rate i→j — PAPER. Η immutable dataclass implementation = RECONSTRUCTION.

---

### 4.7 `Liu/Spatial.py` — `NEW_FOR_LIU`

**Σκοπός:** Node positions (x_km, y_km) για γεωγραφικό Gini.

```python
@dataclass(frozen=True, slots=True)
class SpatialProfile:
    node_id: int; x_km: float; y_km: float

@dataclass(frozen=True, slots=True)
class SpatialProfileSet:
    profiles: tuple[SpatialProfile, ...]
    region_width_km: float
    region_height_km: float
    
    @classmethod
    def from_coordinates(cls, coordinates_km, ...) -> "SpatialProfileSet": ...
```

**Paper:** x ∈ S(t) — PAPER. Η 2D km representation = RECONSTRUCTION.

---

### 4.8 `Liu/Feasibility.py` — `NEW_FOR_LIU`

**Σκοπός:** Immutable αποτελέσματα constraint evaluation — wrapper γύρω από (name, satisfied, measured, threshold).

```python
@dataclass(frozen=True, slots=True)
class ConstraintResult(CanonicalSerializable):
    constraint_name: str
    satisfied: bool
    measured_value: float | None
    threshold_value: float | None
    reason: str | None

@dataclass(frozen=True, slots=True)
class FeasibilityResult(CanonicalSerializable):
    constraint_results: tuple[ConstraintResult, ...]

    @property
    def feasible(self) -> bool:
        return all(r.satisfied for r in self.constraint_results)
```

---

### 4.9 `Liu/Threat.py` — `NEW_FOR_LIU`

**Σκοπός:** Byzantine threat scenario — ποιοι κόμβοι είναι malicious.

```python
@dataclass(frozen=True, slots=True)
class ThreatScenario:
    node_count: int
    malicious_node_ids: tuple[int, ...]

    def malicious_validator_count(self, validator_ids) -> int: ...
```

**Paper:** f = αριθμός faulty validators — PAPER. Η εξωτερική καταγραφή malicious IDs = RECONSTRUCTION.

---

### 4.10 `Liu/LinkFSMC.py` — `NEW_FOR_LIU`

**Σκοπός:** Πλήρης Finite-State Markov Chain implementation για το dynamic network channel.

**Κλάσεις:**

| Κλάση | Σκοπός |
|-------|--------|
| `LinkRateLevels` | Ordered tuple ρυθμών: (10, 55, 100) Mbps |
| `LinkTransitionMatrix` | L×L row-stochastic πίνακας (diagonal=0.70) |
| `LinkTransitionTensor` | N×N πίνακες (shared ή per-link) |
| `LinkFSMCState` | Τρέχουσα κατάσταση + advance() |

**Βασικές μέθοδοι:**

```python
class LinkFSMCState:
    def advance(rng) -> "LinkFSMCState":
        # Κάθε link i→j: sample νέο επίπεδο από current_level
        return self.with_current_links(self.transition(rng))

class LinkTransitionMatrix:
    def sample_next_level(current_level, rng) -> int:
        # CDF sampling: Σωρευτική πιθανότητα με uniform draw
```

**Paper:** FSMC model — PAPER (Liu et al. §IV.C). Η CDF sampling implementation = RECONSTRUCTION.

**Παράδειγμα:** Link σε επίπεδο 1 (55 Mbps), draw=0.82 > cumulative(0+1)=0.85 → μένει 55 Mbps.

**Tests:** `tests/test_liu_geography_fsmc.py`

---

### 4.11 `Liu/ReferenceCore.py` — `NEW_FOR_LIU`

**Σκοπός:** Pure paper-equation constraint/reward evaluation χωρίς DES. Χρησιμοποιείται για ανεξάρτητη επαλήθευση των DES αποτελεσμάτων.

**Βασικές συναρτήσεις:**

```python
def paper_stake_gini(producer_stakes) -> float:
    """Eq. (2): G(Υ) pairwise Gini."""
    return canonical_pairwise_gini(stakes)

@dataclass(frozen=True, slots=True)
class GeographicGiniConfig:
    x0, x1, y0, y1: float    # Region bounds
    resolution: int = 64      # Grid cells per axis
    convergence_tol: float = 1e-3

def _geographic_gini_at(lambda_fn, config, resolution) -> float:
    """Eq. (3): Numerical integration of geographic Gini."""
    # Grid-based discrete evaluation: Σ_i Σ_j |λ(x_i)-λ(y_j)| dy dx / ...
```

**Paper:** Eq. 2-5 για C1, Eq. 13 για ανταμοιβή — PAPER. Grid integration = RECONSTRUCTION (paper uses continuous integral).

**Tests:** `tests/test_liu_reference_core.py`

---

### 4.12 `Liu/AnalyticalConsensus.py` + `AnalyticalPBFT.py` + `AnalyticalZyzzyva.py` + `AnalyticalLiuQuorum.py` — `NEW_FOR_LIU`

**Σκοπός:** Pure analytical (closed-form) computation του consensus latency T_C βάσει των εξισώσεων Appendix B του paper. Δεν αγγίζει DES — αποτελεί ανεξάρτητο oracle για paper equations vs DES comparison.

```python
class AnalyticalConsensusInput:
    """Όλα τα inputs για τις Eq. 15-17."""
    validator_ids, validator_count_k, protocol, ...
    block_size_mb, transaction_size_bytes, batch_size_m, ...
    computational_capabilities_ghz, directed_link_rates_mbps, ...
    signature_verification_cycles_alpha, mac_operation_cycles_beta, ...

class LiuTransmissionUnitPolicy(Enum):
    PAPER_LITERAL_MB_PER_MBPS_V1 = ...   # Paper literal: S_B/R (no bit conversion)
    SI_MEGABYTE_TO_MEGABIT_V1 = ...       # SI: 8*S_B/R (MByte → Mbit)
```

**Σημαντική αβεβαιότητα:** Το paper χρησιμοποιεί S_B/R αλλά δεν ξεκαθαρίζει αν S_B σε MB και R σε Mbps απαιτούν μετατροπή (×8). Η `LiuTransmissionUnitPolicy` εκφράζει ρητά αυτή την ambiguity.

**Paper:** Eq. 15-17 — PAPER. Unit policy decision = PAPER_UNDERSPECIFIED (ρητά documented).

**Tests:** `tests/test_liu_reference_core.py`

---

### 4.13 `Liu/ActionCandidates.py` — `NEW_FOR_LIU`

**Σκοπός:** Γεννήτρια όλων των valid LiuAction combinations. Υπολογίζει C(N,K) × |protocols| × |S_B values| × |T_I values|.

**Χρησιμοποιείται για:** Δημιουργία του `reduced_action_domain.csv` (3915 ενέργειες για N=30, K=28).

---

### 4.14 Λοιπά Liu/ αρχεία — `NEW_FOR_LIU`

| Αρχείο | Σκοπός |
|--------|--------|
| `Liu/EpochConfiguration.py` | `EpochConfiguration` dataclass (validator_set, protocol, S_B, T_I) |
| `Liu/SpatialIntensity.py` | `GridSpatialIntensityModel` — geographic Gini με grid integration |
| `Liu/ContinuousSpatial.py` | `ContinuousSpatialIntensityModel` — smooth spatial density |
| `Liu/DigitalTwinPaired.py` | `LiuDigitalTwinPairedEvaluation` — σύγκριση paper vs DES |
| `Liu/ExperimentManifest.py` | `ExperimentManifest` — metadata για experiment reproducibility |
| `Liu/DQN.py` | Πρώιμο DQN prototype (superseded από experiments/drl/) |
| `Liu/DQNTraining.py` | Πρώιμο training script (superseded) |
| `Liu/__init__.py` | Package exports (152 γραμμές — comprehensive public API) |

---

## 5. Utils/

### 5.1 `Utils/ComputationalDelay.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Utils/ComputationalDelay.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Κλιμάκωση processing delay βάσει GHz capability.

```python
REFERENCE_CAPABILITY_GHZ = 20.0

def scale(base_delay: float, ghz: float) -> float:
    """Inverse scaling: ισχυρότερος κόμβος → μικρότερος χρόνος."""
    return base_delay * REFERENCE_CAPABILITY_GHZ / ghz
```

**Inputs:** `base_delay` (s σε reference 20 GHz), `ghz` (GHz του κόμβου).  
**Outputs:** Scaled delay (s).

**Liu concept:** Processing delay ∝ 1/c_i — PAPER (Appendix B). Η αναφορά σε 20 GHz = RECONSTRUCTION.

**Παράδειγμα:** base_delay=0.01s, ghz=5.0: `0.01 * 20/5 = 0.04 s`.

**Καλείται από:** `Chain/Network.py` (για classic nodes με NodeProfile)  
**Tests:** `tests/test_liu_node_validator.py` (έμμεσα)

---

### 5.2 `Utils/DecentralizationMetrics.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Utils/DecentralizationMetrics.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Canonical pairwise Gini coefficient — βασικός δείκτης για το C1 constraint.

```python
def canonical_pairwise_gini(values: list[float]) -> float:
    """
    G = Σ_i Σ_j |x_i - x_j| / (2 * n * Σ_i x_i)
    
    Range: [0, 1] — 0=ομοιόμορφη κατανομή, 1=πλήρης συγκέντρωση
    """
    n = len(values)
    total = sum(values)
    if total == 0: return 0.0
    numerator = sum(abs(xi - xj) for xi in values for xj in values)
    return numerator / (2 * n * total)
```

**Inputs:** `list[float]` τιμών (stakes ή geographic density).  
**Outputs:** Gini coefficient ∈ [0,1].

**Paper:** Eq. 2 G(Υ) — PAPER formula. Python implementation = RECONSTRUCTION.

**Παράδειγμα:** stakes=[1,1,1]: G=0. stakes=[0,0,10]: G≈0.667.

**Καλείται από:** `RuntimeEvaluation.LiuRuntimeConstraintEvaluator`, `ReferenceCore.paper_stake_gini()`, `StateBuilder.LiuRuntimeStateBuilder`  
**Tests:** `tests/test_liu_reference_core.py`

---

### 5.3 `Utils/Instrumentation.py` — `EXISTING_MODIFIED`

**Path:** `src/Simulator/Utils/Instrumentation.py`

**Κατάταξη:** Το αρχείο υπήρχε (αρχικά μικρό), επεκτάθηκε σημαντικά στο 8511e7e.

**Σκοπός:** Append-only records για την κλασική SymBChainSim instrumentation (transactions, blocks, consensus decisions). Χρησιμοποιείται παράλληλα με το `LiuRuntimeInstrumentation`.

**Βασικά records:**

```python
@dataclass(frozen=True, slots=True)
class TransactionCreationRecord:
    transaction_id: int; creator_node_id: int
    original_creation_time: float; transaction_size: float

class BlockProposalRecord:
    block_id: int; proposer: int; consensus_protocol: str
    proposal_time: float; block_size: float; transaction_count: int

class LocalConsensusDecisionRecord:
    block_id: int; node_id: int; decision_time: float
    decision_path: str; quorum_size: int

class BlockObservationRecord:
    block_id: int; node_id: int; observation_time: float; cause: str

class InstrumentationCollector:
    transactions: list; block_proposals: list; decisions: list; observations: list
```

**Καλείται από:** Κλασικά πρωτόκολλα (PBFT, Tendermint, BigFoot) και LiuRuntime (για cross-reference)

---

### 5.4 `Utils/InstrumentationMetrics.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Utils/InstrumentationMetrics.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Aggregation layer — μετατρέπει raw instrumentation records σε research metrics (distributions, first quorum time, throughput κ.λπ.).

**Βασικά types:**

```python
@dataclass(frozen=True, slots=True)
class DistributionSummary:
    count: int; mean: float; median: float
    minimum: float; maximum: float; standard_deviation: float

@dataclass(frozen=True, slots=True)
class FirstQuorumEvidence:
    logical_block_key: LogicalBlockKey
    decision_time: float; node_id: int; decision_path: str

@dataclass(frozen=True, slots=True)
class FinalizedBlockMetric:
    logical_block_key: LogicalBlockKey
    # + DistributionSummary για decision times
```

**Σκοπός ανάλυσης:** Υπολογισμός throughput, latency distributions, finality time per block — για reports.

---

### 5.5 `Utils/LiuRuntimeInstrumentation.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Utils/LiuRuntimeInstrumentation.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Append-only records αποκλειστικά για τα Liu runtime protocols. **Βασικό output** της DES εκτέλεσης.

**Κρίσιμα records:**

```python
@dataclass(frozen=True, slots=True)
class ProtocolMessageRecord:
    # Κάθε μήνυμα που στάλθηκε: sender, receiver, sent_at, arrival_at, rate_mbps, payload_size_mb

@dataclass(frozen=True, slots=True)
class ProtocolPhaseTransitionRecord:
    # Κάθε αλλαγή φάσης: previous_phase→new_phase, processing times

@dataclass(frozen=True, slots=True)
class ProtocolCertificateRecord:
    # Κάθε certificate που δημιουργήθηκε: signer_ids, threshold, creation_time

@dataclass(frozen=True, slots=True)
class ProtocolFinalityRecord:  # ← ΤΟ ΠΙΑΟ ΚΡΙΣΙΜΟ
    protocol: str; epoch_id: int; height: int
    block_digest: str; parent_digest: str
    client_id: int
    request_sent_at: float   # ← T_request
    finality_time: float     # ← T_finality
    finality_path: str       # "fast_path" | "recovery_path" | "quorum_all_replies"
    certificate_hash: str

    @property
    def consensus_latency_s(self) -> float:
        return self.finality_time - self.request_sent_at  # T_C

@dataclass(frozen=True, slots=True)
class ProtocolViewChangeRecord:
    from_view: int; target_view: int; event_type: str; status: str

@dataclass(frozen=True, slots=True)
class ProtocolFailureRecord:
    phase: str; reason: str; event_time: float

@dataclass(frozen=True, slots=True)
class ProtocolTimeoutAdaptationRecord:
    consecutive_timeout_view_changes: int; effective_timeout_s: float
    backoff_factor: float

@dataclass(frozen=True, slots=True)
class QuorumRequestTimeoutRecord:
    estimated_round_trip_s: float; scheduled_timeout_s: float
    safety_factor: float

class LiuRuntimeInstrumentationCollector:
    # Class-level singleton (shared across all nodes in one DES run)
    protocol_messages: list[ProtocolMessageRecord]
    phase_transitions: list[ProtocolPhaseTransitionRecord]
    certificates: list[ProtocolCertificateRecord]
    finality_records: list[ProtocolFinalityRecord]
    view_changes: list[ProtocolViewChangeRecord]
    protocol_failures: list[ProtocolFailureRecord]
```

**Σημαντικό:** `ProtocolFinalityRecord` είναι η κύρια έξοδος — από αυτό παράγεται το `t_c_des` και η αξιολόγηση C2.

**Καλείται από:** Όλα τα LiuRuntime Transitions (append records)  
**Καλείται από:** `RuntimeEvaluation.LiuRuntimeExecutionMeasurement.from_runtime()` (reads records)  
**Tests:** `tests/test_instrumentation.py`

---

### 5.6 `Utils/LiuEnvironmentInstrumentation.py` — `NEW_FOR_LIU`

**Path:** `src/Simulator/Utils/LiuEnvironmentInstrumentation.py`

**Κατάταξη:** NEW_FOR_LIU (8511e7e)

**Σκοπός:** Per-environment (per RL episode) instrumentation — καταγράφει reset/step/evaluation events.

```python
@dataclass(frozen=True, slots=True)
class LiuEnvironmentEventRecord:
    event_type: str; episode_id: int; decision_step: int; event_time: float
    state_hash: str | None; action_hash: str | None
    evaluation_hash: str | None; status: str | None

class LiuEnvironmentInstrumentationCollector:
    records: list[LiuEnvironmentEventRecord]
    def reset(); def append(record); def snapshot(); def deterministic_hash()
```

**Διαφορά από LiuRuntimeInstrumentation:** Το `LiuRuntimeInstrumentation` είναι class-level singleton (μοιράζεται σε όλους τους κόμβους μιας DES εκτέλεσης). Το `LiuEnvironmentInstrumentation` είναι instance-level (ένα per RL environment).

---

### 5.7 `Utils/Metrics.py` — `EXISTING_MODIFIED`

**Κατάταξη:** EXISTING_MODIFIED (dc9bc4d, +5 γραμμές)

**Σκοπός:** Κλασικές SymBChainSim metrics (throughput, latency) για τα original πρωτόκολλα. Μικρές τροποποιήσεις για συμβατότητα.

---

### 5.8 `Utils/Serialise.py`, `Utils/Tools.py`, `Utils/Snapshots.py` — `EXISTING_UNCHANGED`

**Κατάταξη:** EXISTING_UNCHANGED

| Αρχείο | Σκοπός |
|--------|--------|
| `Serialise.py` | JSON serialization helpers για κλασικά objects |
| `Tools.py` | Utility functions (logging, formatting) |
| `Snapshots.py` | Node/network state snapshots για analysis |

Δεν τροποποιήθηκαν για Liu. Χρησιμοποιούνται μόνο από τα original πρωτόκολλα.

---

## 6. End-to-end dependency/call graph

Αυτό το graph καλύπτει **αποκλειστικά** το `src/Simulator/` — χωρίς experiments/.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINT (experiments/)                        │
│           LiuDynamicEpochEnv.step(action: dict)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Liu/Action.py                                                           │
│  LiuAction(node_count, validator_count, validator_ids,                   │
│            consensus_protocol, block_size_mb, block_interval_s)         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Chain/Consensus/LiuRuntime/Common/SingleActionRuntime.py                │
│  run_single_action(action, snapshot, omega=6.0, ...)                    │
│  ├── LiuRuntimeStateBuilder.build()      → LiuState S0                  │
│  ├── LiuRuntimeActionApplicator.apply()  → LiuRuntimeEpochContext       │
│  ├── Network.setup(nodes, epoch_context)                                 │
│  ├── LiuRuntimeProtocolFactory.create()  → LiuPBFT|LiuZyzzyva|Quorum   │
│  └── _run_to_height(sim, target_height, C2_deadline)                    │
│      ↑ ① DES LOOP                                                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ① DES ENGINE (Engine/Simulation.py + EventQueue.py + Handler.py)       │
│                                                                          │
│  sim.sim_next_event()                                                    │
│  ├── EventQueue.Queue.pop()    → event (min time)                        │
│  ├── self.clock = event.time                                             │
│  └── Handler.handle_event(event)                                         │
│      ├── Check actor.state.alive                                         │
│      ├── Check liu_context.epoch_id == current_epoch.epoch_id            │
│      └── event.handler(event)  → dispatches to protocol                 │
│                                                                          │
│  Repeat until: ProtocolFinalityRecord found                              │
│            OR: sim.clock > C2_deadline → C2DeadlineExceeded             │
│            OR: queue empty → DESQueueExhausted                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  event.handler()
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PROTOCOL EXECUTION (one of three):                                      │
│                                                                          │
│  ② LiuPBFT/Protocol.py + Transitions.py                                 │
│     CLIENT → REQUEST → PRIMARY → PRE-PREPARE → BACKUPS                  │
│     → PREPARE(2f+1) → PBFTPreparedCertificate                           │
│     → COMMIT(2f+1)  → PBFTCommitCertificate → FINALIZED                 │
│     [timeout] → view-change → [cap] → DESQueueExhausted                 │
│                                                                          │
│  ③ LiuZyzzyva/Protocol.py + Transitions.py                              │
│     CLIENT → REQUEST → PRIMARY → ORDER → BACKUPS                        │
│     → REPLY(K all) → fast path → FINALIZED (fast)                       │
│     → REPLY(2f+1<K) → COMMIT → 2f+1 → FINALIZED (recovery)             │
│                                                                          │
│  ④ LiuQuorum/Protocol.py + Transitions.py                               │
│     CLIENT → [T_I delay] → REQUEST → ALL_REPLICAS                       │
│     → REPLY(K-1 all) → LiuQuorumReplyCertificate → FINALIZED           │
│     [timeout] → DES expires → C2DeadlineExceeded                        │
│                                                                          │
│  All protocols use:                                                      │
│  ├── Transport/NetworkAdapter.LiuDirectedTransport.send()               │
│  │   └── R_{ij} from EpochContext.link_state_matrix                     │
│  │   └── Creates MessageEvent → EventQueue                              │
│  ├── Processing/ProcessingCostModel.duration_s()                        │
│  │   └── cycles(LiuProcessingWork) / (c_i GHz * 1e9)                   │
│  ├── Processing/NodeComputeQueue.reserve()                               │
│  │   └── Serializes CPU work → schedules completion event               │
│  └── Common/Certificates.py (PBFTPrepared, Commit, Zyzzyva, Quorum)    │
│                                                                          │
│  All transitions record to:                                              │
│  └── Utils/LiuRuntimeInstrumentation.LiuRuntimeInstrumentationCollector │
│      ├── protocol_messages (every send)                                  │
│      ├── phase_transitions (every state change)                         │
│      ├── certificates (every quorum)                                    │
│      ├── finality_records ← ProtocolFinalityRecord (TARGET)             │
│      └── view_changes (PBFT only)                                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  ProtocolFinalityRecord
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Common/RuntimeEvaluation.py                                             │
│  LiuRuntimeConstraintEvaluator.evaluate(state, action, measurement)     │
│                                                                          │
│  C1: canonical_pairwise_gini(selected_stakes) ≤ η_s=0.20               │
│      canonical_pairwise_gini(selected_positions) ≤ η_l=0.30            │
│      ↑ Utils/DecentralizationMetrics.canonical_pairwise_gini()          │
│                                                                          │
│  C2: t_c_des = finality_time - request_sent_at                          │
│      t_c_des ≤ (omega-1) * T_I  (i.e. T_F = T_I+T_C ≤ omega*T_I)      │
│                                                                          │
│  C3: faulty_validators ≤ tolerated_faults(protocol, K)                  │
│      Quorum→0, PBFT/Zyzzyva→floor((K-1)/3)                             │
│                                                                          │
│  Reward Ω = floor(S_B*1e6/chi)/T_I  if C1∧C2∧C3  else 0               │
│                                                                          │
│  → LiuRuntimeEpochEvaluation(reward, status, c1, c2, c3, t_c_des)      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Common/StateEvolution.py                                                │
│  advance_after_epoch(action, finality_record)                           │
│  ├── Liu/LinkFSMC.LinkFSMCState.advance(rng)  → νέο link matrix R_{t+1} │
│  └── Common/StateBuilder.LiuRuntimeStateBuilder.build()                 │
│      → LiuState S_{t+1} (χ, Υ, x, c, R_{t+1})                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RETURN TO experiments/                                                  │
│  StepResult(reward, des_c1, des_c2, des_c3, t_c_des, next_state, ...)  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Συνοπτική απεικόνιση εξαρτήσεων

```
LiuAction
    └──▶ SingleActionRuntime
              ├──▶ Engine.Simulation (DES loop)
              │         └──▶ Engine.Handler
              │                   └──▶ {LiuPBFT | LiuZyzzyva | LiuQuorum}.handle_event
              │                             ├──▶ Transport.LiuDirectedTransport → Engine.Event
              │                             ├──▶ Processing.LiuProcessingCostModel
              │                             ├──▶ Common.Certificates
              │                             └──▶ Utils.LiuRuntimeInstrumentationCollector
              │                                         └──▶ ProtocolFinalityRecord ──▶
              ├──▶ Common.RuntimeEvaluation
              │         ├──▶ Utils.DecentralizationMetrics.canonical_pairwise_gini
              │         └──▶ LiuRuntimeEpochEvaluation (reward Ω, C1/C2/C3)
              └──▶ Common.StateEvolution
                        ├──▶ Liu.LinkFSMC.advance → R_{t+1}
                        └──▶ Common.StateBuilder → LiuState S_{t+1}
```

---

*Τέλος εγγράφου. Όλες οι κατατάξεις επαληθεύθηκαν από `git log` ανά αρχείο.*
