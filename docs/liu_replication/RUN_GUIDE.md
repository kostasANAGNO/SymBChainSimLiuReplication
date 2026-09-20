# Liu Replication — Run Guide

## Requirements

- Python 3.13+ (3.14 confirmed working)
- PyTorch, NumPy, PyYAML (managed via `uv` or the project `.venv`)

## Setup

```bash
# From repository root — install dependencies into .venv
.venv\Scripts\python.exe -m pip install -e .   # or: uv sync
```

The project ships a `.venv` with all dependencies installed. Activate it or
point your IDE interpreter at `.venv\Scripts\python.exe`.

## PyCharm Setup

| Setting | Value |
|---------|-------|
| Interpreter | `.venv\Scripts\python.exe` (project-root-relative) |
| Sources Root | Mark `src/Simulator` as a **Sources Root** (right-click → Mark Directory as) |
| Working directory | Repository root (`SymBChainSim/`) |

With `src/Simulator` marked as Sources Root, PyCharm resolves `from Liu.* import ...`
correctly and provides full code completion. The `pythonpath = ["src/Simulator"]` entry
in `pyproject.toml` covers the same for pytest's test-discovery.

## Running Tests

From repository root:

```bash
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: **62+ tests, all pass, under 30 s.**

## Single-Decision Demo (Supervisor Presentation)

```bash
.venv\Scripts\python.exe experiments/liu_replication/demo_single_decision.py
```

Loads the final ep400 DQN checkpoint, runs one real SymBChainSim DES step, and
prints a formatted terminal output showing the state, DQN decision, DES execution
result, constraint check, and Liu reward.

With comparison mode:

```bash
.venv\Scripts\python.exe experiments/liu_replication/demo_single_decision.py --compare
```

Executes the same starting state for DQN greedy, random, and strong-fixed policies
and prints a side-by-side table.

## Final Policy Evaluation (320-step, ~71 min)

```bash
.venv\Scripts\python.exe experiments/liu_replication/drl/n30/eval_n30_seed42.py
```

Evaluates ep400 and ep350 checkpoints against random and strong-fixed baselines over
16 trajectories × 20 steps. Outputs:
- `experiments/liu_replication/drl/n30/n30_seed42_policy_evaluation.json`
- `experiments/liu_replication/drl/n30/n30_seed42_policy_evaluation.md`

## Checkpoint Location

```
experiments/liu_replication/drl/n30/checkpoints/ckpt_seed42_clean_ep400.pt
```

`.pt` files are excluded from git by `.gitignore`. Copy the checkpoint if moving
the repository to a new machine.

## Output Locations

| Script | Output |
|--------|--------|
| demo_single_decision.py | Terminal only |
| eval_n30_seed42.py | `drl/n30/n30_seed42_policy_evaluation.json` |
| train_n30.py | `drl/n30/episode_log_seed42_clean.csv`, checkpoints/ |
| controlled_sweep | `controlled_sweep/controlled_full_results.csv` |

## Notes

- All experiment scripts add their own `sys.path` entries relative to `__file__`.
  They work correctly from any working directory as long as the file is in the
  right location within the repository.
- `.idea/` is in `.gitignore`; do not commit PyCharm project files.
- Checkpoint `.pt` files are excluded from git; they are large and machine-specific.
