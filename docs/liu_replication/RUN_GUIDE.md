# Liu Replication — Run Guide

## Requirements

- Python 3.13+ (3.14 confirmed working)
- PyTorch, NumPy, PyYAML (installed via pip into the project `.venv`)

## Setup

From the repository root, install the project in editable mode:

```bash
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

This registers the simulator packages (`Liu`, `Chain`, `Engine`, `Utils`, `Manager`)
via a `.pth` file in site-packages, so the interpreter resolves them without any
manual PYTHONPATH or IDE source-root configuration.

If starting from a fresh environment (no `.venv`):

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # or: uv sync
.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## PyCharm Setup

| Setting | Value |
|---------|-------|
| Interpreter | `.venv\Scripts\python.exe` (project-root-relative) |
| Working directory | Repository root (`SymBChainSim/`) |

**No manual Sources Root configuration is needed.** The editable install places
`src/Simulator` on `sys.path` via the installed `.pth` file, so PyCharm resolves
`from Liu.* import ...` and `from Chain.* import ...` through the interpreter alone.

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
