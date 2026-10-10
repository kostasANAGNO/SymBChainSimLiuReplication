#!/usr/bin/env bash
# Run all Fig. 7-10 jobs at N=100/K=21 (4 figs x 5 schemes = 20 jobs), P in parallel. Resumable.
#   ./run_sweeps.sh [parallel=4] [episodes=1000] [eval_epochs=40]
cd "$(dirname "$0")/../../.."
P=${1:-4}; EP=${2:-1000}; EV=${3:-40}
for fig in fig7 fig8 fig9 fig10; do for s in proposed no_cas fixed_block_size fixed_block_interval static; do echo "$fig $s"; done; done |
  xargs -P "$P" -L1 bash -c 'uv run python experiments/liu_replication/paper_figures/sweeps_n100.py --fig $0 --scheme $1 --episodes '"$EP"' --eval-epochs '"$EV"' --out experiments/liu_replication/paper_figures/runs/sweeps_n100 >> experiments/liu_replication/paper_figures/runs/sweeps_n100/$0_$1.out 2>&1'
