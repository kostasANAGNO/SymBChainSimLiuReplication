#!/usr/bin/env bash
# Fig. 7-10 at N=100/K=21 with the high link-rate profile (FSMC 55/100 Mbps). Resumable.
#   ./run_sweeps_n100_hi.sh [parallel=4] [episodes=300] [eval_epochs=40]
cd "$(dirname "$0")/../../.."
P=${1:-4}; EP=${2:-300}; EV=${3:-40}
OUT=experiments/liu_replication/paper_figures/runs/sweeps_n100_hi
mkdir -p "$OUT"
for fig in fig7 fig8 fig9 fig10; do for s in proposed no_cas fixed_block_size fixed_block_interval static; do echo "$fig $s"; done; done |
  xargs -P "$P" -L1 bash -c 'uv run python experiments/liu_replication/paper_figures/sweeps_n100.py --fig $0 --scheme $1 --episodes '"$EP"' --eval-epochs '"$EV"' --link-profile hi --out '"$OUT"' >> '"$OUT"'/$0_$1.out 2>&1'
