#!/usr/bin/env bash
# Fig. 7-10 at N=100/K=21 (4 figs x 5 schemes = 20 jobs), P in parallel. Resumable.
#   ./run_sweeps_n100.sh [default|hi] [parallel=4] [episodes=300] [eval_epochs=40]
# default -> runs/sweeps_n100 (FSMC 10/55/100 Mbps), hi -> runs/sweeps_n100_hi (FSMC 55/100 Mbps).
cd "$(dirname "$0")/../../.."
PROFILE=${1:-default}; P=${2:-4}; EP=${3:-300}; EV=${4:-40}
OUT=experiments/liu_replication/paper_figures/runs/sweeps_n100; [ "$PROFILE" = hi ] && OUT=${OUT}_hi
mkdir -p "$OUT"
for fig in fig7 fig8 fig9 fig10; do for s in proposed no_cas fixed_block_size fixed_block_interval static; do echo "$fig $s"; done; done |
  xargs -P "$P" -L1 bash -c 'uv run python experiments/liu_replication/paper_figures/sweeps_n100.py --fig $0 --scheme $1 --episodes '"$EP"' --eval-epochs '"$EV"' --link-profile '"$PROFILE"' --out '"$OUT"' >> '"$OUT"'/$0_$1.out 2>&1'
