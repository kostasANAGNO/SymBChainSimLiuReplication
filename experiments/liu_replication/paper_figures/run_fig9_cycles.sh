#!/usr/bin/env bash
# Fig. 9 (TTF vs compute) at N=100/K=21, hi links, with the verification cost model varied (alpha/beta sensitivity).
#   ./run_fig9_cycles.sh <block|per_tx> <scale> <tag> [parallel=4] [episodes=300] [eval_epochs=40]
# Output: runs/fig9_cycles/<tag>/fig9_<scheme>.jsonl  (resumable)
cd "$(dirname "$0")/../../.."
MODE=$1; SCALE=$2; TAG=$3; P=${4:-4}; EP=${5:-300}; EV=${6:-40}
OUT=experiments/liu_replication/paper_figures/runs/fig9_cycles/$TAG
mkdir -p "$OUT"
for s in proposed no_cas fixed_block_size fixed_block_interval static; do echo "$s"; done |
  xargs -P "$P" -L1 bash -c 'uv run python experiments/liu_replication/paper_figures/sweeps_n100.py --fig fig9 --scheme $0 --episodes '"$EP"' --eval-epochs '"$EV"' --link-profile hi --cycle-mode '"$MODE"' --cycle-scale '"$SCALE"' --out '"$OUT"' >> '"$OUT"'/fig9_$0.out 2>&1'
