"""Fig. 5 (convergence) from runs/fig5/*.csv: ours (DES) vs the paper's relative levels.

Left: moving-average throughput per episode (full length of each run). Right: plateau level of each scheme
over a COMMON window (the last 1000 episodes of the shortest run, so every scheme has seen the same number
of episodes) relative to the proposed scheme, ours vs the paper (levels read off the printed Fig. 5:
proposed ~2.1e5, w/o CAS ~1.65e5, fixed size ~1.25e5, fixed interval ~0.95e5 TPS).
    python plot_fig5.py [runs/fig5] [out.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCHEMES = ["proposed", "no_cas", "fixed_block_size", "fixed_block_interval"]
LABEL = {"proposed": "Proposed scheme", "no_cas": "w/o CAS", "fixed_block_size": "Fixed block size (4 MB)",
         "fixed_block_interval": "Fixed block interval (1 s)"}
PAPER = {"proposed": 2.10e5, "no_cas": 1.65e5, "fixed_block_size": 1.25e5, "fixed_block_interval": 0.95e5}
W = 200


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fig5")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src / "fig5.png")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    runs = {s: np.array([float(r["reward"]) for r in csv.DictReader(open(src / f"{s}_seed42.csv"))]) for s in SCHEMES}
    horizon = min(len(v) for v in runs.values())      # common horizon: runs were stopped at equal wall-clock, not equal episodes
    plateau = {s: v[horizon - 1000:horizon].mean() for s, v in runs.items()}
    for k, s in enumerate(SCHEMES):
        rw = runs[s]
        ma = np.convolve(rw, np.ones(W) / W, mode="valid")
        a0.plot(np.arange(W - 1, len(rw)), ma, color=f"C{k}", label=f"{LABEL[s]} ({len(rw)} ep.)")
    a0.axvspan(horizon - 1000, horizon, color="grey", alpha=.12)
    a0.axvline(4000, color="grey", ls=":", lw=1)
    a0.text(4000, a0.get_ylim()[1] * 0.02, " paper: converged ~4000", fontsize=7, color="grey")
    a0.set(xlabel="Episode (1 episode = 1 DES decision epoch)", ylabel=f"Throughput (TPS), {W}-episode moving avg -- SymBChainSim DES",
           title="Fig. 5: convergence (ours)")
    a0.legend(fontsize=7, loc="lower right"); a0.grid(alpha=.3)
    x = np.arange(len(SCHEMES))
    a1.bar(x - 0.2, [plateau[s] / plateau["proposed"] for s in SCHEMES], 0.4, label="ours (DES)")
    a1.bar(x + 0.2, [PAPER[s] / PAPER["proposed"] for s in SCHEMES], 0.4, label="paper (read off figure)", alpha=.6)
    a1.set_xticks(x, ["proposed", "w/o CAS", "fixed\nsize", "fixed\ninterval"])
    a1.set(ylabel="plateau relative to proposed", title=f"Plateau, episodes {horizon-1000}-{horizon}"); a1.legend(fontsize=7); a1.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out, {s: round(v) for s, v in plateau.items()})


if __name__ == "__main__":
    main()
