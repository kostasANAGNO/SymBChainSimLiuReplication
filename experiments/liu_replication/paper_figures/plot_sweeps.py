"""Plot Fig. 7-10 from runs/sweeps/*.jsonl: ours (DES) vs the paper's curve endpoints.

Left panel: our absolute values. Right panel: values normalised to the first sweep point, for ours and
for the paper's start/end points, so the *shape* is comparable despite the different absolute scale.
    python plot_sweeps.py [runs/sweeps] [runs/sweeps/plots]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMES = ["proposed", "no_cas", "fixed_block_size", "fixed_block_interval", "static"]
LABEL = {"proposed": "Proposed scheme", "no_cas": "w/o CAS", "fixed_block_size": "Fixed block size",
         "fixed_block_interval": "Fixed block interval", "static": "Existing static scheme"}
XLAB = {"fig7": "Threshold of TTF ($\\omega$ block intervals)", "fig8": "Average transaction size (B)",
        "fig9": "Average computational resources (GHz)", "fig10": "Block size limit (MB)"}
METRIC = {"fig7": "throughput", "fig8": "throughput", "fig9": "ttf", "fig10": "throughput"}
YLAB = {"throughput": "Throughput (TPS)", "ttf": "Average TTF (s)"}


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sweeps")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else src / "plots")
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((Path(__file__).parent / "paper_reference.json").read_text())
    for fig, m in METRIC.items():
        fig_, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
        any_data = False
        for k, sch in enumerate(SCHEMES):
            p = src / f"{fig}_{sch}.jsonl"
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines()]
            rows = [r for r in rows if r[m] is not None]
            if not rows:
                continue
            any_data = True
            xs, ys = [r["value"] for r in rows], [r[m] for r in rows]
            c = f"C{k}"
            a0.plot(xs, ys, "o-", color=c, label=LABEL[sch])
            a1.plot(xs, [y / ys[0] for y in ys], "o-", color=c, label=f"{LABEL[sch]} (ours)")
            rx, ry = ref[fig]["x"], ref[fig][sch]
            a1.plot(rx, [y / ry[0] for y in ry], "s--", color=c, alpha=0.6, ms=7)
        if not any_data:
            plt.close(fig_)
            continue
        a0.set(xlabel=XLAB[fig], ylabel=YLAB[m] + " -- SymBChainSim DES", title=f"{fig}: ours (absolute)")
        a1.set(xlabel=XLAB[fig], ylabel="normalised to first point", title="shape: ours (solid) vs paper endpoints (dashed squares)")
        a0.legend(fontsize=7); a0.grid(alpha=.3); a1.grid(alpha=.3)
        fig_.tight_layout()
        fig_.savefig(out / f"{fig}.png", dpi=130)
        plt.close(fig_)
        print("wrote", out / f"{fig}.png")


if __name__ == "__main__":
    main()
