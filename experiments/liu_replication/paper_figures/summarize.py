"""Paper-vs-DES summary: endpoints of every sweep curve and Fig. 5 plateaus, default and hi links.
    python summarize.py > SUMMARY_TABLE.md"""
import csv, json
from pathlib import Path
import numpy as np

REF = json.load(open("paper_reference.json"))
SCH = ["proposed", "no_cas", "fixed_block_size", "fixed_block_interval", "static"]
FIG = {"fig7": ("throughput", "ω"), "fig8": ("throughput", "χ"), "fig9": ("ttf", "GHz"), "fig10": ("throughput", "Ṡ")}

def curve(d, fig, s):
    f = Path(d) / f"{fig}_{s}.jsonl"
    if not f.exists(): return {}
    return {float(r["value"]): r for r in map(json.loads, open(f))}

def f(x, k): return "n/a" if x is None else (f"{x:.2f}" if k == "ttf" else f"{x/1e3:.1f}k")

print("## Sweeps (endpoints; ours = DES, paper = read off figures)\n")
for fig, (key, xl) in FIG.items():
    x0, x1 = REF[fig]["x"]
    print(f"### {fig} ({key}, x: {x0}→{x1} {xl})\n")
    print("| scheme | paper | DES default links | DES hi links |\n|---|---|---|---|")
    for s in SCH:
        p = REF[fig].get(s)
        ps = f"{f(p[0],key)} → {f(p[1],key)}" if p else "-"
        cells = []
        for d in ("runs/sweeps_n100", "runs/sweeps_n100_hi"):
            c = curve(d, fig, s)
            a, b = c.get(float(x0)), c.get(float(x1))
            g = lambda r: None if r is None or r[key] is None else r[key]
            cells.append(f"{f(g(a),key)} → {f(g(b),key)}")
        print(f"| {s} | {ps} | {cells[0]} | {cells[1]} |")
    print()

print("## Fig. 5 plateau (common last-1000-episode window), TPS, relative to proposed\n")
P = {"proposed": 2.10e5, "no_cas": 1.65e5, "fixed_block_size": 1.25e5, "fixed_block_interval": 0.95e5}
print("| scheme | paper | rel. | DES default | rel. | DES hi | rel. |\n|---|---|---|---|---|---|---|")
res = {}
for d in ("runs/fig5_n100", "runs/fig5_n100_hi"):
    runs = {s: np.array([float(r["reward"]) for r in csv.DictReader(open(f"{d}/{s}_seed42.csv"))]) for s in P}
    h = min(len(v) for v in runs.values())
    res[d] = {s: v[h-1000:h].mean() for s, v in runs.items()}
for s in P:
    row = f"| {s} | {P[s]/1e3:.0f}k | {P[s]/P['proposed']:.2f} |"
    for d in res: row += f" {res[d][s]/1e3:.1f}k | {res[d][s]/res[d]['proposed']:.2f} |"
    print(row)

print("\n## Fig. 9 alpha/beta sensitivity (hi links): TTF in s at 10 -> 30 GHz, change, and paper\n")
VAR = [("hi baseline (alpha, beta per block)", "runs/sweeps_n100_hi"), ("V1 per_tx x1", "runs/fig9_cycles/pertx_x1"),
       ("V2 block x1000", "runs/fig9_cycles/block_x1000"), ("V3 per_tx x0.1", "runs/fig9_cycles/pertx_x0.1")]
print("| scheme | paper | " + " | ".join(n for n, _ in VAR) + " |\n|---|---|" + "---|" * len(VAR))
for s in SCH:
    p = REF["fig9"][s]
    row = f"| {s} | {p[0]:.1f} → {p[1]:.1f} ({(p[1]-p[0])/p[0]*100:+.0f}%) |"
    for _, d in VAR:
        c = curve(d, "fig9", s)
        a, b = c[10.0]["ttf"], c[30.0]["ttf"]
        row += " n/a |" if a is None or b is None else f" {a:.1f} → {b:.1f} ({(b-a)/a*100:+.0f}%) |"
    print(row)
