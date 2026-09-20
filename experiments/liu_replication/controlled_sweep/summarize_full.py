"""Compute controlled_full_summary.{json,md} from controlled_full_results.csv.

Runs independently of the sweep and is safe to re-run. Implements §8/§9 analysis:
run health, paper-vs-DES C2 2×2, tied best-reward sets, per-protocol stats, largest
paper↔DES discrepancies, and invariant/sanity checks (C1 constant, C3=True, S0/hash).
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

OUT_DIR = Path(__file__).parent
CSV_PATH = OUT_DIR / "controlled_full_results.csv"
JSON_PATH = OUT_DIR / "controlled_full_summary.json"
MD_PATH = OUT_DIR / "controlled_full_summary.md"

_PROTOCOLS = ("PBFT", "ZYZZYVA", "LIU_QUORUM")


def _f(value: str) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _b(value: str) -> bool | None:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    return None


def _load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _action_key(row: dict) -> str:
    return f"{row['protocol']}|S_B={row['block_size_mb']}|T_I={row['block_interval_s']}"


def _find_ties_max(rows: list[dict], numeric_field: str) -> tuple[float, list[dict]]:
    values = [(_f(r[numeric_field]), r) for r in rows]
    values = [(v, r) for v, r in values if v is not None]
    if not values:
        return (math.nan, [])
    best = max(v for v, _ in values)
    tied = [r for v, r in values if v == best]
    return (best, tied)


def _monotonic_t_c_in_sb(rows: list[dict]) -> list[str]:
    """Report actions where increasing S_B decreases T_C_DES at the same T_I (non-monotonic)."""
    non_monotonic = []
    for protocol in _PROTOCOLS:
        by_ti = {}
        for r in rows:
            if r["protocol"] != protocol or r["run_status"] != "SUCCESS":
                continue
            ti = _f(r["block_interval_s"])
            sb = _f(r["block_size_mb"])
            tc = _f(r["des_t_c"])
            if ti is None or sb is None or tc is None:
                continue
            by_ti.setdefault(ti, []).append((sb, tc, r))
        for ti, items in by_ti.items():
            items.sort(key=lambda x: x[0])
            prev_tc = None
            for sb, tc, r in items:
                if prev_tc is not None and tc < prev_tc - 1e-9:
                    non_monotonic.append(
                        f"{protocol} T_I={ti}: T_C_DES dipped from {prev_tc:.4f} to {tc:.4f} at S_B={sb}"
                    )
                prev_tc = tc
    return non_monotonic


def summarize(csv_path: Path = CSV_PATH) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    rows = _load_rows(csv_path)

    # A. Run health
    statuses = Counter(r["run_status"] for r in rows)
    wall = [_f(r["wall_clock_seconds"]) for r in rows]
    wall = [w for w in wall if w is not None]
    total_wall = sum(wall)
    fail_reasons = Counter(r["failure_reason"] for r in rows if r["run_status"] != "SUCCESS")

    successful_rows = [r for r in rows if r["run_status"] == "SUCCESS"]

    # B. Feasibility 2×2
    paper_pass = [_b(r["paper_c2"]) for r in successful_rows]
    des_pass = [_b(r["des_c2"]) for r in successful_rows]
    matrix = {"paper_pass_des_pass": 0, "paper_pass_des_fail": 0,
              "paper_fail_des_pass": 0, "paper_fail_des_fail": 0}
    for pp, dp in zip(paper_pass, des_pass):
        if pp is None or dp is None:
            continue
        key = f"paper_{'pass' if pp else 'fail'}_des_{'pass' if dp else 'fail'}"
        matrix[key] += 1
    disagreement_rows_pfail_dpass = [
        _action_key(r) for r in successful_rows
        if _b(r["paper_c2"]) is False and _b(r["des_c2"]) is True
    ]
    disagreement_rows_ppass_dfail = [
        _action_key(r) for r in successful_rows
        if _b(r["paper_c2"]) is True and _b(r["des_c2"]) is False
    ]

    # C. Best-reward tied SETS
    best_paper_reward, best_paper_ties = _find_ties_max(successful_rows, "reward_paper_reference")
    best_des_reward, best_des_ties = _find_ties_max(successful_rows, "reward_des_liu_objective")
    best_paper_set = {_action_key(r) for r in best_paper_ties}
    best_des_set = {_action_key(r) for r in best_des_ties}
    intersection_set = sorted(best_paper_set & best_des_set)

    # D. Per-protocol
    per_protocol: dict = {}
    for protocol in _PROTOCOLS:
        proto_rows = [r for r in successful_rows if r["protocol"] == protocol]
        des_feasible = [
            r for r in proto_rows
            if _b(r["des_c1"]) and _b(r["des_c2"]) and _b(r["des_c3"])
        ]
        paper_feasible = [
            r for r in proto_rows
            if _b(r["paper_c1"]) and _b(r["paper_c2"]) and _b(r["paper_c3"])
        ]
        best_des_row = max(
            proto_rows,
            key=lambda r: (_f(r["reward_des_liu_objective"]) or -1, -int(r["action_id"])),
            default=None,
        )
        best_paper_reward_p, _ = _find_ties_max(proto_rows, "reward_paper_reference")
        per_protocol[protocol] = {
            "attempted": len(proto_rows),
            "des_feasible": len(des_feasible),
            "paper_feasible": len(paper_feasible),
            "best_des_action": _action_key(best_des_row) if best_des_row else None,
            "best_des_liu_reward": _f(best_des_row["reward_des_liu_objective"]) if best_des_row else None,
            "best_paper_reward": best_paper_reward_p if not math.isnan(best_paper_reward_p) else None,
        }

    # E. Largest discrepancies
    def _abs_delta(r, name):
        v = _f(r[name])
        return abs(v) if v is not None else -math.inf

    def _worst(field):
        if not successful_rows:
            return None
        r = max(successful_rows, key=lambda row: _abs_delta(row, field))
        return {
            "action": _action_key(r),
            "value": _f(r[field]),
        }

    # Only latency deltas remain in the cleaned schema; the Liu objective is identical on
    # both sides so any reward delta is exactly explained by the C1/C2/C3 disagreement matrix.
    largest_deltas = {
        "abs_delta_t_c": _worst("delta_t_c"),
        "abs_delta_t_f": _worst("delta_t_f"),
    }

    # Invariants (§7)
    stakes_hashes = {r["validator_set_hash"] for r in successful_rows if _f(r["block_size_mb"]) == 0.2 and _f(r["block_interval_s"]) == 1.0}
    # Cross-check C1: paper_c1 and des_c1 must be True everywhere; C3 True everywhere (f=0)
    c1_violations = [
        _action_key(r) for r in successful_rows
        if not (_b(r["paper_c1"]) and _b(r["des_c1"]))
    ]
    c3_violations = [
        _action_key(r) for r in successful_rows
        if not (_b(r["paper_c3"]) and _b(r["des_c3"]))
    ]
    seeds = {r["seed"] for r in rows}

    # Sanity (§9)
    monotonicity_flags = _monotonic_t_c_in_sb(successful_rows)
    nan_negative = []
    for r in successful_rows:
        for f_name in ("des_t_c", "des_t_f", "reward_des_liu_objective"):
            v = _f(r[f_name])
            if v is None or (isinstance(v, float) and math.isnan(v)) or v < 0:
                nan_negative.append(f"{_action_key(r)} {f_name}={v}")

    summary = {
        "actions_attempted": len(rows),
        "statuses": dict(statuses),
        "failure_reasons": dict(fail_reasons),
        "total_wall_clock_s": total_wall,
        "mean_wall_clock_s": statistics.fmean(wall) if wall else 0.0,
        "median_wall_clock_s": statistics.median(wall) if wall else 0.0,
        "max_wall_clock_s": max(wall) if wall else 0.0,
        "paper_c2_pass_count": sum(1 for r in successful_rows if _b(r["paper_c2"])),
        "paper_c2_fail_count": sum(1 for r in successful_rows if _b(r["paper_c2"]) is False),
        "des_c2_pass_count": sum(1 for r in successful_rows if _b(r["des_c2"])),
        "des_c2_fail_count": sum(1 for r in successful_rows if _b(r["des_c2"]) is False),
        "c2_disagreement_matrix": matrix,
        "paper_fail_des_pass_actions": disagreement_rows_pfail_dpass,
        "paper_pass_des_fail_actions": disagreement_rows_ppass_dfail,
        "best_paper_reward": best_paper_reward if not math.isnan(best_paper_reward) else None,
        "best_paper_action_set": sorted(best_paper_set),
        "best_paper_action_set_size": len(best_paper_set),
        "best_des_liu_reward": best_des_reward if not math.isnan(best_des_reward) else None,
        "best_des_action_set": sorted(best_des_set),
        "best_des_action_set_size": len(best_des_set),
        "optimal_sets_intersection": intersection_set,
        "optimal_sets_intersect": bool(intersection_set),
        "per_protocol": per_protocol,
        "largest_discrepancies": largest_deltas,
        "invariants": {
            "unique_seeds": sorted(seeds),
            "c1_violations": c1_violations,
            "c3_violations": c3_violations,
            "validator_hashes_at_02_10_only": sorted(stakes_hashes),
        },
        "sanity": {
            "monotonicity_dips": monotonicity_flags,
            "nan_or_negative_metrics": nan_negative,
        },
    }
    return summary


def emit(summary: dict) -> None:
    JSON_PATH.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Full 2400-action controlled sweep — summary\n")
    lines.append(f"Attempted: **{summary['actions_attempted']} / 2400**  \n"
                 f"SUCCESS: **{summary['statuses'].get('SUCCESS', 0)}** · "
                 f"ERROR: {summary['statuses'].get('ERROR', 0)} · "
                 f"TIMEOUT: {summary['statuses'].get('TIMEOUT', 0)} · "
                 f"INVALID: {summary['statuses'].get('INVALID', 0)}\n")
    lines.append(f"Total wall: **{summary['total_wall_clock_s']:.1f} s "
                 f"({summary['total_wall_clock_s']/60:.2f} min)**; "
                 f"mean {summary['mean_wall_clock_s']:.3f} s; "
                 f"median {summary['median_wall_clock_s']:.3f} s; "
                 f"max {summary['max_wall_clock_s']:.3f} s\n")

    lines.append("## C2 paper-vs-DES 2×2\n")
    m = summary["c2_disagreement_matrix"]
    lines.append("| | DES PASS | DES FAIL |")
    lines.append("|---|---|---|")
    lines.append(f"| **Paper PASS** | {m['paper_pass_des_pass']} | {m['paper_pass_des_fail']} |")
    lines.append(f"| **Paper FAIL** | {m['paper_fail_des_pass']} | {m['paper_fail_des_fail']} |")
    lines.append(f"\npaper_c2 pass: {summary['paper_c2_pass_count']}, fail: {summary['paper_c2_fail_count']}  \n"
                 f"des_c2 pass: {summary['des_c2_pass_count']}, fail: {summary['des_c2_fail_count']}  \n"
                 f"paper_fail_des_pass count: {len(summary['paper_fail_des_pass_actions'])}  \n"
                 f"paper_pass_des_fail count: {len(summary['paper_pass_des_fail_actions'])}\n")

    lines.append("## Best-reward tied action sets\n")
    lines.append(f"**Best paper reward:** {summary['best_paper_reward']}  \n"
                 f"Tied set size: {summary['best_paper_action_set_size']}  \n"
                 f"Actions: {summary['best_paper_action_set'][:8]}"
                 f"{' …' if summary['best_paper_action_set_size'] > 8 else ''}\n")
    lines.append(f"\n**Best DES-Liu-objective reward:** {summary['best_des_liu_reward']}  \n"
                 f"Tied set size: {summary['best_des_action_set_size']}  \n"
                 f"Actions: {summary['best_des_action_set'][:8]}"
                 f"{' …' if summary['best_des_action_set_size'] > 8 else ''}\n")
    lines.append(f"\n**Intersection:** {len(summary['optimal_sets_intersection'])} action(s): "
                 f"{summary['optimal_sets_intersection'][:8]}"
                 f"{' …' if len(summary['optimal_sets_intersection']) > 8 else ''}\n")

    lines.append("## Per protocol\n")
    lines.append("| protocol | attempted | des_feasible | paper_feasible | best_des_action | best_des_liu_reward | best_paper_reward |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, s in summary["per_protocol"].items():
        lines.append(
            f"| {name} | {s['attempted']} | {s['des_feasible']} | {s['paper_feasible']} | "
            f"{s['best_des_action']} | {s['best_des_liu_reward']} | {s['best_paper_reward']} |"
        )

    lines.append("\n## Largest paper↔DES discrepancies\n")
    for name, entry in summary["largest_discrepancies"].items():
        lines.append(f"- **{name}**: {entry}")

    lines.append("\n## Invariants\n")
    inv = summary["invariants"]
    lines.append(f"- seeds observed: {inv['unique_seeds']}")
    lines.append(f"- C1 violations: {len(inv['c1_violations'])}")
    lines.append(f"- C3 violations: {len(inv['c3_violations'])}")
    if inv["c1_violations"]:
        lines.append(f"  - first: {inv['c1_violations'][:5]}")
    if inv["c3_violations"]:
        lines.append(f"  - first: {inv['c3_violations'][:5]}")

    lines.append("\n## Sanity\n")
    lines.append(f"- non-monotonic T_C_DES dips: {len(summary['sanity']['monotonicity_dips'])}")
    if summary["sanity"]["monotonicity_dips"]:
        for line in summary["sanity"]["monotonicity_dips"][:8]:
            lines.append(f"  - {line}")
    lines.append(f"- NaN/negative metrics: {len(summary['sanity']['nan_or_negative_metrics'])}")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {MD_PATH}")


if __name__ == "__main__":
    emit(summarize())
