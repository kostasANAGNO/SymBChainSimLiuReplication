"""Full 2400-action controlled fixed-validator sweep (3 protocols × 40 S_B × 20 T_I).

Same S0 for every action (independent runs, no S_t chaining). Reuses the verified
`run_controlled_smoke.py` helpers (`build_snapshot`, `_run_one`, `CSV_COLUMNS`).
Resumable: after each action the full CSV is rewritten in action-id order, so a re-launch
skips every already-SUCCESS action_id and re-attempts prior failures.

Do NOT change consensus, metrics, reward, or state semantics here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "Simulator"))

from Chain.Consensus.LiuRuntime.Common.SingleActionRuntime import (
    LiuSingleActionSnapshot,
    run_single_action,
)
from Liu.Action import LiuAction
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from PaperReference.Comparison import paper_reference_for

# ── Fixed sweep parameters (N=100, K=21, paper Table I population) ────────────
FIXED_SEED = 0
_N = 100
_K = 21
_CHI_BYTES = 200
_TX_SIZE_MB = _CHI_BYTES / 1_000_000.0
_LAMBDA_SLOPE = 0.6
_ZYZZYVA_RECOVERY_DELAY_S = 0.05
_OFFERED_WORKLOAD_TX = 100_000

_STAKES = tuple(float(4 + (i % 10)) for i in range(_N))
_CAPABILITIES = tuple(float(10 + (i % 21)) for i in range(_N))
_POSITIONS = tuple((0.1 * (i % 10), 0.1 * (i // 10)) for i in range(_N))
_LINK_ROWS = tuple(
    tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(_N))
    for i in range(_N)
)

CSV_COLUMNS = (
    "action_id", "protocol", "block_size_mb", "block_interval_s",
    "validator_set_hash", "seed",
    "throughput_paper_nominal", "paper_t_c", "paper_t_f",
    "paper_c1", "paper_c2", "paper_c3", "reward_paper_reference",
    "des_t_c", "des_t_f", "des_c1", "des_c2", "des_c3", "reward_des_liu_objective",
    "delta_t_c", "delta_t_f", "constraint_agreement",
    "run_status", "failure_reason", "wall_clock_seconds",
)


@dataclass
class _RunResult:
    status: str
    reason: str
    row: dict
    wall_clock_s: float


def _build_geo_model() -> ContinuousSpatialIntensityModel:
    return ContinuousSpatialIntensityModel(
        planar_gradient_intensity(_K, _LAMBDA_SLOPE), _K,
        lambda_form=f"planar_gradient s={_LAMBDA_SLOPE}",
    )


def _build_params() -> LiuReferenceParameters:
    return LiuReferenceParameters(
        signature_verification_cycles_alpha=2_000_000.0,
        mac_operation_cycles_beta=1_000_000.0,
        network_timeout_s=100.0,
        finality_multiplier_omega=6.0,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=_CHI_BYTES,
        recovery_delay_s=_ZYZZYVA_RECOVERY_DELAY_S,
    )


def build_snapshot() -> LiuSingleActionSnapshot:
    return LiuSingleActionSnapshot(
        node_count=_N,
        transaction_size_bytes=float(_CHI_BYTES),
        stakes_tokens=_STAKES,
        capabilities_ghz=_CAPABILITIES,
        positions_km=_POSITIONS,
        link_rows_mbps=_LINK_ROWS,
        faulty_node_ids=(),
        epoch0_validator_ids=tuple(range(_K)),
    )


def _run_one(index: int, protocol_name: str, s_b: float, t_i: float) -> _RunResult:
    row: dict = {name: "" for name in CSV_COLUMNS}
    row["action_id"] = index
    row["protocol"] = protocol_name
    row["block_size_mb"] = s_b
    row["block_interval_s"] = t_i
    row["seed"] = FIXED_SEED
    started = time.time()
    try:
        protocol = LiuConsensusProtocol[protocol_name]
        validator_ids = tuple(range(_K))
        row["validator_set_hash"] = hashlib.sha256(
            json.dumps(list(validator_ids)).encode()
        ).hexdigest()[:16]
        action = LiuAction(_N, _K, validator_ids, protocol, s_b, t_i)
        snapshot = build_snapshot()
        result = run_single_action(
            snapshot,
            action,
            _build_geo_model(),
            reference_params=_build_params(),
            stake_gini_threshold=0.2,
            geographic_gini_threshold=0.3,
            finality_multiplier_omega=6.0,
            signature_cycles_alpha=2_000_000.0,
            mac_cycles_beta=1_000_000.0,
            offered_workload_tx=_OFFERED_WORKLOAD_TX,
            workload_tx_size_mb=_TX_SIZE_MB,
            max_events=2_000_000,
        )
        p = paper_reference_for(snapshot, action, _build_geo_model(), _build_params())
        d = result.des_observed
        paper_t_c = p.consensus_latency_s
        paper_t_f = p.finality_latency_s
        des_t_c = d.t_c_des_s
        des_t_f = d.t_f_des_s
        row.update(
            throughput_paper_nominal=p.throughput_paper_nominal,
            paper_t_c=paper_t_c,
            paper_t_f=paper_t_f,
            paper_c1=p.c1_decentralization_passed,
            paper_c2=p.c2_finality_passed,
            paper_c3=p.c3_security_passed,
            reward_paper_reference=p.reward_paper_reference,
            des_t_c=des_t_c,
            des_t_f=des_t_f,
            des_c1=d.c1_decentralization_passed,
            des_c2=d.c2_finality_passed,
            des_c3=d.c3_security_passed,
            reward_des_liu_objective=d.reward_des_liu_objective,
            delta_t_c=round(des_t_c - paper_t_c, 6) if des_t_c != float("inf") else "",
            delta_t_f=round(des_t_f - paper_t_f, 6) if des_t_f != float("inf") else "",
            constraint_agreement=(
                d.c1_decentralization_passed == p.c1_decentralization_passed
                and d.c2_finality_passed == p.c2_finality_passed
                and d.c3_security_passed == p.c3_security_passed
            ),
            run_status="SUCCESS",
        )
        elapsed = time.time() - started
        return _RunResult(status="SUCCESS", reason="", row=row, wall_clock_s=elapsed)
    except Exception as exc:
        traceback.print_exc()
        reason = f"{type(exc).__name__}: {exc}"
        row["run_status"] = "ERROR"
        row["failure_reason"] = reason
        elapsed = time.time() - started
        return _RunResult(status="ERROR", reason=reason, row=row, wall_clock_s=elapsed)

OUT_DIR = Path(__file__).parent
CSV_PATH = OUT_DIR / "controlled_full_results.csv"
JSON_PATH = OUT_DIR / "controlled_full_summary.json"
MD_PATH = OUT_DIR / "controlled_full_summary.md"
PROGRESS_EVERY = 1  # rewrite CSV after every completed action

_PROTOCOLS = ("PBFT", "ZYZZYVA", "LIU_QUORUM")


def _rounded(x: float, decimals: int = 4) -> float:
    return round(float(x), decimals)


def full_action_grid() -> tuple[tuple[str, float, float], ...]:
    """3 protocols × S_B ∈ {0.2, 0.4, …, 8.0} × T_I ∈ {0.5, 1.0, …, 10.0} = 2400 actions."""
    sbs = tuple(_rounded(0.2 * i) for i in range(1, 41))
    tis = tuple(_rounded(0.5 * i) for i in range(1, 21))
    return tuple(
        (protocol, sb, ti) for protocol in _PROTOCOLS for sb in sbs for ti in tis
    )


def _load_prior_rows(csv_path: Path) -> dict[int, dict]:
    """Return {action_id: row_dict} from an existing CSV, if any."""
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return {int(row["action_id"]): row for row in csv.DictReader(fh)}


def _write_csv_all(csv_path: Path, results: dict[int, dict]) -> None:
    """Rewrite the full CSV in action-id order (2400 rows × ~30 cols is trivial).

    The atomic swap can transiently fail on Windows with `PermissionError` (WinError 5)
    when the destination file is held for a few ms by an AV/indexer. Retry with backoff.
    """
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for action_id in sorted(results):
            writer.writerow({name: results[action_id].get(name, "") for name in CSV_COLUMNS})
    last_error: Exception | None = None
    for attempt in range(20):
        try:
            tmp_path.replace(csv_path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"could not atomically replace {csv_path} after retries: {last_error}")


def _validate_grid(actions: tuple[tuple[str, float, float], ...]) -> None:
    assert len(actions) == 2400, f"expected 2400 actions, got {len(actions)}"
    assert len(set(actions)) == 2400, "grid contains duplicates"
    protocols = {a[0] for a in actions}
    assert protocols == set(_PROTOCOLS), protocols
    sbs = sorted({a[1] for a in actions})
    assert sbs[0] == 0.2 and sbs[-1] == 8.0 and len(sbs) == 40, sbs
    tis = sorted({a[2] for a in actions})
    assert tis[0] == 0.5 and tis[-1] == 10.0 and len(tis) == 20, tis


def main(actions_override: tuple | None = None) -> dict:
    """Run the full 2400 sweep. `actions_override` is for resume-tests only."""
    actions = tuple(actions_override) if actions_override is not None else full_action_grid()
    if actions_override is None:
        _validate_grid(actions)

    prior = _load_prior_rows(CSV_PATH)
    successful_ids = {i for i, r in prior.items() if r.get("run_status") == "SUCCESS"}
    print(f"loaded {len(prior)} prior rows ({len(successful_ids)} SUCCESS, "
          f"{len(prior) - len(successful_ids)} non-SUCCESS to be re-attempted)")

    results: dict[int, dict] = dict(prior)
    started = time.time()
    attempted_now = 0
    for index, (protocol_name, s_b, t_i) in enumerate(actions):
        if index in successful_ids:
            continue
        r = _run_one(index, protocol_name, s_b, t_i)
        r.row["wall_clock_seconds"] = r.wall_clock_s
        results[index] = r.row
        attempted_now += 1
        if attempted_now % PROGRESS_EVERY == 0:
            _write_csv_all(CSV_PATH, results)
        elapsed = time.time() - started
        rate = attempted_now / elapsed if elapsed > 0 else 0
        remaining = len(actions) - (index + 1) - sum(
            1 for j in range(index + 1, len(actions)) if j in successful_ids
        )
        eta_min = (remaining / rate / 60.0) if rate > 0 else float("inf")
        print(f"[{index + 1:04d}/{len(actions)}] {protocol_name:11s} "
              f"S_B={s_b:>4} MB T_I={t_i:>4} s  {r.status:8s}  "
              f"({r.wall_clock_s:5.2f} s)  ETA {eta_min:6.1f} min", flush=True)
    # final flush
    _write_csv_all(CSV_PATH, results)
    return results


if __name__ == "__main__":
    main()
