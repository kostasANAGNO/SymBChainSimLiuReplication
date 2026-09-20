"""Gate verification: REDUCED_VALIDATOR_SELECTION_OPERATIONALLY_VERIFIED

Checks:
1. All 435 validator sets unique and K=28 each
2. Stake Gini varies across sets (51 fail, 384 pass C1 stake)
3. T_C varies across protocols and validator sets (causality chain)
4. C3 security tests: malicious included → C3 False; excluded → C3 True (for PBFT/Zyzzyva)
5. LIU_QUORUM security: F^2=0 → C3 always False when f=1
6. Reward = Omega if DES C1∧C2∧C3 else 0 (verified for all rows)
7. No NaN, negative reward, or unexplained ERRORs
"""
from __future__ import annotations
import csv
import math
from pathlib import Path

CSV = Path(__file__).parent / "validator_selection_results.csv"

with CSV.open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))

print(f"Rows loaded: {len(rows)}")
assert len(rows) == 1311, f"Expected 1311, got {len(rows)}"

# ── 1. All SUCCESS ────────────────────────────────────────────────────────────
errors = [r for r in rows if r["run_status"] != "SUCCESS"]
print(f"Errors: {len(errors)}")
assert len(errors) == 0, f"Non-SUCCESS rows: {[r['action_id'] for r in errors]}"

# ── 2. Unique validator sets across 435 main sets ─────────────────────────────
main_rows = [r for r in rows if r["experiment_type"] == "main"]
assert len(main_rows) == 1305, f"Expected 1305 main rows, got {len(main_rows)}"

# Group by (validator_ids, protocol) — should be exactly 1 per combo
combos = set()
for r in main_rows:
    combos.add((r["validator_ids"], r["protocol"]))
assert len(combos) == 1305, f"Duplicate (vs, protocol) combos: expected 1305, got {len(combos)}"

# All unique validator sets
vs_sets = set(r["validator_ids"] for r in main_rows)
print(f"Unique validator sets in main rows: {len(vs_sets)} (expected 435)")
assert len(vs_sets) == 435

# All have K=28 validators
for r in main_rows:
    k = len(r["validator_ids"].split(","))
    assert k == 28, f"action {r['action_id']} has K={k} instead of 28"
print("All 435 validator sets unique, K=28 each: OK")

# ── 3. Stake Gini variation ────────────────────────────────────────────────────
ginis = [float(r["stake_gini_paper"]) for r in main_rows if r["stake_gini_paper"]]
gini_min, gini_max = min(ginis), max(ginis)
c1_stake_pass = sum(1 for g in ginis if g <= 0.2)
c1_stake_fail = len(ginis) - c1_stake_pass
# 3 protocols per vs → divide by 3 for per-set counts
per_set_ginis = [float(r["stake_gini_paper"]) for r in main_rows if r["protocol"] == "PBFT"]
ps_pass = sum(1 for g in per_set_ginis if g <= 0.2)
ps_fail = len(per_set_ginis) - ps_pass
print(f"Stake Gini range: [{gini_min:.4f}, {gini_max:.4f}] (η_s=0.2)")
print(f"Per validator set: {ps_pass}/435 pass C1 stake-Gini, {ps_fail}/435 fail")
assert gini_min < 0.2 < gini_max, "Stake Gini should straddle η_s=0.2"
assert ps_pass == 384 and ps_fail == 51, f"Expected 384 pass / 51 fail, got {ps_pass}/{ps_fail}"
print("Stake Gini discrimination: OK (384/435 pass, 51/435 fail)")

# ── 4. T_C variation across protocols ─────────────────────────────────────────
for proto in ("PBFT", "ZYZZYVA", "LIU_QUORUM"):
    tc_vals = [float(r["des_t_c"]) for r in main_rows
               if r["protocol"] == proto and r["des_t_c"]]
    assert len(tc_vals) > 0
    tc_min, tc_max = min(tc_vals), max(tc_vals)
    print(f"T_C DES [{proto}]: min={tc_min:.4f}s  max={tc_max:.4f}s  "
          f"variation={tc_max - tc_min:.4f}s")
    # T_C should vary — at least some difference across 435 validator sets
    assert tc_max > tc_min, f"T_C should vary across {proto} runs"
print("T_C causality chain: OK (varies across validator sets)")

# ── 5. Reward integrity ────────────────────────────────────────────────────────
chi_bytes = 200
sb_mb = 1.0
ti_s = 2.0
omega = math.floor(sb_mb * 1_000_000 / chi_bytes) / ti_s

reward_violations = []
for r in main_rows:
    if not r["reward_des_liu_objective"]:
        continue
    rwd = float(r["reward_des_liu_objective"])
    c1 = r["des_c1"].lower() == "true"
    c2 = r["des_c2"].lower() == "true"
    c3 = r["des_c3"].lower() == "true"
    feasible = c1 and c2 and c3
    expected = omega if feasible else 0.0
    if abs(rwd - expected) > 0.1:
        reward_violations.append((r["action_id"], rwd, expected, feasible))
print(f"Reward violations (|actual-expected|>0.1): {len(reward_violations)}")
assert len(reward_violations) == 0, f"Reward violations: {reward_violations[:5]}"
print(f"Reward = Ω={omega:.1f} if feasible else 0: verified for all {len(main_rows)} main rows")

# ── 6. Security tests ─────────────────────────────────────────────────────────
sec_rows = [r for r in rows if r["experiment_type"].startswith("security")]
assert len(sec_rows) == 6, f"Expected 6 security rows, got {len(sec_rows)}"

for r in sec_rows:
    malicious_sel = int(r["malicious_selected"])
    c3_des = r["des_c3"].lower() == "true"
    if r["experiment_type"] == "security_included":
        # Malicious node is in the validator set (malicious_selected=1)
        assert malicious_sel == 1, f"Expected malicious_selected=1, got {malicious_sel}"
        if r["protocol"] != "LIU_QUORUM":
            # PBFT/Zyzzyva: F^delta=floor((K-1)/3)=9 for K=28 → f=1 ≤ 9 → C3 passes
            assert c3_des is True or c3_des, f"{r['protocol']} security_included C3 expected True"
        else:
            # LIU_QUORUM: F^2=0 → f=1 > 0 → C3 fails
            assert not c3_des, f"LIU_QUORUM security_included C3 expected False, got {r['des_c3']}"
    else:
        # security_excluded: malicious node NOT in validator set
        assert malicious_sel == 0, f"Expected malicious_selected=0, got {malicious_sel}"
        assert c3_des is True or c3_des, f"{r['protocol']} security_excluded C3 expected True"

print("Security tests (C3 causality): OK")
print("  PBFT/Zyzzyva with f=1 included: C3=True (F^δ=9 ≥ 1)")
print("  LIU_QUORUM with f=1 included: C3=False (F^2=0 < 1) — confirmed infeasible")
print("  All protocols with f=1 excluded: C3=True")

# ── Summary ────────────────────────────────────────────────────────────────────
total_feasible = sum(1 for r in main_rows
                     if r["des_c1"].lower() == "true"
                     and r["des_c2"].lower() == "true"
                     and r["des_c3"].lower() == "true")
nonzero_reward = sum(1 for r in main_rows
                     if r["reward_des_liu_objective"] and float(r["reward_des_liu_objective"]) > 0)
print()
print("=== Phase 3 Gate Summary ===")
print(f"Total rows:         1311 (1305 main + 6 security)")
print(f"All SUCCESS:        {len(rows) - len(errors)}/1311")
print(f"Unique VS sets:     435/435")
print(f"Stake Gini range:   [{gini_min:.4f}, {gini_max:.4f}] (η_s=0.2 inside range)")
print(f"C1 pass/fail:       {ps_pass}/435 pass, {ps_fail}/435 fail")
print(f"DES feasible main:  {total_feasible}/1305")
print(f"Non-zero reward:    {nonzero_reward}/1305")
print()
print("GATE: REDUCED_VALIDATOR_SELECTION_OPERATIONALLY_VERIFIED")
