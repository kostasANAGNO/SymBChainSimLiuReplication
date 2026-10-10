"""Paper-scale setup (Table I): N=100 IIoT nodes, K=21 block producers, everything DES-evaluated.

Table I: area 1 km x 1 km; stakes 1-50 tokens; compute 10-30 GHz; link rates 10-100 Mbps; chi = 200 B;
S_dot = 8 MB; T_dot = 10 s; eta_s = 0.2, eta_l = 0.3; omega = 6.
Node attributes are drawn once from a fixed seed (the paper gives only the ranges), the FSMC link levels,
alpha/beta, recovery delay and timeout are ours -- OUR_RECONSTRUCTION.
Reward R = Omega = floor(S_B/chi)/T_I if the DES reports C1 & C2 & C3, else 0.
Action = [validator set a (K of N), protocol delta, S_B in {0.2, 0.4, ..} MB, T_I in {0.5, .., 10} s].
"""
from __future__ import annotations

import csv
import dataclasses
import random
import sys
from decimal import Decimal
from math import floor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIU_ROOT = ROOT / "experiments" / "liu_replication"
sys.path.insert(0, str(ROOT / "src" / "Simulator"))
sys.path.insert(0, str(LIU_ROOT / "dynamic_epoch"))
sys.path.insert(0, str(LIU_ROOT / "drl"))

import liu_dynamic_env as _lde
from liu_dynamic_env import LiuDynamicEpochEnv
from Liu.Action import LiuAction
from Liu.LinkFSMC import LinkRateLevels, LinkTransitionMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import LiuReferenceParameters
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

PROTOCOLS = (LiuConsensusProtocol.PBFT, LiuConsensusProtocol.ZYZZYVA, LiuConsensusProtocol.LIU_QUORUM)
PROTOCOL_NAMES = ("PBFT", "ZYZZYVA", "LIU_QUORUM")
T_I_GRID = tuple(float(Decimal("0.5") * i) for i in range(1, 21))
ALPHA_CYCLES, BETA_CYCLES, NETWORK_TIMEOUT_S, RECOVERY_DELAY_S = 2_000_000.0, 1_000_000.0, 100.0, 0.05


def block_size_grid(s_max_mb: float) -> tuple[float, ...]:
    steps = int((Decimal(str(s_max_mb)) / Decimal("0.2")).to_integral_value())
    return tuple(float(Decimal("0.2") * i) for i in range(1, steps + 1))


N100, K21 = 100, 21
_rng = random.Random(2019)
STAKES100 = tuple(float(_rng.randint(1, 50)) for _ in range(N100))
CAPS100 = tuple(round(_rng.uniform(10.0, 30.0), 2) for _ in range(N100))
POS100 = tuple((round(_rng.random(), 4), round(_rng.random(), 4)) for _ in range(N100))
LINKS100 = tuple(tuple(None if i == j else float(10 + ((i + j) % 91)) for j in range(N100)) for i in range(N100))
GEO100 = ContinuousSpatialIntensityModel(planar_gradient_intensity(K21, 0.6), K21, lambda_form="planar_gradient s=0.6")


def make_action100(validators, proto: int, sb: float, ti: float) -> LiuAction:
    return LiuAction(N100, K21, tuple(sorted(validators)), PROTOCOLS[proto], sb, ti)


# Link-rate profiles (the paper gives only the 10-100 Mbps range; the FSMC levels/transitions are ours).
LINK_PROFILES = {
    "default": ((10.0, 55.0, 100.0), ((0.70, 0.20, 0.10), (0.15, 0.70, 0.15), (0.10, 0.20, 0.70))),  # the runs so far
    "hi": ((55.0, 100.0), ((0.80, 0.20), (0.20, 0.80))),                                              # no 10 Mbps links
    "all100": ((100.0,), ((1.0,),)),                                                                  # static 100 Mbps
}


class _CycleScaledEnv(LiuDynamicEpochEnv):
    """Scales the cryptographic cycle costs alpha/beta read by every DES step (sensitivity runs only).

    mode 'block':  alpha, beta x scale                      (cost per block, as in the DES default when scale=1)
    mode 'per_tx': alpha, beta x scale x floor(S_B / chi)   (every transaction carries its own signature/MACs)
    """

    def configure_cycles(self, mode: str, scale: float) -> None:
        self._cyc_mode, self._cyc_scale, self._ref0 = mode, scale, self._ref_params

    def step(self, action):
        k = floor(action.block_size_mb * 1_000_000.0 / self._ref0.transaction_size_bytes) if self._cyc_mode == "per_tx" else 1
        f = self._cyc_scale * k
        self._ref_params = dataclasses.replace(self._ref0, signature_verification_cycles_alpha=self._ref0.signature_verification_cycles_alpha * f,
                                               mac_operation_cycles_beta=self._ref0.mac_operation_cycles_beta * f)
        return super().step(action)


def make_env100(seed: int, *, chi_bytes: int = 200, omega: float = 6.0, offered_tx: int = 100_000, capability_scale: float = 1.0,
                 link_profile: str = "default", cycle_mode: str = "block", cycle_scale: float = 1.0) -> LiuDynamicEpochEnv:
    levels, trans = LINK_PROFILES[link_profile]
    _lde.FSMC_LEVELS = LinkRateLevels(rates_mbps=levels)            # read by _make_fsmc_state at reset()
    _lde.FSMC_TRANSITION = LinkTransitionMatrix(probabilities=trans)
    ref = LiuReferenceParameters(
        signature_verification_cycles_alpha=ALPHA_CYCLES,
        mac_operation_cycles_beta=BETA_CYCLES,
        network_timeout_s=NETWORK_TIMEOUT_S,
        finality_multiplier_omega=omega,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=int(chi_bytes),
        recovery_delay_s=RECOVERY_DELAY_S,
    )
    env = _CycleScaledEnv(
        node_count=N100, transaction_size_bytes=float(chi_bytes), stakes_tokens=STAKES100,
        capabilities_ghz=tuple(c * capability_scale for c in CAPS100), positions_km=POS100, initial_link_rows_mbps=LINKS100, faulty_node_ids=(),
        reference_params=ref, geo_model=GEO100, offered_workload_tx=offered_tx, max_events=4_000_000, seed=seed)
    env.configure_cycles(cycle_mode, cycle_scale)
    env.reset(seed=seed)
    return env


# ---- configuration grid (delta, S_B, T_I), scheme masks, sweep definitions ---------------------------------
def make_grid(s_head_max_mb: float):
    """(grid, n_cfg, decode): decode(c) -> (protocol index, S_B MB, T_I s). Fig. 5 uses 8 MB, Fig. 7-10 use 10 MB."""
    grid = block_size_grid(s_head_max_mb)
    n_cfg = len(PROTOCOLS) * len(grid) * len(T_I_GRID)

    def decode(c: int) -> tuple[int, float, float]:
        p, rem = divmod(c, len(grid) * len(T_I_GRID))
        sb_i, ti_i = divmod(rem, len(T_I_GRID))
        return p, grid[sb_i], T_I_GRID[ti_i]

    return grid, n_cfg, decode


def scheme_mask(scheme: str, s_max_mb: float, n_cfg: int, decode) -> np.ndarray:
    m = np.zeros(n_cfg, dtype=bool)
    for c in range(n_cfg):
        p, sb, ti = decode(c)
        ok = sb <= s_max_mb + 1e-9
        if scheme == "no_cas":
            ok &= p == 0
        elif scheme == "fixed_block_size":
            ok &= abs(sb - min(4.0, s_max_mb)) < 1e-9
        elif scheme == "fixed_block_interval":
            ok &= abs(ti - 1.0) < 1e-9
        m[c] = ok
    return m


SWEEPS = {
    "fig7": {"param": "omega", "values": [float(v) for v in range(2, 21, 2)], "metric": "throughput"},
    "fig8": {"param": "chi_bytes", "values": [float(v) for v in range(100, 551, 50)], "metric": "throughput"},
    "fig9": {"param": "avg_ghz", "values": [float(v) for v in range(10, 31, 2)], "metric": "ttf"},
    "fig10": {"param": "s_max_mb", "values": [float(v) for v in range(1, 11)], "metric": "throughput"},
}
DEFAULTS = {"omega": 6.0, "chi_bytes": 200.0, "avg_ghz": 20.0, "s_max_mb": 8.0}  # Table I / population mean
SCHEMES = ("proposed", "no_cas", "fixed_block_size", "fixed_block_interval", "static")


def truncate_log(path: Path, start: int) -> None:
    """Drop CSV rows for episodes >= start (the log can run ahead of the last checkpoint)."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        fields, keep = rd.fieldnames, [r for r in rd if int(r["episode"]) < start]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
