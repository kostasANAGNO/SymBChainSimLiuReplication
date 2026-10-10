"""Paper-scale population (Table I): N=100 IIoT nodes, K=21 block producers, DES-evaluated.

Table I: area 1 km x 1 km; stakes 1-50 tokens; compute 10-30 GHz; link rates 10-100 Mbps (FSMC levels as in
the N=30 runs); chi = 200 B; S_dot = 8 MB; T_dot = 10 s; eta_s = 0.2, eta_l = 0.3; omega = 6.
Node attributes are drawn once from a fixed seed (the paper gives only the ranges) -- OUR_RECONSTRUCTION.
"""
from __future__ import annotations

import random

from common import *  # noqa: F401,F403  (puts the Liu packages on sys.path)
import liu_dynamic_env as _lde
from Liu.LinkFSMC import LinkRateLevels, LinkTransitionMatrix
from Liu.ContinuousSpatial import ContinuousSpatialIntensityModel, planar_gradient_intensity

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


def make_env100(seed: int, *, chi_bytes: int = 200, omega: float = 6.0, offered_tx: int = 100_000, capability_scale: float = 1.0,
                 link_profile: str = "default") -> LiuDynamicEpochEnv:
    levels, trans = LINK_PROFILES[link_profile]
    _lde.FSMC_LEVELS = LinkRateLevels(rates_mbps=levels)            # read by _make_fsmc_state at reset()
    _lde.FSMC_TRANSITION = LinkTransitionMatrix(probabilities=trans)
    ref = LiuReferenceParameters(
        signature_verification_cycles_alpha=base.REF_PARAMS.signature_verification_cycles_alpha,
        mac_operation_cycles_beta=base.REF_PARAMS.mac_operation_cycles_beta,
        network_timeout_s=base.REF_PARAMS.network_timeout_s,
        finality_multiplier_omega=omega,
        stake_gini_threshold_eta_s=0.2,
        geographic_gini_threshold_eta_l=0.3,
        transaction_size_bytes=int(chi_bytes),
        recovery_delay_s=base.REF_PARAMS.recovery_delay_s,
    )
    env = LiuDynamicEpochEnv(
        node_count=N100, transaction_size_bytes=float(chi_bytes), stakes_tokens=STAKES100,
        capabilities_ghz=tuple(c * capability_scale for c in CAPS100), positions_km=POS100, initial_link_rows_mbps=LINKS100, faulty_node_ids=(),
        reference_params=ref, geo_model=GEO100, offered_workload_tx=offered_tx, max_events=4_000_000, seed=seed)
    env.reset(seed=seed)
    return env
