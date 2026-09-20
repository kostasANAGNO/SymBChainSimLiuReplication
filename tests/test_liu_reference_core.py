"""Golden tests for LIU_PAPER_REFERENCE_MODE core (paper equations only).

Each assertion cites the Liu et al. (IEEE TII 2019) equation/appendix it verifies. Expected
values are computed independently from the paper formulas for simple inputs. No DES.
"""
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Liu.AnalyticalConsensus import (
    AnalyticalConsensusInput,
    LiuTransmissionUnitPolicy,
    ZyzzyvaAnalyticalPath,
)
from Liu.AnalyticalLiuQuorum import LiuQuorumAnalyticalModel
from Liu.AnalyticalPBFT import PBFTAnalyticalModel
from Liu.AnalyticalZyzzyva import ZyzzyvaAnalyticalModel
from Liu.LinkState import LinkStateMatrix
from Liu.Protocol import LiuConsensusProtocol
from Liu.ReferenceCore import (
    GeographicGiniConfig,
    LiuReferenceParameters,
    assert_lambda_normalized,
    evaluate_reference,
    geographic_gini_eq3,
    integral_of_lambda,
    paper_stake_gini,
    reference_consensus,
)

_K = 4
_DEFAULT_CAPS = (1.0, 1.0, 1.0, 1.0)


def uniform_links(n, rate):
    return LinkStateMatrix.from_rows(tuple(tuple(None if i == j else float(rate) for j in range(n)) for i in range(n)))


def fixed_links(rate_mbps=10.0, node_count=None):
    n = node_count if node_count is not None else _K
    return LinkStateMatrix.from_rows(
        tuple(tuple(None if i == j else float(rate_mbps) for j in range(n)) for i in range(n))
    )


def analytical_input(
    protocol=LiuConsensusProtocol.PBFT,
    faulty_replica_count=0,
    zyzzyva_path=None,
    recovery_delay_s=0.0,
    alpha=1e9,
    beta=1e9,
    capabilities=None,
    links=None,
    block_size_mb=1.0,
    transmission_policy=LiuTransmissionUnitPolicy.PAPER_LITERAL_MB_PER_MBPS_V1,
    transaction_size_bytes=200,
    timeout_s=10_000.0,
):
    caps = capabilities if capabilities is not None else _DEFAULT_CAPS
    lnks = links if links is not None else fixed_links(10.0)
    if protocol is LiuConsensusProtocol.LIU_QUORUM:
        return AnalyticalConsensusInput(
            validator_ids=tuple(range(_K)),
            validator_count_k=_K,
            protocol=protocol,
            client_validator_id=0,
            primary_validator_id=None,
            block_size_mb=block_size_mb,
            transaction_size_bytes=transaction_size_bytes,
            batch_size_m=1,
            block_interval_s=2.0,
            computational_capabilities_ghz=caps,
            directed_link_rates_mbps=lnks,
            signature_verification_cycles_alpha=alpha,
            mac_operation_cycles_beta=beta,
            network_timeout_s=timeout_s,
            faulty_replica_count=0,
            zyzzyva_path=None,
            recovery_delay_s=0.0,
            transmission_unit_policy=transmission_policy,
        )
    return AnalyticalConsensusInput(
        validator_ids=tuple(range(_K)),
        validator_count_k=_K,
        protocol=protocol,
        client_validator_id=0,
        primary_validator_id=1,
        block_size_mb=block_size_mb,
        transaction_size_bytes=transaction_size_bytes,
        batch_size_m=2,
        block_interval_s=2.0,
        computational_capabilities_ghz=caps,
        directed_link_rates_mbps=lnks,
        signature_verification_cycles_alpha=alpha,
        mac_operation_cycles_beta=beta,
        network_timeout_s=timeout_s,
        faulty_replica_count=faulty_replica_count,
        zyzzyva_path=zyzzyva_path,
        recovery_delay_s=recovery_delay_s,
        transmission_unit_policy=transmission_policy,
    )

class StakeGiniTests(unittest.TestCase):
    def test_equal_stakes_zero_gini(self):  # Eq. (2)
        self.assertEqual(paper_stake_gini([5.0, 5.0, 5.0, 5.0]), 0.0)

    def test_hand_computed_gini(self):  # Eq. (2): sum|xi-xj|/(2 K sum xi)
        # stakes [1,2,3,4]: sum|diff|=20, K=4, sum=10 -> 20/(2*4*10)=0.25
        self.assertAlmostEqual(paper_stake_gini([1.0, 2.0, 3.0, 4.0]), 0.25)


class GeographicGiniTests(unittest.TestCase):
    def test_constant_density_zero(self):  # Eq. (3): uniform lambda -> G=0
        g = geographic_gini_eq3(lambda x, y: 21.0, GeographicGiniConfig(resolution=32))
        self.assertEqual(g, 0.0)

    def test_integral_equals_K(self):  # int_Xi lambda = K for normalized intensity
        self.assertAlmostEqual(integral_of_lambda(lambda x, y: 21.0, GeographicGiniConfig(resolution=40)), 21.0)

    def test_piecewise_half_density_analytic(self):  # Eq. (3) closed form = 0.5
        # lambda=2 on left half (x<0.5), 0 on right half, unit region:
        # numerator=1, denominator=2 -> G=0.5 (boundary-aligned, exact at even resolution)
        g = geographic_gini_eq3(lambda x, y: 2.0 if x < 0.5 else 0.0, GeographicGiniConfig(resolution=50))
        self.assertAlmostEqual(g, 0.5, places=6)

    def test_convergence_check_passes_for_smooth(self):
        g = geographic_gini_eq3(lambda x, y: 1.0 + x, GeographicGiniConfig(resolution=64, convergence_tol=1e-2), check_convergence=True)
        self.assertGreater(g, 0.0)

    def test_not_a_distance_surrogate(self):
        # a coordinate-pairwise-distance metric would be non-zero for uniform-density distinct
        # points; Eq. (3) on a constant density is exactly 0. Guards against the surrogate.
        self.assertEqual(geographic_gini_eq3(lambda x, y: 4.0, GeographicGiniConfig(resolution=16)), 0.0)


def _params(alpha=3e9, beta=1e9, T=10_000.0, omega=6.0, eta_s=0.2, eta_l=0.3, chi=200, M=3):
    return LiuReferenceParameters(
        signature_verification_cycles_alpha=alpha, mac_operation_cycles_beta=beta,
        network_timeout_s=T, finality_multiplier_omega=omega, stake_gini_threshold_eta_s=eta_s,
        geographic_gini_threshold_eta_l=eta_l, transaction_size_bytes=chi, pbft_zyzzyva_batch_m=M,
        recovery_delay_s=5.0, transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
    )


class ConsensusGoldenTests(unittest.TestCase):
    # K=4 producers {0,1,2,3}; uniform links r, uniform capability c; factor=8 (SI); no timeout bind.
    K, r, c, S_B, TI = 4, 100.0, 1.0, 1.0, 5.0
    caps = (1.0, 1.0, 1.0, 1.0)

    def _links(self):
        return uniform_links(self.K, self.r)

    def test_pbft_eq15(self):  # Appendix B Eq. (15), M=3, f=0
        p = _params()
        res = reference_consensus(producer_ids=[0, 1, 2, 3], protocol=LiuConsensusProtocol.PBFT,
                                  block_size_mb=self.S_B, block_interval_s=self.TI,
                                  node_capabilities_ghz=self.caps, link_rates=self._links(),
                                  malicious_count=0, params=p)
        M = 3
        # O0_bp = M a + (2M+4(K-1)) b ; O0_bi = M a + (M+4(K-1)) b
        primary = M * p.signature_verification_cycles_alpha + (2 * M + 4 * (self.K - 1)) * p.mac_operation_cycles_beta
        # T_V,0 = (1/M) max{O/c} = primary/(c*1e9)/M
        tv = primary / (self.c * 1e9) / M
        # T_D,0 = (1/M)(t1..t5), uniform links, factor 8: each t = M*S_B*8/r -> T_D = 5*S_B*8/r
        td = 5 * self.S_B * 8.0 / self.r
        self.assertAlmostEqual(res.validation_delay_s, tv, places=9)
        self.assertAlmostEqual(res.delivery_delay_s, td, places=9)
        self.assertAlmostEqual(res.consensus_delay_s, tv + td, places=9)          # Eq. (7)
        self.assertAlmostEqual(res.finality_delay_s, self.TI + tv + td, places=9)  # Eq. (6)

    def test_zyzzyva_fast_eq16(self):  # Appendix B Eq. (16), f=0
        p = _params()
        res = reference_consensus(producer_ids=[0, 1, 2, 3], protocol=LiuConsensusProtocol.ZYZZYVA,
                                  block_size_mb=self.S_B, block_interval_s=self.TI,
                                  node_capabilities_ghz=self.caps, link_rates=self._links(),
                                  malicious_count=0, params=p)
        M = 3
        primary = M * p.signature_verification_cycles_alpha + (2 * M + self.K - 1) * p.mac_operation_cycles_beta
        tv = primary / (self.c * 1e9) / M
        td = 3 * self.S_B * 8.0 / self.r  # fast case: 3 phases (t1+t2+t3), each M*S_B*8/r, /M
        self.assertAlmostEqual(res.validation_delay_s, tv, places=9)
        self.assertAlmostEqual(res.delivery_delay_s, td, places=9)

    def test_zyzzyva_recovery_eq17(self):  # Appendix B Eq. (17), f>=1
        p = _params()
        res = reference_consensus(producer_ids=[0, 1, 2, 3], protocol=LiuConsensusProtocol.ZYZZYVA,
                                  block_size_mb=self.S_B, block_interval_s=self.TI,
                                  node_capabilities_ghz=self.caps, link_rates=self._links(),
                                  malicious_count=1, params=p)
        M = 3
        primary = M * p.signature_verification_cycles_alpha + (4 * M + self.K + 1 - 1) * p.mac_operation_cycles_beta
        tv = primary / (self.c * 1e9) / M
        # recovery: (t1+t2+t3+tr+t4+t5)/M; 5 transfer phases each M*S_B*8/r + tr -> 5*S_B*8/r + tr/M
        td = 5 * self.S_B * 8.0 / self.r + p.recovery_delay_s / M
        self.assertAlmostEqual(res.validation_delay_s, tv, places=9)
        self.assertAlmostEqual(res.delivery_delay_s, td, places=9)

    def test_quorum_no_batching(self):  # Appendix B Quorum: O2=a+2b, T_V no 1/M, T_D=t1+t2
        p = _params()
        res = reference_consensus(producer_ids=[0, 1, 2, 3], protocol=LiuConsensusProtocol.LIU_QUORUM,
                                  block_size_mb=self.S_B, block_interval_s=self.TI,
                                  node_capabilities_ghz=self.caps, link_rates=self._links(),
                                  malicious_count=0, params=p)
        cyc = p.signature_verification_cycles_alpha + 2 * p.mac_operation_cycles_beta  # a+2b
        tv = cyc / (self.c * 1e9)  # no 1/M
        td = 2 * self.S_B * 8.0 / self.r  # t1+t2, no batching (payload=S_B)
        self.assertAlmostEqual(res.validation_delay_s, tv, places=9)
        self.assertAlmostEqual(res.delivery_delay_s, td, places=9)


class ConstraintAndRewardTests(unittest.TestCase):
    def _state(self, n=4, stake_lo=1.0, stake_hi=4.0, cap=1.0, rate=100.0):
        stakes = [stake_lo + (stake_hi - stake_lo) * i / (n - 1) for i in range(n)]
        caps = [cap] * n
        return stakes, caps, uniform_links(n, rate)

    def test_c1_pass_and_fail(self):  # Eq. (12) C1 = Eq.(4) AND Eq.(5)
        stakes, caps, links = self._state()
        p = _params(eta_s=0.3, eta_l=0.3)
        ev = evaluate_reference(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
                                geographic_gini=0.1, producer_mask=[1, 1, 1, 1],
                                protocol=LiuConsensusProtocol.LIU_QUORUM, block_size_mb=1.0,
                                block_interval_s=5.0, malicious_count=0, params=p)
        self.assertTrue(ev.stake_constraint_passed and ev.geographic_constraint_passed and ev.c1_decentralization_passed)
        # geo Gini above eta_l -> C1 fails
        ev2 = evaluate_reference(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
                                 geographic_gini=0.9, producer_mask=[1, 1, 1, 1],
                                 protocol=LiuConsensusProtocol.LIU_QUORUM, block_size_mb=1.0,
                                 block_interval_s=5.0, malicious_count=0, params=p)
        self.assertFalse(ev2.geographic_constraint_passed)
        self.assertFalse(ev2.c1_decentralization_passed)
        self.assertEqual(ev2.reward, 0.0)  # Eq. (13)

    def test_c3_security(self):  # Eq. (9)/(12) C3: f <= F^delta
        stakes, caps, links = self._state(n=4)
        p = _params()
        # Quorum F^2=0: any malicious -> fail
        ev = evaluate_reference(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
                                geographic_gini=0.0, producer_mask=[1, 1, 1, 1],
                                protocol=LiuConsensusProtocol.LIU_QUORUM, block_size_mb=1.0,
                                block_interval_s=5.0, malicious_count=1, params=p)
        self.assertEqual(ev.tolerated_faults, 0)
        self.assertFalse(ev.c3_security_passed)
        self.assertEqual(ev.reward, 0.0)
        # PBFT F^0 = floor((4-1)/3) = 1: one malicious tolerated
        ev2 = evaluate_reference(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
                                 geographic_gini=0.0, producer_mask=[1, 1, 1, 1],
                                 protocol=LiuConsensusProtocol.PBFT, block_size_mb=1.0,
                                 block_interval_s=5.0, malicious_count=1, params=p)
        self.assertEqual(ev2.tolerated_faults, 1)
        self.assertTrue(ev2.c3_security_passed)

    def test_c2_finality_and_reward(self):  # Eq. (6)(7)(8)(13)
        stakes, caps, links = self._state(n=4, rate=100.0)
        p = _params(omega=6.0, chi=200)
        ev = evaluate_reference(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
                                geographic_gini=0.0, producer_mask=[1, 1, 1, 1],
                                protocol=LiuConsensusProtocol.LIU_QUORUM, block_size_mb=1.0,
                                block_interval_s=5.0, malicious_count=0, params=p)
        self.assertAlmostEqual(ev.finality_limit_s, 6.0 * 5.0)  # omega*T_I
        self.assertAlmostEqual(ev.finality_latency_s, ev.consensus_latency_s + 5.0)  # Eq. (6)
        if ev.feasible:
            # Eq. (1)/(13): floor(1MB*1e6/200)/5 = floor(5000)/5 = 1000
            self.assertAlmostEqual(ev.throughput_tps, 1000.0)
            self.assertAlmostEqual(ev.reward, 1000.0)

    def test_reward_is_pure_no_des(self):
        # identical inputs -> identical output (determinism); mode label present
        stakes, caps, links = self._state()
        p = _params()
        kw = dict(node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links, geographic_gini=0.0,
                  producer_mask=[1, 1, 1, 1], protocol=LiuConsensusProtocol.LIU_QUORUM,
                  block_size_mb=1.0, block_interval_s=5.0, malicious_count=0, params=p)
        a = evaluate_reference(**kw)
        b = evaluate_reference(**kw)
        self.assertEqual(a, b)
        self.assertEqual(a.reference_mode, "LIU_PAPER_REFERENCE_MODE")


class StakeThresholdTests(unittest.TestCase):
    def test_threshold_boundary(self):  # Eq. (4): G(Y) <= eta_s (<= is inclusive)
        # stakes [1,2,3,4] -> Gini 0.25 exactly
        g = paper_stake_gini([1, 2, 3, 4])
        self.assertAlmostEqual(g, 0.25)
        stakes, caps, links = [1.0, 2.0, 3.0, 4.0], [1.0] * 4, uniform_links(4, 100.0)
        common = dict(node_capabilities_ghz=caps, link_rates=links, geographic_gini=0.0,
                      producer_mask=[1, 1, 1, 1], protocol=LiuConsensusProtocol.LIU_QUORUM,
                      block_size_mb=1.0, block_interval_s=5.0, malicious_count=0)
        at = evaluate_reference(node_stakes=stakes, params=_params(eta_s=0.25), **common)
        below = evaluate_reference(node_stakes=stakes, params=_params(eta_s=0.26), **common)
        above = evaluate_reference(node_stakes=stakes, params=_params(eta_s=0.24), **common)
        self.assertTrue(at.stake_constraint_passed)     # exactly at threshold passes (<=)
        self.assertTrue(below.stake_constraint_passed)  # eta_s just above Gini
        self.assertFalse(above.stake_constraint_passed) # eta_s just below Gini


class LambdaNormalizationTests(unittest.TestCase):
    def test_normalized_passes(self):  # int_Xi lambda = K
        self.assertAlmostEqual(assert_lambda_normalized(lambda x, y: 21.0, 21, GeographicGiniConfig(resolution=40)), 21.0)

    def test_unnormalized_detected(self):
        with self.assertRaisesRegex(ValueError, "not normalized"):
            assert_lambda_normalized(lambda x, y: 10.0, 21, GeographicGiniConfig(resolution=40))


class EndToEndGoldenTest(unittest.TestCase):
    """One fully hand-computed S,A example asserted end-to-end (PBFT, K=3, f=0, M=3)."""

    def test_full_reference_evaluation(self):
        # State: 3 producers, equal stakes -> stake Gini 0; uniform caps 2 GHz; uniform links 200 Mbps.
        stakes = [10.0, 10.0, 10.0]
        caps = [2.0, 2.0, 2.0]
        links = uniform_links(3, 200.0)
        # Params: alpha=2e9, beta=1e9, T large, omega=6, chi=200, M=3.
        p = LiuReferenceParameters(
            signature_verification_cycles_alpha=2e9, mac_operation_cycles_beta=1e9,
            network_timeout_s=10_000.0, finality_multiplier_omega=6.0, stake_gini_threshold_eta_s=0.2,
            geographic_gini_threshold_eta_l=0.3, transaction_size_bytes=200, pbft_zyzzyva_batch_m=3,
            transmission_unit_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
        )
        S_B, TI, K, r, c, M = 2.0, 4.0, 3, 200.0, 2.0, 3
        ev = evaluate_reference(
            node_stakes=stakes, node_capabilities_ghz=caps, link_rates=links,
            geographic_gini=0.1,  # <= eta_l = 0.3
            producer_mask=[1, 1, 1], protocol=LiuConsensusProtocol.PBFT,
            block_size_mb=S_B, block_interval_s=TI, malicious_count=0, params=p,
        )
        # --- independent hand computation (Appendix B Eq.15) ---
        # primary cycles = M a + (2M + 4(K-1)) b = 6e9 + 14e9 = 20e9 ; T_V = (1/M) primary/(c*1e9)
        tv = (M * 2e9 + (2 * M + 4 * (K - 1)) * 1e9) / (c * 1e9) / M
        td = 5 * S_B * 8.0 / r          # (1/M) * 5 phases * (M*S_B*8/r)
        tc = tv + td                    # Eq. (7)
        tf = TI + tc                    # Eq. (6)
        capacity = 10000                # floor(2 MB * 1e6 / 200) = 10000  (Eq. 1)
        omega = capacity / TI           # 2500
        self.assertAlmostEqual(ev.stake_gini, 0.0)                 # Eq. (2)
        self.assertAlmostEqual(ev.validation_delay_s, tv, places=9)  # Appendix B T_V
        self.assertAlmostEqual(ev.delivery_delay_s, td, places=9)    # Appendix B T_D
        self.assertAlmostEqual(ev.consensus_latency_s, tc, places=9) # Eq. (7)
        self.assertAlmostEqual(ev.finality_latency_s, tf, places=9)  # Eq. (6)
        self.assertAlmostEqual(ev.finality_limit_s, 6.0 * TI)        # Eq. (8)
        self.assertTrue(ev.c1_decentralization_passed)               # Eq. (12) C1
        self.assertTrue(ev.c2_finality_passed)                       # T_F=7.733 <= 24
        self.assertEqual(ev.tolerated_faults, (K - 1) // 3)          # Eq. (9): floor(2/3)=0
        self.assertTrue(ev.c3_security_passed)                       # f=0 <= 0
        self.assertEqual(ev.transaction_capacity, capacity)          # Eq. (1)
        self.assertAlmostEqual(ev.throughput_tps, omega)            # Eq. (1)
        self.assertAlmostEqual(ev.reward, omega)                     # Eq. (13): feasible
        self.assertTrue(ev.feasible)


class HandCalculatedEquationTests(unittest.TestCase):
    def test_pbft_equation_15_fixture(self):
        result = PBFTAnalyticalModel().evaluate(analytical_input())

        # M=2, K=4, f=0: O_p=18e9 and O_i=16e9 cycles per batch.
        # T_V=max(O_k/c_k)/M=18/2=9 s. Each of five phases is
        # M*S/R=2/10=0.2 s, so T_D=5*0.2/2=0.5 s.
        self.assertAlmostEqual(result.delivery_delay_s, 0.5)
        self.assertAlmostEqual(result.validation_delay_s, 9.0)
        self.assertAlmostEqual(result.consensus_delay_s, 9.5)
        self.assertAlmostEqual(result.finality_delay_s, 11.5)
        self.assertEqual(result.tolerated_faults, 1)

    def test_zyzzyva_fast_equation_16_fixture(self):
        source = analytical_input(
            LiuConsensusProtocol.ZYZZYVA,
            zyzzyva_path=ZyzzyvaAnalyticalPath.FAST,
        )
        result = ZyzzyvaAnalyticalModel().evaluate(source)

        # O_p=9e9, O_i=5e9; T_V=9/2=4.5 s.
        # Three 0.2-second batch phases divided by M=2 give 0.3 s.
        self.assertAlmostEqual(result.delivery_delay_s, 0.3)
        self.assertAlmostEqual(result.validation_delay_s, 4.5)
        self.assertAlmostEqual(result.consensus_delay_s, 4.8)
        self.assertAlmostEqual(result.finality_delay_s, 6.8)
        self.assertEqual(result.tolerated_faults, 1)

    def test_zyzzyva_recovery_equation_17_fixture(self):
        source = analytical_input(
            LiuConsensusProtocol.ZYZZYVA,
            faulty_replica_count=1,
            zyzzyva_path=ZyzzyvaAnalyticalPath.RECOVERY,
            recovery_delay_s=2.0,
        )
        result = ZyzzyvaAnalyticalModel().evaluate(source)

        # O_p=14e9, O_i=9e9; T_V=14/2=7 s.
        # (five 0.2-second phases + t_r=2)/M=1.5 s.
        self.assertAlmostEqual(result.delivery_delay_s, 1.5)
        self.assertAlmostEqual(result.validation_delay_s, 7.0)
        self.assertAlmostEqual(result.consensus_delay_s, 8.5)
        self.assertAlmostEqual(result.finality_delay_s, 10.5)
        self.assertEqual(result.tolerated_faults, 1)

    def test_liu_quorum_two_hop_fixture(self):
        source = analytical_input(LiuConsensusProtocol.LIU_QUORUM)
        result = LiuQuorumAnalyticalModel().evaluate(source)

        # O_k=alpha+2*beta=3e9 cycles, T_V=3 s.
        # Request and reply are each S/R=1/10=0.1 s.
        self.assertAlmostEqual(result.delivery_delay_s, 0.2)
        self.assertAlmostEqual(result.validation_delay_s, 3.0)
        self.assertAlmostEqual(result.consensus_delay_s, 3.2)
        self.assertAlmostEqual(result.finality_delay_s, 5.2)
        self.assertEqual(result.tolerated_faults, 0)

    def test_si_unit_policy_applies_explicit_eight_bit_factor(self):
        literal = PBFTAnalyticalModel().evaluate(analytical_input(alpha=0, beta=0))
        si = PBFTAnalyticalModel().evaluate(
            analytical_input(
                alpha=0,
                beta=0,
                transmission_policy=LiuTransmissionUnitPolicy.SI_MEGABYTE_TO_MEGABIT_V1,
            )
        )

        self.assertAlmostEqual(si.delivery_delay_s, 8 * literal.delivery_delay_s)

    def test_transaction_size_is_explicit_but_not_in_appendix_b_equations(self):
        small_transaction = PBFTAnalyticalModel().evaluate(
            analytical_input(transaction_size_bytes=200)
        )
        large_transaction = PBFTAnalyticalModel().evaluate(
            analytical_input(transaction_size_bytes=800)
        )

        self.assertEqual(
            small_transaction.consensus_delay_s,
            large_transaction.consensus_delay_s,
        )
        self.assertEqual(
            dict(small_transaction.diagnostics)["transaction_size_usage"],
            "not_used_by_appendix_b_equations_15_17",
        )


class AnalyticalMonotonicityTests(unittest.TestCase):
    def test_homogeneous_capabilities_match_hand_fixture(self):
        result = PBFTAnalyticalModel().evaluate(
            analytical_input(capabilities=(2.0, 2.0, 2.0, 2.0))
        )
        self.assertAlmostEqual(result.validation_delay_s, 4.5)

    def test_heterogeneous_capability_uses_slowest_role_adjusted_replica(self):
        homogeneous = PBFTAnalyticalModel().evaluate(analytical_input())
        heterogeneous = PBFTAnalyticalModel().evaluate(
            analytical_input(capabilities=(1.0, 2.0, 0.5, 1.0))
        )

        # Primary: 18/2/2=4.5 s; slow backup: 16/0.5/2=16 s.
        self.assertAlmostEqual(heterogeneous.validation_delay_s, 16.0)
        self.assertGreater(heterogeneous.validation_delay_s, homogeneous.validation_delay_s)

    def test_increasing_every_capability_does_not_increase_validation_delay(self):
        slow = PBFTAnalyticalModel().evaluate(
            analytical_input(capabilities=(1.0, 1.0, 1.0, 1.0))
        )
        fast = PBFTAnalyticalModel().evaluate(
            analytical_input(capabilities=(2.0, 2.0, 2.0, 2.0))
        )
        self.assertLessEqual(fast.validation_delay_s, slow.validation_delay_s)

    def test_increasing_fixed_link_rate_does_not_increase_delivery_delay(self):
        slow = PBFTAnalyticalModel().evaluate(
            analytical_input(links=fixed_links(rate_mbps=5.0))
        )
        fast = PBFTAnalyticalModel().evaluate(
            analytical_input(links=fixed_links(rate_mbps=20.0))
        )
        self.assertLessEqual(fast.delivery_delay_s, slow.delivery_delay_s)

    def test_increasing_block_size_does_not_decrease_delivery_delay(self):
        small = PBFTAnalyticalModel().evaluate(analytical_input(block_size_mb=0.5))
        large = PBFTAnalyticalModel().evaluate(analytical_input(block_size_mb=4.0))
        self.assertGreaterEqual(large.delivery_delay_s, small.delivery_delay_s)

    def test_timeout_caps_each_delivery_phase_as_in_paper(self):
        result = PBFTAnalyticalModel().evaluate(
            analytical_input(links=fixed_links(rate_mbps=0.01), timeout_s=0.25)
        )
        self.assertAlmostEqual(result.delivery_delay_s, 5 * 0.25 / 2)


class AnalyticalDomainValidationTests(unittest.TestCase):
    def test_pbft_and_zyzzyva_have_equal_liu_fault_bounds(self):
        pbft = PBFTAnalyticalModel().evaluate(analytical_input())
        zyzzyva = ZyzzyvaAnalyticalModel().evaluate(
            analytical_input(
                LiuConsensusProtocol.ZYZZYVA,
                zyzzyva_path=ZyzzyvaAnalyticalPath.FAST,
            )
        )
        self.assertEqual(pbft.tolerated_faults, zyzzyva.tolerated_faults)
        self.assertEqual(pbft.tolerated_faults, 1)

    def test_wrong_model_protocol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires protocol"):
            PBFTAnalyticalModel().evaluate(
                analytical_input(LiuConsensusProtocol.LIU_QUORUM)
            )

    def test_invalid_units_and_shapes_are_rejected(self):
        valid = analytical_input()
        invalid_changes = (
            {"block_size_mb": 0},
            {"transaction_size_bytes": 0},
            {"batch_size_m": 0},
            {"block_interval_s": 0},
            {"computational_capabilities_ghz": (1.0, 1.0, 1.0)},
            {"signature_verification_cycles_alpha": -1},
            {"mac_operation_cycles_beta": -1},
            {"network_timeout_s": 0},
            {"directed_link_rates_mbps": fixed_links(node_count=3)},
            {"transmission_unit_policy": "implicit"},
            {"protocol": "PBFT"},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(valid, **changes)

    def test_invalid_roles_faults_and_protocol_options_are_rejected(self):
        valid = analytical_input()
        with self.assertRaises(ValueError):
            replace(valid, validator_ids=(0, 1, 1, 3))
        with self.assertRaises(ValueError):
            replace(valid, primary_validator_id=0)
        with self.assertRaises(ValueError):
            replace(valid, faulty_replica_count=2)
        with self.assertRaises(ValueError):
            replace(valid, zyzzyva_path=ZyzzyvaAnalyticalPath.FAST)
        with self.assertRaises(ValueError):
            analytical_input(
                LiuConsensusProtocol.ZYZZYVA,
                faulty_replica_count=1,
                zyzzyva_path=ZyzzyvaAnalyticalPath.FAST,
            )

    def test_liu_quorum_rejects_batching_faults_primary_and_recovery(self):
        valid = analytical_input(LiuConsensusProtocol.LIU_QUORUM)
        with self.assertRaises(ValueError):
            replace(valid, batch_size_m=2)
        with self.assertRaises(ValueError):
            replace(valid, faulty_replica_count=1)
        with self.assertRaises(ValueError):
            replace(valid, primary_validator_id=1)
        with self.assertRaises(ValueError):
            replace(valid, recovery_delay_s=1.0)

    def test_serialization_hashing_and_immutability_are_deterministic(self):
        first_input = analytical_input()
        second_input = analytical_input()
        first = PBFTAnalyticalModel().evaluate(first_input)
        second = PBFTAnalyticalModel().evaluate(second_input)

        self.assertEqual(first_input.canonical_json(), second_input.canonical_json())
        self.assertEqual(first_input.deterministic_hash(), second_input.deterministic_hash())
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())
        with self.assertRaises(FrozenInstanceError):
            first.consensus_delay_s = 0


if __name__ == "__main__":
    unittest.main()
