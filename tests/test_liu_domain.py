import hashlib
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


from collections import deque
from types import SimpleNamespace

SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Chain.TransactionFactory import Transaction, TransactionFactory
from Chain.ValidatorSet import ValidatorSet
from Parameters import Parameters
from Liu import (
    ConstraintResult,
    EpochConfiguration,
    FeasibilityResult,
    LinkStateMatrix,
    LiuAction,
    LiuConsensusProtocol,
    LiuState,
    SpatialProfile,
    SpatialProfileSet,
    ThreatScenario,
)


def spatial_profiles() -> SpatialProfileSet:
    return SpatialProfileSet.from_coordinates(
        ((0.0, 0.25), (0.5, 1.0), (1.0, 0.0)),
        region_width_km=1.0,
        region_height_km=1.0,
    )


def link_matrix() -> LinkStateMatrix:
    return LinkStateMatrix.from_rows(
        (
            (None, 10, 20),
            (30, None, 40),
            (50, 60, None),
        )
    )


def liu_state() -> LiuState:
    return LiuState(
        transaction_size_bytes=200,
        stakes_tokens=(1, 2, 3),
        spatial_profiles=spatial_profiles(),
        computational_capabilities_ghz=(10, 20, 30),
        link_state_matrix=link_matrix(),
    )

class LiuStateTests(unittest.TestCase):
    def test_valid_state_exposes_complete_liu_shape(self):
        state = liu_state()

        self.assertEqual(state.node_count, 3)
        self.assertEqual(state.transaction_size_bytes, 200.0)
        self.assertEqual(state.stakes_tokens, (1.0, 2.0, 3.0))
        self.assertEqual(state.computational_capabilities_ghz, (10.0, 20.0, 30.0))
        self.assertEqual(state.link_state_matrix.rate(0, 2), 20.0)

    def test_all_state_components_must_have_the_same_n(self):
        with self.assertRaisesRegex(ValueError, "same N"):
            LiuState(200, (1, 2), spatial_profiles(), (10, 20), link_matrix())
        with self.assertRaisesRegex(ValueError, "same N"):
            LiuState(200, (1, 2, 3), spatial_profiles(), (10, 20), link_matrix())

    def test_state_rejects_invalid_physical_values(self):
        with self.assertRaisesRegex(ValueError, "transaction_size_bytes must be positive"):
            LiuState(0, (1, 2, 3), spatial_profiles(), (10, 20, 30), link_matrix())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LiuState(200, (1, -1, 3), spatial_profiles(), (10, 20, 30), link_matrix())
        with self.assertRaisesRegex(ValueError, "positive"):
            LiuState(200, (1, 2, 3), spatial_profiles(), (10, 0, 30), link_matrix())


class LiuActionTests(unittest.TestCase):
    def test_valid_action_normalizes_validator_mask_to_node_order(self):
        action = LiuAction(5, 3, (4, 0, 2), LiuConsensusProtocol.ZYZZYVA, 4, 0.5)

        self.assertEqual(action.validator_ids, (0, 2, 4))
        self.assertEqual(action.consensus_protocol.value, "ZYZZYVA")
        self.assertEqual(action.block_size_mb, 4.0)
        self.assertEqual(action.block_interval_s, 0.5)

    def test_action_enforces_n_k_consistency_and_exact_validator_count(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            LiuAction(2, 3, (0, 1, 2), LiuConsensusProtocol.PBFT, 1, 1)
        with self.assertRaisesRegex(ValueError, "exactly K"):
            LiuAction(4, 3, (0, 1), LiuConsensusProtocol.PBFT, 1, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            LiuAction(4, 2, (0, 4), LiuConsensusProtocol.PBFT, 1, 1)

    def test_action_rejects_duplicate_validators(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            LiuAction(4, 3, (0, 1, 1), LiuConsensusProtocol.PBFT, 1, 1)

    def test_action_rejects_invalid_protocol(self):
        with self.assertRaisesRegex(ValueError, "LiuConsensusProtocol"):
            LiuAction(3, 2, (0, 1), "PBFT", 1, 1)

    def test_action_rejects_invalid_block_size_and_interval(self):
        with self.assertRaisesRegex(ValueError, "block_size_mb must be positive"):
            LiuAction(3, 2, (0, 1), LiuConsensusProtocol.PBFT, 0, 1)
        with self.assertRaisesRegex(ValueError, "block_interval_s must be positive"):
            LiuAction(3, 2, (0, 1), LiuConsensusProtocol.PBFT, 1, -0.5)
        with self.assertRaisesRegex(ValueError, "0.2 MB"):
            LiuAction(3, 2, (0, 1), LiuConsensusProtocol.PBFT, 0.3, 1)
        with self.assertRaisesRegex(ValueError, "0.5 second"):
            LiuAction(3, 2, (0, 1), LiuConsensusProtocol.PBFT, 1, 0.75)


class SpatialDomainTests(unittest.TestCase):
    def test_profiles_are_complete_ordered_and_inside_bounds(self):
        profiles = spatial_profiles()

        self.assertEqual(profiles.node_count, 3)
        self.assertEqual(profiles.profile_for(1), SpatialProfile(1, 0.5, 1.0))
        with self.assertRaisesRegex(ValueError, "complete and ordered"):
            SpatialProfileSet((SpatialProfile(1, 0, 0),), 1, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            SpatialProfileSet((SpatialProfile(0, 1.01, 0),), 1, 1)

    def test_coordinates_and_region_dimensions_must_be_valid(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SpatialProfile(0, -0.1, 0)
        with self.assertRaisesRegex(ValueError, "region_width_km must be positive"):
            SpatialProfileSet((SpatialProfile(0, 0, 0),), 0, 1)


class LinkStateTests(unittest.TestCase):
    def test_directed_rates_and_masked_self_links(self):
        links = link_matrix()

        self.assertEqual(links.rate(0, 1), 10.0)
        self.assertEqual(links.rate(1, 0), 30.0)
        with self.assertRaisesRegex(ValueError, "self-links"):
            links.rate(0, 0)

    def test_matrix_must_be_exact_n_by_n(self):
        with self.assertRaisesRegex(ValueError, "2x2"):
            LinkStateMatrix(2, ((None, 10),))
        with self.assertRaisesRegex(ValueError, "2x2"):
            LinkStateMatrix(2, ((None, 10, 20), (30, None, 40)))

    def test_diagonal_is_none_and_off_diagonal_is_positive_mbps(self):
        with self.assertRaisesRegex(ValueError, "masked"):
            LinkStateMatrix(2, ((0, 10), (20, None)))
        with self.assertRaisesRegex(ValueError, "explicit Mbps"):
            LinkStateMatrix(2, ((None, None), (20, None)))
        with self.assertRaisesRegex(ValueError, "positive"):
            LinkStateMatrix(2, ((None, 0), (20, None)))


class EpochAndThreatTests(unittest.TestCase):
    def test_epoch_configuration_is_an_immutable_validator_snapshot(self):
        validators = ValidatorSet((0, 2))
        epoch = EpochConfiguration(7, validators, LiuConsensusProtocol.LIU_QUORUM, 2, 1)

        self.assertIs(epoch.validator_set, validators)
        self.assertEqual(epoch.to_dict()["validator_ids"], [0, 2])
        with self.assertRaises(FrozenInstanceError):
            epoch.block_size_mb = 4

    def test_threat_scenario_is_explicit_ordered_and_validated(self):
        threat = ThreatScenario(5, (4, 1))

        self.assertEqual(threat.malicious_node_ids, (1, 4))
        self.assertEqual(threat.malicious_validator_count((0, 1, 2)), 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            ThreatScenario(5, (1, 1))
        with self.assertRaisesRegex(ValueError, "outside"):
            ThreatScenario(5, (5,))


class FeasibilityTests(unittest.TestCase):
    def test_feasibility_is_derived_from_immutable_constraint_results(self):
        result = FeasibilityResult(
            (
                ConstraintResult("stake_gini", True, 0.1, 0.2),
                ConstraintResult("security", False, 2, 1, "too many malicious validators"),
            )
        )

        self.assertFalse(result.feasible)
        self.assertFalse(result.to_dict()["feasible"])
        with self.assertRaises(FrozenInstanceError):
            result.constraint_results[0].satisfied = False
        with self.assertRaisesRegex(ValueError, "unique"):
            FeasibilityResult((ConstraintResult("c1", True), ConstraintResult("c1", False)))


class CanonicalSerializationTests(unittest.TestCase):
    def test_serialization_and_sha256_are_deterministic(self):
        first = LiuAction(5, 3, (4, 0, 2), LiuConsensusProtocol.PBFT, 4, 0.5)
        second = LiuAction(5, 3, (2, 4, 0), LiuConsensusProtocol.PBFT, 4.0, 0.5)
        expected_json = (
            '{"block_interval_s":0.5,"block_size_mb":4.0,"consensus_protocol":"PBFT",'
            '"node_count":5,"validator_count":3,"validator_ids":[0,2,4]}'
        )

        self.assertEqual(first.canonical_json(), expected_json)
        self.assertEqual(second.canonical_json(), expected_json)
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())
        self.assertEqual(first.deterministic_hash(), hashlib.sha256(expected_json.encode("utf-8")).hexdigest())

    def test_nested_state_serialization_is_stable(self):
        first = liu_state()
        second = liu_state()

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())

    def test_domain_objects_defensively_freeze_input_sequences(self):
        validator_ids = [2, 0]
        rates = [[None, 10], [20, None]]
        action = LiuAction(3, 2, validator_ids, LiuConsensusProtocol.PBFT, 1, 1)
        links = LinkStateMatrix(2, rates)

        validator_ids.append(1)
        rates[0][1] = 999
        self.assertEqual(action.validator_ids, (0, 2))
        self.assertEqual(links.rate(0, 1), 10.0)
        with self.assertRaises(FrozenInstanceError):
            action.block_size_mb = 2
        with self.assertRaises(TypeError):
            links.rates_mbps[0][1] = 999


BYTES_PER_MB_EXACT = 1_000_000


def size_in_exact_bytes(mb: float) -> int:
    return round(mb * BYTES_PER_MB_EXACT)


def _fill(block_size_mb: float, tx_size_mb: float, count: int):
    Parameters.application = {"Nn": 1, "transaction_model": "local"}
    Parameters.data = {"base_block_size": 0.0}
    pool = deque(Transaction(0, i, 0.0, tx_size_mb) for i in range(count))
    configuration = SimpleNamespace(block_size=block_size_mb)
    transactions, _ = TransactionFactory._get_transactions_from_pool(configuration, pool, 1.0)
    size_mb = sum(t.size for t in transactions)
    return transactions, size_mb


class HelperTests(unittest.TestCase):
    def test_size_in_exact_bytes_maps_paper_grid_and_tiny_tx_exactly(self):
        # Would drift under naive int() due to 0.0002 float representation
        self.assertEqual(size_in_exact_bytes(0.0002), 200)
        for grid_mb in (0.2, 0.4, 0.6, 1.0, 4.0, 8.0):
            self.assertEqual(size_in_exact_bytes(grid_mb), int(grid_mb * BYTES_PER_MB_EXACT))


class CaseABlockCapacityIs1000(unittest.TestCase):
    def test_02mb_over_200b_fits_exactly_1000_transactions(self):
        transactions, size_mb = _fill(block_size_mb=0.2, tx_size_mb=0.0002, count=1500)
        self.assertEqual(len(transactions), 1000)  # Case A: capacity = floor(0.2MB/200B)
        self.assertEqual(size_mb, 0.2)             # exact, no float drift


class CaseBExactFillIsAccepted(unittest.TestCase):
    def test_last_tx_of_exact_fill_is_included(self):
        transactions, _ = _fill(block_size_mb=0.2, tx_size_mb=0.0002, count=1000)
        self.assertEqual(len(transactions), 1000)


class CaseC1001stIsRejected(unittest.TestCase):
    def test_supplying_1001_yields_1000_included_and_leaves_one_pending(self):
        Parameters.application = {"Nn": 1, "transaction_model": "local"}
        Parameters.data = {"base_block_size": 0.0}
        pool = deque(Transaction(0, i, 0.0, 0.0002) for i in range(1001))
        configuration = SimpleNamespace(block_size=0.2)
        transactions, _ = TransactionFactory._get_transactions_from_pool(configuration, pool, 1.0)
        self.assertEqual(len(transactions), 1000)
        self.assertEqual(len(pool), 1)  # 1001st stays in mempool


class CaseD04MbCapacityIs2000(unittest.TestCase):
    def test_04mb_over_200b_fits_2000(self):
        transactions, _ = _fill(block_size_mb=0.4, tx_size_mb=0.0002, count=2500)
        self.assertEqual(len(transactions), 2000)


class CaseENonDividingTxSize(unittest.TestCase):
    def test_capacity_is_floor_when_tx_size_does_not_divide_block(self):
        # 200000 B / 300 B = 666.66... -> floor = 666, remainder 200 B unused
        transactions, size_mb = _fill(block_size_mb=0.2, tx_size_mb=0.0003, count=1000)
        self.assertEqual(len(transactions), 666)
        self.assertEqual(size_in_exact_bytes(size_mb), 666 * 300)  # exact bytes, no drift


class CaseFRepeatedAdditionsMatchFloor(unittest.TestCase):
    def test_no_floating_point_drift_across_paper_grid_and_tx_range(self):
        # Sweep a range: capacity must always equal floor(S_B * 1e6 / tx_bytes) exactly.
        for block_mb in (0.2, 0.4, 0.6, 1.0, 4.0, 8.0):
            for tx_mb in (0.0002, 0.0003, 0.0005, 0.001):
                tx_bytes = size_in_exact_bytes(tx_mb)
                cap_bytes = size_in_exact_bytes(block_mb)
                expected = cap_bytes // tx_bytes
                transactions, _ = _fill(block_mb, tx_mb, count=expected + 5)
                self.assertEqual(
                    len(transactions),
                    expected,
                    msg=f"block={block_mb}MB tx={tx_mb}MB expected {expected} got {len(transactions)}",
                )


if __name__ == "__main__":
    unittest.main()
