import random
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path


SIMULATOR_ROOT = Path(__file__).resolve().parents[1] / "src" / "Simulator"
sys.path.insert(0, str(SIMULATOR_ROOT))

from Liu import (
    GridSpatialIntensityModel,
    LinkFSMCState,
    LinkRateLevels,
    LinkStateMatrix,
    LinkTransitionMatrix,
    LinkTransitionTensor,
    SpatialProfile,
    SpatialProfileSet,
)


def profiles(coordinates):
    return SpatialProfileSet.from_coordinates(
        coordinates,
        region_width_km=1.0,
        region_height_km=1.0,
    )


def two_node_fsmc(matrix=None):
    levels = LinkRateLevels((10.0, 20.0))
    transition_matrix = matrix or LinkTransitionMatrix(
        probabilities=((0.75, 0.25), (0.40, 0.60))
    )
    tensor = LinkTransitionTensor.shared(2, transition_matrix)
    links = LinkStateMatrix(
        node_count=2,
        rates_mbps=((None, 10.0), (20.0, None)),
    )
    return LinkFSMCState(levels, tensor, links)


class GeographicIntensityTests(unittest.TestCase):
    def test_uniform_grid_intensity_has_zero_geographic_gini(self):
        spatial_profiles = profiles(
            ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
        )
        result = GridSpatialIntensityModel(spatial_profiles, 2, 2).evaluate(
            (0, 1, 2, 3)
        )

        self.assertEqual(result.cell_validator_counts, (1, 1, 1, 1))
        self.assertEqual(result.cell_intensities_per_km2, (4.0, 4.0, 4.0, 4.0))
        self.assertAlmostEqual(result.integrated_intensity, 4.0)
        self.assertAlmostEqual(result.geographic_gini, 0.0)
        self.assertEqual(result.estimator_version, "grid_v1")

    def test_concentrated_validators_have_larger_geographic_gini(self):
        uniform_profiles = profiles(
            ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
        )
        concentrated_profiles = profiles(
            ((0.10, 0.10), (0.12, 0.10), (0.10, 0.12), (0.12, 0.12))
        )

        uniform = GridSpatialIntensityModel(uniform_profiles, 2, 2).evaluate(
            range(4)
        )
        concentrated = GridSpatialIntensityModel(
            concentrated_profiles, 2, 2
        ).evaluate(range(4))

        self.assertAlmostEqual(uniform.geographic_gini, 0.0)
        self.assertAlmostEqual(concentrated.geographic_gini, 0.75)
        self.assertGreater(concentrated.geographic_gini, uniform.geographic_gini)

    def test_result_is_order_invariant_and_deterministic(self):
        spatial_profiles = profiles(
            ((0.10, 0.10), (0.60, 0.10), (0.10, 0.60), (0.60, 0.60))
        )
        model = GridSpatialIntensityModel(spatial_profiles, 2, 2)

        first = model.evaluate((3, 1, 0))
        reordered = model.evaluate((0, 3, 1))
        repeated = model.evaluate((3, 1, 0))

        self.assertEqual(first, reordered)
        self.assertEqual(first, repeated)
        self.assertEqual(first.canonical_json(), reordered.canonical_json())
        self.assertEqual(first.deterministic_hash(), reordered.deterministic_hash())
        with self.assertRaises(FrozenInstanceError):
            first.geographic_gini = 1.0

    def test_intensity_integrates_to_selected_validator_count(self):
        spatial_profiles = profiles(
            ((0.01, 0.01), (0.20, 0.30), (0.50, 0.50), (1.00, 1.00))
        )
        result = GridSpatialIntensityModel(spatial_profiles, 7, 5).evaluate(
            (0, 1, 2, 3)
        )

        reconstructed_integral = (
            sum(result.cell_intensities_per_km2) * result.cell_area_km2
        )
        self.assertAlmostEqual(reconstructed_integral, 4.0)
        self.assertAlmostEqual(result.integrated_intensity, 4.0)
        self.assertAlmostEqual(result.integration_error, 0.0, places=12)

    def test_grid_resolution_sensitivity_fixture(self):
        spatial_profiles = profiles(((0.10, 0.10), (0.12, 0.12)))

        coarse = GridSpatialIntensityModel(spatial_profiles, 1, 1).evaluate((0, 1))
        medium = GridSpatialIntensityModel(spatial_profiles, 2, 2).evaluate((0, 1))
        fine = GridSpatialIntensityModel(spatial_profiles, 4, 4).evaluate((0, 1))

        self.assertAlmostEqual(coarse.geographic_gini, 0.0)
        self.assertAlmostEqual(medium.geographic_gini, 0.75)
        self.assertAlmostEqual(fine.geographic_gini, 0.9375)

    def test_grid_rejects_invalid_resolution(self):
        for rows, columns in (
            (0, 2),
            (2, 0),
            (-1, 2),
            (2, 1.5),
            (True, 2),
        ):
            with self.subTest(rows=rows, columns=columns):
                with self.assertRaises((TypeError, ValueError)):
                    GridSpatialIntensityModel(profiles(((0.5, 0.5),)), rows, columns)

    def test_grid_rejects_invalid_selected_validator_sets(self):
        model = GridSpatialIntensityModel(
            profiles(((0.25, 0.25), (0.75, 0.75))), 2, 2
        )

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            model.evaluate(())
        with self.assertRaisesRegex(ValueError, "unique"):
            model.evaluate((0, 0))
        with self.assertRaisesRegex(ValueError, "outside"):
            model.evaluate((0, 2))


class LinkFSMCTests(unittest.TestCase):
    def test_transition_matrix_rejects_invalid_probability_rows(self):
        invalid_matrices = (
            ((0.5, 0.4), (0.4, 0.6)),
            ((1.0, 0.0),),
            ((1.1, -0.1), (0.5, 0.5)),
            ((float("nan"), 0.0), (0.5, 0.5)),
        )
        for probabilities in invalid_matrices:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    LinkTransitionMatrix(probabilities)

    def test_rates_must_map_to_declared_levels(self):
        levels = LinkRateLevels((10.0, 20.0))
        tensor = LinkTransitionTensor.shared(
            2, LinkTransitionMatrix(((1.0, 0.0), (0.0, 1.0)))
        )

        with self.assertRaisesRegex(ValueError, "declared link-rate levels"):
            LinkFSMCState(
                levels,
                tensor,
                LinkStateMatrix(2, ((None, 15.0), (20.0, None))),
            )

    def test_per_link_models_support_asymmetric_directed_links(self):
        levels = LinkRateLevels((10.0, 20.0))
        always_high = LinkTransitionMatrix(((0.0, 1.0), (0.0, 1.0)))
        always_low = LinkTransitionMatrix(((1.0, 0.0), (1.0, 0.0)))
        tensor = LinkTransitionTensor(
            node_count=2,
            matrices=((None, always_high), (always_low, None)),
        )
        state = LinkFSMCState(
            levels,
            tensor,
            LinkStateMatrix(2, ((None, 10.0), (20.0, None))),
        )

        transitioned = state.transition(random.Random(9))

        self.assertEqual(transitioned.rates_mbps, ((None, 20.0), (10.0, None)))

    def test_same_seed_produces_identical_trajectory(self):
        initial = two_node_fsmc()

        def trajectory(seed):
            rng = random.Random(seed)
            state = initial
            hashes = []
            for _ in range(30):
                state = state.advance(rng)
                hashes.append(state.current_links.deterministic_hash())
            return hashes

        self.assertEqual(trajectory(8128), trajectory(8128))

    def test_different_seeds_can_produce_different_trajectories(self):
        matrix = LinkTransitionMatrix(((0.5, 0.5), (0.5, 0.5)))
        initial = two_node_fsmc(matrix)

        def trajectory(seed):
            rng = random.Random(seed)
            state = initial
            values = []
            for _ in range(20):
                state = state.advance(rng)
                values.append(state.current_links.rates_mbps)
            return values

        self.assertNotEqual(trajectory(1), trajectory(2))

    def test_empirical_transition_frequency_matches_configured_row(self):
        state = two_node_fsmc(
            LinkTransitionMatrix(((0.75, 0.25), (0.75, 0.25)))
        )
        rng = random.Random(1837413)
        high_count = 0
        samples = 10_000

        for _ in range(samples):
            transitioned = state.transition(rng)
            high_count += transitioned.rate(0, 1) == 20.0

        self.assertAlmostEqual(high_count / samples, 0.25, delta=0.02)

    def test_diagonal_remains_masked_and_previous_state_is_not_mutated(self):
        state = two_node_fsmc()
        original_links = state.current_links
        original_serialization = original_links.canonical_json()

        transitioned = state.transition(random.Random(44))

        self.assertIsNone(transitioned.rates_mbps[0][0])
        self.assertIsNone(transitioned.rates_mbps[1][1])
        self.assertIs(state.current_links, original_links)
        self.assertEqual(state.current_links.canonical_json(), original_serialization)
        self.assertIsNot(transitioned, original_links)

    def test_transition_does_not_consume_module_global_random_state(self):
        state = two_node_fsmc()
        global_state = random.getstate()

        state.transition(random.Random(1234))

        self.assertEqual(random.getstate(), global_state)

    def test_transition_requires_an_external_rng_and_state_is_immutable(self):
        state = two_node_fsmc()

        with self.assertRaisesRegex(ValueError, "external RNG"):
            state.transition(None)
        with self.assertRaises(FrozenInstanceError):
            state.current_links = LinkStateMatrix(
                2, ((None, 20.0), (10.0, None))
            )

    def test_serialization_and_hashing_are_deterministic(self):
        first = two_node_fsmc()
        second = two_node_fsmc()

        self.assertEqual(first, second)
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.deterministic_hash(), second.deterministic_hash())


if __name__ == "__main__":
    unittest.main()
