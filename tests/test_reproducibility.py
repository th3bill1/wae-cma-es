import unittest
from pathlib import Path

import pandas as pd

import run_experiments
from src.analysis import create_wilcoxon_tests
from src.experiment import run_single_experiment


class CMAESReproducibilityTests(unittest.TestCase):
    def test_p_sigma_zero_initialization_has_zero_norm(self):
        result = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="zero",
            generator_name="pcg64",
            seed=1,
            max_evaluations=20,
        )

        self.assertEqual(result.p_sigma_norm_history[0], 0.0)

    def test_p_sigma_random_initialization_has_nonzero_norm(self):
        result = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="random",
            generator_name="pcg64",
            seed=1,
            max_evaluations=20,
        )

        self.assertGreater(result.p_sigma_norm_history[0], 0.0)

    def test_same_seed_is_deterministic(self):
        first = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="random",
            generator_name="mt19937",
            seed=3,
            max_evaluations=100,
        )
        second = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="random",
            generator_name="mt19937",
            seed=3,
            max_evaluations=100,
        )

        self.assertEqual(first.history, second.history)
        self.assertEqual(first.sigma_history, second.sigma_history)
        self.assertEqual(first.p_sigma_norm_history, second.p_sigma_norm_history)

    def test_sampling_rng_is_shared_between_variants(self):
        zero = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="zero",
            generator_name="pcg64",
            seed=7,
            max_evaluations=8,
        )
        random = run_single_experiment(
            "sphere",
            dimension=2,
            p_sigma_mode="random",
            generator_name="pcg64",
            seed=7,
            max_evaluations=8,
        )

        self.assertEqual(zero.mean_history[0], random.mean_history[0])
        self.assertEqual(zero.mean_history[1], random.mean_history[1])
        self.assertEqual(zero.p_sigma_norm_history[0], 0.0)
        self.assertGreater(random.p_sigma_norm_history[0], 0.0)


class ExperimentConfigurationTests(unittest.TestCase):
    def test_full_experiment_grid_size(self):
        total = (
            len(run_experiments.FUNCTIONS)
            * len(run_experiments.DIMENSIONS)
            * len(run_experiments.P_SIGMA_MODES)
            * len(run_experiments.GENERATORS)
            * len(run_experiments.SEEDS)
        )

        self.assertEqual(total, 1800)


class WilcoxonAnalysisTests(unittest.TestCase):
    def test_degenerate_wilcoxon_result_has_explicit_status(self):
        rows = []
        for seed in [1, 2, 3]:
            rows.append(
                {
                    "function": "gaussian_noise",
                    "dimension": 2,
                    "p_sigma_mode": "zero",
                    "generator": "pcg64",
                    "seed": seed,
                    "best_f": 0.5,
                    "evaluations": 10,
                    "reached_target": False,
                }
            )
            rows.append(
                {
                    "function": "gaussian_noise",
                    "dimension": 2,
                    "p_sigma_mode": "random",
                    "generator": "pcg64",
                    "seed": seed,
                    "best_f": 0.5,
                    "evaluations": 10,
                    "reached_target": False,
                }
            )

        results_path = Path("tests") / "_wilcoxon_input.csv"
        output_path = Path("tests") / "_wilcoxon_output.csv"

        try:
            pd.DataFrame(rows).to_csv(results_path, index=False)
            create_wilcoxon_tests(str(results_path), str(output_path))

            output = pd.read_csv(output_path)
        finally:
            results_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertEqual(len(output), 1)
        self.assertEqual(output.loc[0, "status"], "no_difference")
        self.assertEqual(output.loc[0, "p_value"], 1.0)
        self.assertEqual(output.loc[0, "n_pairs"], 3)
        self.assertEqual(output.loc[0, "n_zero_differences"], 3)


if __name__ == "__main__":
    unittest.main()
