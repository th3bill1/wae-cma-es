from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def create_summary(results_path: str, output_path: str) -> None:
    df = pd.read_csv(results_path)

    summary = (
        df.groupby(["function", "dimension", "p_sigma_mode", "generator"])
        .agg(
            median_best_f=("best_f", "median"),
            mean_best_f=("best_f", "mean"),
            median_evaluations=("evaluations", "median"),
            mean_evaluations=("evaluations", "mean"),
            success_rate=("reached_target", "mean"),
        )
        .reset_index()
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def create_wilcoxon_tests(results_path: str, output_path: str) -> None:
    df = pd.read_csv(results_path)
    rows = []

    grouped = df.groupby(["function", "dimension", "generator"])

    for (function_name, dimension, generator), group in grouped:
        zero = group[group["p_sigma_mode"] == "zero"][["seed", "best_f"]]
        random = group[group["p_sigma_mode"] == "random"][["seed", "best_f"]]

        paired = zero.merge(
            random,
            on="seed",
            suffixes=("_zero", "_random"),
        ).sort_values("seed")

        zero_values = pd.to_numeric(paired["best_f_zero"], errors="coerce").to_numpy()
        random_values = pd.to_numeric(paired["best_f_random"], errors="coerce").to_numpy()
        finite_mask = np.isfinite(zero_values) & np.isfinite(random_values)
        zero_values = zero_values[finite_mask]
        random_values = random_values[finite_mask]

        differences = zero_values - random_values
        n_pairs = len(differences)
        n_zero_differences = int(np.sum(differences == 0.0))

        statistic = np.nan
        p_value = np.nan
        status = "ok"
        interpretation = "Wilcoxon signed-rank test completed."

        if n_pairs < 2:
            status = "insufficient_pairs"
            interpretation = "Not enough finite paired observations for Wilcoxon test."
        elif n_zero_differences == n_pairs:
            statistic = 0.0
            p_value = 1.0
            status = "no_difference"
            interpretation = "All paired differences are zero."
        else:
            statistic, p_value = wilcoxon(zero_values, random_values)

        rows.append(
            {
                "function": function_name,
                "dimension": dimension,
                "generator": generator,
                "metric": "best_f",
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "n_pairs": n_pairs,
                "n_zero_differences": n_zero_differences,
                "status": status,
                "interpretation": interpretation,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
