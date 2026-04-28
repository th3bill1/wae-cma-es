from pathlib import Path

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
        zero = group[group["p_sigma_mode"] == "zero"].sort_values("seed")
        random = group[group["p_sigma_mode"] == "random"].sort_values("seed")

        common_seeds = sorted(set(zero["seed"]) & set(random["seed"]))
        zero_values = zero[zero["seed"].isin(common_seeds)].sort_values("seed")["best_f"]
        random_values = random[random["seed"].isin(common_seeds)].sort_values("seed")["best_f"]

        if len(common_seeds) < 2:
            continue

        statistic, p_value = wilcoxon(zero_values, random_values)

        rows.append(
            {
                "function": function_name,
                "dimension": dimension,
                "generator": generator,
                "metric": "best_f",
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)