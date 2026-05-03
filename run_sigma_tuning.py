from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.experiment import run_single_experiment


FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "ackley"]
DIMENSIONS = [2, 10]
P_SIGMA_MODES = ["zero", "random"]
SIGMAS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
GENERATOR = "pcg64"
SEEDS = list(range(1, 11))

MAX_EVALUATIONS = 20_000
TARGET_F = 1e-8


def main() -> None:
    raw_dir = Path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    configurations = [
        (function_name, dimension, p_sigma_mode, sigma, seed)
        for function_name in FUNCTIONS
        for dimension in DIMENSIONS
        for p_sigma_mode in P_SIGMA_MODES
        for sigma in SIGMAS
        for seed in SEEDS
    ]

    print(f"Running {len(configurations)} experiments...")

    for function_name, dimension, p_sigma_mode, sigma, seed in tqdm(configurations):
        result = run_single_experiment(
            function_name=function_name,
            dimension=dimension,
            p_sigma_mode=p_sigma_mode,
            generator_name=GENERATOR,
            seed=seed,
            max_evaluations=MAX_EVALUATIONS,
            target_f=TARGET_F,
            initial_sigma=sigma,
        )

        rows.append(
            {
                "function": result.function_name,
                "dimension": result.dimension,
                "p_sigma_mode": result.p_sigma_mode,
                "initial_sigma": sigma,
                "seed": result.seed,
                "best_f": result.best_f,
                "evaluations": result.evaluations,
                "reached_target": result.reached_target,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(raw_dir / "sigma_tuning.csv", index=False)

    print(f"\nSaved {len(rows)} results to results/raw/sigma_tuning.csv")

    summary = (
        df.groupby(["function", "dimension", "p_sigma_mode", "initial_sigma"])
        .agg(
            median_best_f=("best_f", "median"),
            mean_best_f=("best_f", "mean"),
            median_evals=("evaluations", "median"),
            success_rate=("reached_target", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(raw_dir / "sigma_tuning_summary.csv", index=False)
    print("Saved summary to results/raw/sigma_tuning_summary.csv")


if __name__ == "__main__":
    main()
