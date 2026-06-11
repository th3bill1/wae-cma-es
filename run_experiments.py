import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.experiment import run_single_experiment


FUNCTIONS = ["sphere", "rosenbrock", "rastrigin", "ackley", "gaussian_noise"]
DIMENSIONS = [2, 10, 30]
P_SIGMA_MODES = ["zero", "random"]
GENERATORS = ["pcg64", "mt19937"]
SEEDS = list(range(1, 31))

MAX_EVALUATIONS = 20_000
TARGET_F = 1e-8


def main() -> None:
    raw_dir = Path("results/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories = []

    configurations = [
        (function_name, dimension, p_sigma_mode, generator_name, seed)
        for function_name in FUNCTIONS
        for dimension in DIMENSIONS
        for p_sigma_mode in P_SIGMA_MODES
        for generator_name in GENERATORS
        for seed in SEEDS
    ]

    for function_name, dimension, p_sigma_mode, generator_name, seed in tqdm(configurations):
        result = run_single_experiment(
            function_name=function_name,
            dimension=dimension,
            p_sigma_mode=p_sigma_mode,
            generator_name=generator_name,
            seed=seed,
            max_evaluations=MAX_EVALUATIONS,
            target_f=TARGET_F,
        )

        rows.append(
            {
                "function": result.function_name,
                "dimension": result.dimension,
                "p_sigma_mode": result.p_sigma_mode,
                "generator": result.generator_name,
                "seed": result.seed,
                "best_f": result.best_f,
                "evaluations": result.evaluations,
                "reached_target": result.reached_target,
            }
        )

        histories.append(
            {
                "function": result.function_name,
                "dimension": result.dimension,
                "p_sigma_mode": result.p_sigma_mode,
                "generator": result.generator_name,
                "seed": result.seed,
                "history": result.history,
                "sigma_history": result.sigma_history,
                "p_sigma_norm_history": result.p_sigma_norm_history,
            }
        )

    pd.DataFrame(rows).to_csv(raw_dir / "results.csv", index=False)

    with open(raw_dir / "histories.json", "w", encoding="utf-8") as file:
        json.dump(histories, file)

    print("Saved results to results/raw/results.csv")
    print("Saved histories to results/raw/histories.json")


if __name__ == "__main__":
    main()
