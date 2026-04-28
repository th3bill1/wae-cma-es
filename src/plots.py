from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.benchmarks import BENCHMARKS
from src.experiment import run_single_experiment

PLOT_RANGES = {
    "sphere": (-5, 5, -5, 5),
    "rosenbrock": (-3, 3, -2, 6),
    "rastrigin": (-5.5, 5.5, -5.5, 5.5),
    "ackley": (-5, 5, -5, 5),
}


def create_boxplots(results_path: str, output_dir: str) -> None:
    df = pd.read_csv(results_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for function_name in sorted(df["function"].unique()):
        for dimension in sorted(df["dimension"].unique()):
            subset = df[
                (df["function"] == function_name)
                & (df["dimension"] == dimension)
            ]

            if subset.empty:
                continue

            labels = []
            data = []

            for p_sigma_mode in ["zero", "random"]:
                values = subset[subset["p_sigma_mode"] == p_sigma_mode]["best_f"]
                labels.append(p_sigma_mode)
                data.append(values)

            plt.figure()
            plt.boxplot(data, labels=labels)
            plt.yscale("log")
            plt.title(f"{function_name}, n={dimension}")
            plt.ylabel("Final best f(x)")
            plt.xlabel("p_sigma initialization")
            plt.tight_layout()

            filename = output / f"boxplot_{function_name}_n{dimension}.png"
            plt.savefig(filename, dpi=200)
            plt.close()

def create_mean_trajectory_plot(
    function_name: str,
    generator_name: str,
    seed: int,
    output_dir: str,
    max_evaluations: int = 5000,
    target_f: float = 1e-8,
) -> None:
    if function_name not in BENCHMARKS:
        raise ValueError(f"Unknown function: {function_name}")

    objective = BENCHMARKS[function_name]

    standard = run_single_experiment(
        function_name=function_name,
        dimension=2,
        p_sigma_mode="zero",
        generator_name=generator_name,
        seed=seed,
        max_evaluations=max_evaluations,
        target_f=target_f,
    )

    modified = run_single_experiment(
        function_name=function_name,
        dimension=2,
        p_sigma_mode="random",
        generator_name=generator_name,
        seed=seed,
        max_evaluations=max_evaluations,
        target_f=target_f,
    )

    x_min, x_max, y_min, y_max = PLOT_RANGES[function_name]

    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))

    standard_path = np.array(standard.mean_history)
    modified_path = np.array(modified.mean_history)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, Z, levels=30)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.plot(
        standard_path[:, 0],
        standard_path[:, 1],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="standard: p_sigma(0)=0",
    )

    plt.plot(
        modified_path[:, 0],
        modified_path[:, 1],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="modified: p_sigma(0)=random",
    )

    plt.scatter(
        standard_path[0, 0],
        standard_path[0, 1],
        marker="x",
        s=80,
        label="start standard",
    )

    plt.scatter(
        modified_path[0, 0],
        modified_path[0, 1],
        marker="x",
        s=80,
        label="start modified",
    )

    plt.scatter(
        standard_path[-1, 0],
        standard_path[-1, 1],
        marker="*",
        s=120,
        label="end standard",
    )

    plt.scatter(
        modified_path[-1, 0],
        modified_path[-1, 1],
        marker="*",
        s=120,
        label="end modified",
    )

    if function_name in ["sphere", "ackley", "rastrigin"]:
        plt.scatter(0, 0, marker="+", s=120, label="minimum globalne")

    if function_name == "rosenbrock":
        plt.scatter(1, 1, marker="+", s=120, label="minimum globalne")

    plt.title(
        f"Trajektoria środka populacji — {function_name}, 2D, seed={seed}, PRNG={generator_name}"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()

    filename = Path(output_dir) / f"trajectory_{function_name}_{generator_name}_seed{seed}.png"
    plt.savefig(filename, dpi=200)
    plt.close()