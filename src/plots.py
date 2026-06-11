import json
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


def create_convergence_plots(histories_path: str, output_dir: str) -> None:
    with open(histories_path, "r", encoding="utf-8") as f:
        histories = json.load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    functions = sorted(set(h["function"] for h in histories))
    dimensions = sorted(set(h["dimension"] for h in histories))

    for function_name in functions:
        for dimension in dimensions:
            zero_histories = [
                h["history"] for h in histories
                if h["function"] == function_name
                and h["dimension"] == dimension
                and h["p_sigma_mode"] == "zero"
            ]
            random_histories = [
                h["history"] for h in histories
                if h["function"] == function_name
                and h["dimension"] == dimension
                and h["p_sigma_mode"] == "random"
            ]

            if not zero_histories or not random_histories:
                continue

            max_len = max(
                max(len(h) for h in zero_histories),
                max(len(h) for h in random_histories),
            )

            def pad_and_stack(hist_list, length):
                padded = []
                for h in hist_list:
                    padded.append(h + [h[-1]] * (length - len(h)))
                return np.array(padded)

            zero_arr = pad_and_stack(zero_histories, max_len)
            random_arr = pad_and_stack(random_histories, max_len)

            zero_median = np.median(zero_arr, axis=0)
            random_median = np.median(random_arr, axis=0)

            generations = np.arange(max_len)

            plt.figure(figsize=(8, 5))
            plt.plot(generations, zero_median, label="standard: p_sigma(0)=0", linewidth=1.5)
            plt.plot(generations, random_median, label="modified: p_sigma(0)=random", linewidth=1.5)
            plt.yscale("log")
            plt.xlabel("Generacja")
            plt.ylabel("Median best f(x)")
            plt.title(f"Zbieżność — {function_name}, n={dimension}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = output / f"convergence_{function_name}_n{dimension}.png"
            plt.savefig(filename, dpi=200)
            plt.close()


def _lambda_for_dim(dimension: int) -> int:
    return 4 + int(3 * np.log(dimension))


def _eval_counts_for_history(history_length: int, dimension: int) -> np.ndarray:
    lambda_ = _lambda_for_dim(dimension)
    return 1 + np.arange(history_length) * lambda_


def create_ecdf_plot(
    histories_path: str,
    output_dir: str,
    targets: np.ndarray | None = None,
    n_xpoints: int = 200,
) -> None:
    targets_arr: np.ndarray = (
        np.logspace(2, -8, 51) if targets is None else targets
    )

    with open(histories_path, "r", encoding="utf-8") as f:
        histories = json.load(f)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    functions = sorted(set(h["function"] for h in histories))
    dimensions = sorted(set(h["dimension"] for h in histories))

    for function_name in functions:
        for dimension in dimensions:
            zero_runs = [
                np.array(h["history"]) for h in histories
                if h["function"] == function_name
                and h["dimension"] == dimension
                and h["p_sigma_mode"] == "zero"
            ]
            random_runs = [
                np.array(h["history"]) for h in histories
                if h["function"] == function_name
                and h["dimension"] == dimension
                and h["p_sigma_mode"] == "random"
            ]

            if not zero_runs or not random_runs:
                continue

            max_evals = max(
                max(_eval_counts_for_history(len(h), dimension)[-1] for h in zero_runs),
                max(_eval_counts_for_history(len(h), dimension)[-1] for h in random_runs),
            )

            x_grid = np.unique(
                np.round(np.logspace(0, np.log10(max_evals), n_xpoints)).astype(int)
            )

            def ecdf_curve(runs: list[np.ndarray]) -> np.ndarray:
                total_pairs = len(runs) * len(targets_arr)
                solved = np.zeros_like(x_grid, dtype=float)
                for run in runs:
                    eval_counts = _eval_counts_for_history(len(run), dimension)
                    for target in targets_arr:
                        reached_mask = run <= target
                        if not reached_mask.any():
                            continue
                        first_idx = int(np.argmax(reached_mask))
                        reached_at = eval_counts[first_idx]
                        solved += (x_grid >= reached_at).astype(float)
                return solved / total_pairs

            zero_ecdf = ecdf_curve(zero_runs)
            random_ecdf = ecdf_curve(random_runs)

            plt.figure(figsize=(8, 5))
            plt.step(x_grid, zero_ecdf, where="post", label="standard: p_sigma(0)=0", linewidth=1.5)
            plt.step(x_grid, random_ecdf, where="post", label="modified: p_sigma(0)=random", linewidth=1.5)
            plt.xscale("log")
            plt.xlabel("Liczba ewaluacji funkcji celu")
            plt.ylabel("Udział par (run × target) rozwiązanych")
            plt.title(
                f"ECDF — {function_name}, n={dimension} "
                f"({len(targets_arr)} targetów: {targets_arr[0]:.0e}..{targets_arr[-1]:.0e})"
            )
            plt.ylim(-0.02, 1.02)
            plt.legend()
            plt.grid(True, alpha=0.3, which="both")
            plt.tight_layout()

            filename = output / f"ecdf_{function_name}_n{dimension}.png"
            plt.savefig(filename, dpi=200)
            plt.close()


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

    standard_path = np.array(standard.mean_history)
    modified_path = np.array(modified.mean_history)

    all_points = np.vstack([standard_path, modified_path])
    margin = 0.5
    data_x_min, data_x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
    data_y_min, data_y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin

    # include global optimum in view
    if function_name in ["sphere", "ackley", "rastrigin"]:
        opt_x, opt_y = 0.0, 0.0
    else:
        opt_x, opt_y = 1.0, 1.0
    data_x_min = min(data_x_min, opt_x - margin)
    data_x_max = max(data_x_max, opt_x + margin)
    data_y_min = min(data_y_min, opt_y - margin)
    data_y_max = max(data_y_max, opt_y + margin)

    x_min, x_max = data_x_min, data_x_max
    y_min, y_max = data_y_min, data_y_max

    xs = np.linspace(x_min, x_max, 300)
    ys = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))

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
        zorder=2,
    )

    plt.plot(
        modified_path[:, 0],
        modified_path[:, 1],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="modified: p_sigma(0)=random",
        zorder=2,
    )

    plt.scatter(
        standard_path[0, 0],
        standard_path[0, 1],
        marker="x",
        s=100,
        label="start standard",
        zorder=3,
    )

    plt.scatter(
        modified_path[0, 0],
        modified_path[0, 1],
        marker="x",
        s=100,
        label="start modified",
        zorder=3,
    )

    plt.scatter(
        standard_path[-1, 0],
        standard_path[-1, 1],
        marker="*",
        s=150,
        label="end standard",
        zorder=4,
    )

    plt.scatter(
        modified_path[-1, 0],
        modified_path[-1, 1],
        marker="*",
        s=150,
        label="end modified",
        zorder=4,
    )

    if function_name in ["sphere", "ackley", "rastrigin"]:
        plt.scatter(0, 0, marker="+", s=120, label="minimum globalne", zorder=5)

    if function_name == "rosenbrock":
        plt.scatter(1, 1, marker="+", s=120, label="minimum globalne", zorder=5)

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