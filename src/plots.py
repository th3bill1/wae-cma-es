from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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