from src.analysis import create_summary, create_wilcoxon_tests
from src.plots import (
    create_boxplots,
    create_convergence_plots,
    create_ecdf_plot,
    create_mean_trajectory_plot,
)


def main() -> None:
    create_summary(
        results_path="results/raw/results.csv",
        output_path="results/summary/summary.csv",
    )

    create_wilcoxon_tests(
        results_path="results/raw/results.csv",
        output_path="results/summary/wilcoxon.csv",
    )

    create_boxplots(
        results_path="results/raw/results.csv",
        output_dir="results/plots",
    )

    create_convergence_plots(
        histories_path="results/raw/histories.json",
        output_dir="results/plots",
    )

    create_ecdf_plot(
        histories_path="results/raw/histories.json",
        output_dir="results/plots",
    )

    for function_name in ["sphere", "rosenbrock", "rastrigin", "ackley"]:
        create_mean_trajectory_plot(
            function_name=function_name,
            generator_name="pcg64",
            seed=1,
            output_dir="results/plots",
        )

    print("Reproduction finished.")


if __name__ == "__main__":
    main()