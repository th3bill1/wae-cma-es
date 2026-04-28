from src.analysis import create_summary, create_wilcoxon_tests
from src.plots import create_boxplots


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

    print("Reproduction finished.")


if __name__ == "__main__":
    main()