from dataclasses import asdict, dataclass

from src.benchmarks import BENCHMARKS
from src.cmaes import CMAES, CMAESConfig
from src.rng import create_rng


@dataclass
class ExperimentResult:
    function_name: str
    dimension: int
    p_sigma_mode: str
    generator_name: str
    seed: int
    best_f: float
    evaluations: int
    reached_target: bool
    history: list[float]


def run_single_experiment(
    function_name: str,
    dimension: int,
    p_sigma_mode: str,
    generator_name: str,
    seed: int,
    max_evaluations: int = 10_000,
    target_f: float = 1e-8,
) -> ExperimentResult:
    objective = BENCHMARKS[function_name]
    rng = create_rng(generator_name, seed)

    config = CMAESConfig(
        dimension=dimension,
        max_evaluations=max_evaluations,
        target_f=target_f,
        p_sigma_mode=p_sigma_mode,
    )

    optimizer = CMAES(config=config, rng=rng)
    result = optimizer.optimize(objective)

    return ExperimentResult(
        function_name=function_name,
        dimension=dimension,
        p_sigma_mode=p_sigma_mode,
        generator_name=generator_name,
        seed=seed,
        best_f=result.best_f,
        evaluations=result.evaluations,
        reached_target=result.best_f <= target_f,
        history=result.history,
    )


def experiment_result_to_dict(result: ExperimentResult) -> dict:
    return asdict(result)