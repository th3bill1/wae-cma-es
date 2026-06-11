from dataclasses import asdict, dataclass

import numpy as np

from src.benchmarks import BENCHMARKS, STOCHASTIC_BENCHMARKS, make_gaussian_noise
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
    mean_history: list[list[float]]
    sigma_history: list[float]
    p_sigma_norm_history: list[float]


def run_single_experiment(
    function_name: str,
    dimension: int,
    p_sigma_mode: str,
    generator_name: str,
    seed: int,
    max_evaluations: int = 10_000,
    target_f: float = 1e-8,
    initial_sigma: float = 1.0,
) -> ExperimentResult:
    if function_name in STOCHASTIC_BENCHMARKS:
        noise_rng = np.random.default_rng(seed + 1_000_003)
        objective = make_gaussian_noise(noise_rng)
    else:
        objective = BENCHMARKS[function_name]

    rng = create_rng(generator_name, seed)

    config = CMAESConfig(
        dimension=dimension,
        max_evaluations=max_evaluations,
        target_f=target_f,
        p_sigma_mode=p_sigma_mode,
        initial_sigma=initial_sigma,
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
        mean_history=[point.tolist() for point in result.mean_history],
        sigma_history=result.sigma_history,
        p_sigma_norm_history=result.p_sigma_norm_history,
    )


def experiment_result_to_dict(result: ExperimentResult) -> dict:
    return asdict(result)
