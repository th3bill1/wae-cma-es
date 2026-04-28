import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))

    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20.0
        + np.e
    )


BENCHMARKS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "ackley": ackley,
}