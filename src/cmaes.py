from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


PSigmaMode = Literal["zero", "random"]

@dataclass
class CMAESResult:
    best_x: np.ndarray
    best_f: float
    evaluations: int
    history: list[float]
    mean_history: list[np.ndarray]


@dataclass
class CMAESConfig:
    dimension: int
    max_evaluations: int = 10_000
    target_f: float = 1e-8
    initial_sigma: float = 0.5
    p_sigma_mode: PSigmaMode = "zero"


class CMAES:
    def __init__(self, config: CMAESConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

        self.n = config.dimension
        self.lambda_ = 4 + int(3 * np.log(self.n))
        self.mu = self.lambda_ // 2

        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2.0) / (self.n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(
            0.0,
            np.sqrt((self.mu_eff - 1.0) / (self.n + 1.0)) - 1.0,
        ) + self.c_sigma

        self.c_c = (4.0 + self.mu_eff / self.n) / (
            self.n + 4.0 + 2.0 * self.mu_eff / self.n
        )
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c1,
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((self.n + 2.0) ** 2 + self.mu_eff),
        )

        self.expected_norm = np.sqrt(self.n) * (
            1.0 - 1.0 / (4.0 * self.n) + 1.0 / (21.0 * self.n ** 2)
        )

    def optimize(self, objective: Callable[[np.ndarray], float]) -> CMAESResult:
        m = self.rng.uniform(-5.0, 5.0, size=self.n)
        sigma = self.config.initial_sigma

        C = np.eye(self.n)
        B = np.eye(self.n)
        D = np.ones(self.n)
        C_inv_sqrt = np.eye(self.n)

        p_c = np.zeros(self.n)

        if self.config.p_sigma_mode == "zero":
            p_sigma = np.zeros(self.n)
        elif self.config.p_sigma_mode == "random":
            p_sigma = self.rng.normal(0.0, 1.0, size=self.n)
        else:
            raise ValueError(f"Unknown p_sigma_mode: {self.config.p_sigma_mode}")

        best_x = m.copy()
        best_f = objective(best_x)
        evaluations = 1
        history = [best_f]
        mean_history = [m.copy()]

        generation = 0

        while evaluations < self.config.max_evaluations and best_f > self.config.target_f:
            generation += 1

            z_population = self.rng.normal(size=(self.lambda_, self.n))
            y_population = z_population @ (B @ np.diag(D)).T
            x_population = m + sigma * y_population

            values = np.array([objective(x) for x in x_population])
            evaluations += self.lambda_

            order = np.argsort(values)
            x_selected = x_population[order[: self.mu]]
            y_selected = y_population[order[: self.mu]]
            values_selected = values[order[: self.mu]]

            if values_selected[0] < best_f:
                best_f = float(values_selected[0])
                best_x = x_selected[0].copy()

            old_m = m.copy()
            m = np.sum(self.weights[:, None] * x_selected, axis=0)

            mean_history.append(m.copy())

            y_w = (m - old_m) / sigma

            p_sigma = (
                (1.0 - self.c_sigma) * p_sigma
                + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff)
                * (C_inv_sqrt @ y_w)
            )

            norm_p_sigma = np.linalg.norm(p_sigma)

            h_sigma_condition = norm_p_sigma / np.sqrt(
                1.0 - (1.0 - self.c_sigma) ** (2.0 * generation)
            ) < (1.4 + 2.0 / (self.n + 1.0)) * self.expected_norm

            h_sigma = 1.0 if h_sigma_condition else 0.0

            p_c = (
                (1.0 - self.c_c) * p_c
                + h_sigma
                * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
                * y_w
            )

            rank_mu_update = np.zeros((self.n, self.n))
            for i in range(self.mu):
                y_i = y_selected[i]
                rank_mu_update += self.weights[i] * np.outer(y_i, y_i)

            C = (
                (1.0 - self.c1 - self.c_mu) * C
                + self.c1
                * (
                    np.outer(p_c, p_c)
                    + (1.0 - h_sigma)
                    * self.c_c
                    * (2.0 - self.c_c)
                    * C
                )
                + self.c_mu * rank_mu_update
            )

            sigma *= np.exp(
                (self.c_sigma / self.d_sigma)
                * (norm_p_sigma / self.expected_norm - 1.0)
            )

            C = np.triu(C) + np.triu(C, 1).T
            eigenvalues, B = np.linalg.eigh(C)

            eigenvalues = np.maximum(eigenvalues, 1e-30)
            D = np.sqrt(eigenvalues)

            C_inv_sqrt = B @ np.diag(1.0 / D) @ B.T

            history.append(best_f)

        return CMAESResult(
            best_x=best_x,
            best_f=best_f,
            evaluations=evaluations,
            history=history,
            mean_history=mean_history,
        )