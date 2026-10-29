"""O-ring logistic regression: data, prior, and log-posterior (Problem 1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mcmc_core import Theta

ROOT = Path(__file__).resolve().parents[2]  # hw3/
DATA_CSV = ROOT / "data" / "orings.csv"
RESULTS = ROOT / "results"

MLE = np.array([15.0429, -0.2322])
MLE_SE = np.array([7.3786, 0.1082])  # glmfit standard errors (oringMLE.m)
LAM = 1 / 7.3786  # Exponential rate for w0
MU, B_LAPLACE = -0.2322, 0.1082  # Laplace location and scale for w1
X_STAR = np.array([60.0, 50.0, 40.0])  # temperatures for the predictive question


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic function, written so it does not overflow for large |z|."""
    return 0.5 * (1 + np.tanh(z / 2))


@dataclass(frozen=True)
class LogisticPosterior:
    """Unnormalised log-posterior of theta = (w0, w1) for P(y=1|x) = sigmoid(w0 + w1 x)."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    lam: float = LAM
    mu: float = MU
    b: float = B_LAPLACE

    @classmethod
    def from_csv(cls, path: Path = DATA_CSV) -> LogisticPosterior:
        data = np.loadtxt(path, delimiter=",")
        return cls(x=data[:, 0], y=data[:, 1])

    def log_prior(self, theta: Theta) -> float:
        w0, w1 = float(theta[0]), float(theta[1])
        if w0 < 0:  # Exponential prior has no mass below zero
            return -np.inf
        return float(
            np.log(self.lam) - self.lam * w0 - np.log(2 * self.b) - abs(w1 - self.mu) / self.b
        )

    def log_likelihood(self, theta: Theta) -> float:
        z = float(theta[0]) + float(theta[1]) * self.x
        return float(np.sum(self.y * z - np.logaddexp(0, z)))

    def log_density(self, theta: Theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def sample_prior(self, rng: np.random.Generator) -> Theta:
        """One draw from the prior (used for over-dispersed chain starts)."""
        return np.array([rng.exponential(1 / self.lam), rng.laplace(self.mu, self.b)])

    def fisher_cov(self, theta: Theta) -> NDArray[np.float64]:
        """Inverse Fisher information at theta: a covariance shape for the proposal."""
        p = sigmoid(float(theta[0]) + float(theta[1]) * self.x)
        design = np.column_stack([np.ones_like(self.x), self.x])
        info = design.T @ (design * (p * (1 - p))[:, None])
        return np.linalg.inv(info)
