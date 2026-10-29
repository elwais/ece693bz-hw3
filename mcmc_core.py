"""Generic Metropolis-Hastings stuff for use in problems 1 and 2"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type Theta = NDArray[np.float64]


class LogTarget(Protocol):
    """Unnormalised log-density. Returns -inf outside the support."""

    def log_density(self, theta: Theta) -> float: ...


class Kernel(ABC):
    """Proposal distribution q(prop | theta)."""

    @abstractmethod
    def propose(self, theta: Theta, rng: np.random.Generator) -> Theta: ...

    def log_proposal_ratio(self, theta: Theta, prop: Theta) -> float:
        """log q(theta | prop) - log q(prop | theta). Zero for symmetric proposals."""
        del theta, prop
        return 0.0


@dataclass(frozen=True)
class GaussianRW(Kernel):
    """Symmetric Gaussian random walk; proposal covariance is chol @ chol.T."""

    chol: NDArray[np.float64]

    @classmethod
    def from_cov(cls, cov: NDArray[np.float64]) -> GaussianRW:
        return cls(np.linalg.cholesky(cov))

    def scaled(self, factor: float) -> GaussianRW:
        """Multiply the proposal standard deviation by ``factor``."""
        return replace(self, chol=self.chol * factor)

    def propose(self, theta: Theta, rng: np.random.Generator) -> Theta:
        return theta + self.chol @ rng.standard_normal(theta.shape[0])


@dataclass(frozen=True)
class Chain:
    """Raw output of one MH run with every state (includes burn-in)"""

    draws: NDArray[np.float64]  # (n, d)
    log_p: NDArray[np.float64]  # (n,) unnormalised log-density of each state
    n_accept: int

    @property
    def accept_rate(self) -> float:
        return self.n_accept / self.draws.shape[0]


class MetropolisHastings:
    def __init__(self, target: LogTarget, kernel: Kernel) -> None:
        self.target = target
        self.kernel = kernel

    def step(
        self, theta: Theta, log_p: float, rng: np.random.Generator
    ) -> tuple[Theta, float, bool]:
        """One MH update. Returns (new state, its log-density, accepted?)."""
        prop = self.kernel.propose(theta, rng)
        log_p_prop = self.target.log_density(prop)
        # log alpha = log pi(prop) - log pi(theta) + log q(theta|prop) - log q(prop|theta)
        log_alpha = log_p_prop - log_p + self.kernel.log_proposal_ratio(theta, prop)
        if np.log(rng.uniform()) < log_alpha:
            return prop, log_p_prop, True
        return theta, log_p, False

    def run(self, theta0: Theta, n: int, rng: np.random.Generator) -> Chain:
        log_p = self.target.log_density(theta0)
        if not np.isfinite(log_p):
            raise ValueError("MH start must have finite log-density")
        draws = np.empty((n, theta0.shape[0]))
        log_ps = np.empty(n)
        n_accept = 0
        theta = theta0
        for t in range(n):
            theta, log_p, accepted = self.step(theta, log_p, rng)
            draws[t] = theta
            log_ps[t] = log_p
            n_accept += accepted
        return Chain(draws, log_ps, n_accept)
