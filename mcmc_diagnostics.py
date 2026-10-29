"""Convergence and efficiency diagnostics for MCMC output (shared across problems).

Conventions: a "chains" array has shape (m, n) for one scalar quantity, with m
chains of n draws each. Use ``chains[..., j]`` to pick one coordinate from a
(m, n, d) array.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type Series = NDArray[np.float64]


def acf(x: Series) -> Series:
    """Normalised autocorrelation rho_0..rho_{n-1} of a 1-D series (FFT-based)."""
    n = x.size
    xc = x - x.mean()
    f = np.fft.rfft(xc, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n)[:n] / n
    if acov[0] == 0.0:
        return np.ones(n)
    return acov / acov[0]


def _geyer_tau(rho: Series) -> float:
    """Integrated autocorrelation time from Geyer's initial positive sequence.

    Sums pairs Gamma_m = rho_{2m} + rho_{2m+1} until the first non-positive pair.
    tau = -1 + 2 * sum(Gamma_m), floored at 1.
    """
    n_pairs = (rho.size - 1) // 2
    if n_pairs == 0:
        return 1.0
    gam = rho[0 : 2 * n_pairs : 2] + rho[1 : 2 * n_pairs + 1 : 2]
    bad = np.flatnonzero(gam <= 0.0)
    if bad.size:
        gam = gam[: bad[0]]
    return max(-1.0 + 2.0 * float(gam.sum()), 1.0)


def iat(x: Series) -> float:
    """Integrated autocorrelation time of a single series."""
    return _geyer_tau(acf(x))


def tau_multi(chains: Series) -> float:
    """Integrated autocorrelation time pooled over chains (averaged ACF)."""
    rho = np.mean([acf(c) for c in chains], axis=0)
    return _geyer_tau(rho)


def ess(chains: Series) -> float:
    """Effective sample size of an (m, n) array: m * n / tau."""
    m, n = chains.shape
    return float(m * n / tau_multi(chains))


def gelman_rubin(chains: Series) -> float:
    """Gelman-Rubin R-hat for an (m, n) array of m chains with n draws each.

    Lec 9, slides 19-22: W is the mean within-chain variance, B = n/(m-1) * sum (xbar_i - xbar)^2
    the between-chain variance, Var-hat = (n-1)/n * W + B/n, and R-hat = sqrt(Var-hat / W).
    Values near 1 mean the chains agree; 1.01 is the usual threshold.
    """
    m, n = chains.shape
    w = float(chains.var(axis=1, ddof=1).mean())
    b = n / (m - 1) * float(((chains.mean(axis=1) - chains.mean()) ** 2).sum())
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def ks_stat(a: Series, b: Series) -> float:
    """Two-sample Kolmogorov-Smirnov statistic K = sup_x |F_n(x) - G_m(x)| (Lec 9, slide 17)."""
    grid = np.concatenate([a, b])
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def ks_crit(n: int, m: int, c_alpha: float = 1.358) -> float:
    """Approximate critical value of the two-sample KS test: c(alpha) * sqrt((n + m) / (n m)).

    c(0.05) = 1.358 (Smirnov's asymptotic constant).
    """
    return c_alpha * float(np.sqrt((n + m) / (n * m)))


def mean_and_se(chains: Series) -> tuple[float, float, float]:
    """Pooled mean, Monte Carlo SE = sd / sqrt(ESS), and ESS for an (m, n) array."""
    flat = chains.ravel()
    e = ess(chains)
    return float(flat.mean()), float(flat.std(ddof=1) / np.sqrt(e)), e
