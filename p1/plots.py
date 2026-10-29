"""Figures for Problem 1. PDF output (vector) so Typst can embed them directly."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from mcmc_diagnostics import acf  # noqa: E402
from p1.model import B_LAPLACE, LAM, MLE, MLE_SE, MU  # noqa: E402

PARAM_LABELS = (r"$w_0$", r"$w_1$")
TRACE_EVERY = 20  # plot every k-th draw to keep the PDF small

plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "savefig.bbox": "tight"})


def plot_trace(draws: NDArray[np.float64], burn: int, path: Path, zoom: int = 5_000) -> None:
    """Raw traces of every chain: full run (left) and the first `zoom` iterations (right).

    The burn-in and the MLE are marked on both panels.
    """
    m, n, d = draws.shape
    fig, axes = plt.subplots(d, 2, figsize=(6.3, 4.6), gridspec_kw={"width_ratios": [2, 1]})
    for col, last in enumerate((n, zoom)):
        idx = np.arange(0, last, TRACE_EVERY if col == 0 else 1)
        for j in range(d):
            ax = axes[j, col]
            for c in range(m):
                ax.plot(idx, draws[c, idx, j], lw=0.4, alpha=0.7)
            ax.axvline(burn, color="k", ls="--", lw=0.8, label=f"burn-in $B={burn}$")
            ax.axhline(MLE[j], color="r", ls=":", lw=0.9, label="MLE")
            if col == 0:
                ax.set_ylabel(PARAM_LABELS[j])
    axes[0, 0].set_title("full run (every 20th draw)", fontsize=8)
    axes[0, 1].set_title(f"first {zoom} iterations", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    fig.suptitle(f"Trace plots of {m} chains from dispersed prior starts")
    fig.savefig(path)
    plt.close(fig)


def plot_burn_in(
    win_grid: NDArray[np.int_],
    win_rhat: NDArray[np.float64],
    ks_grid: NDArray[np.int_],
    ks_stat: NDArray[np.float64],
    ks_crit_value: float,
    burn: int,
    path: Path,
) -> None:
    """Burn-in figure. Left: windowed Gelman-Rubin R-hat (chosen B). Right: KS check.

    Left: R-hat over windows of GR_WIN draws starting at b (Lec 9, slides 10, 19-22).
    Right: KS statistic of segment b against the final segment, one curve per chain
    (Lec 9, slides 16-17), with its critical value.
    """
    m, d = ks_stat.shape[0], ks_stat.shape[1]
    fig, axes = plt.subplots(d, 2, figsize=(6.3, 4.0))
    for j in range(d):
        ax = axes[j, 0]
        ax.plot(win_grid, win_rhat[:, j], lw=0.8)
        ax.axhline(1.01, color="r", ls=":", lw=0.9, label=r"$\hat R = 1.01$")
        ax.axvline(burn, color="k", ls="--", lw=0.8, label=f"$B={burn}$")
        ax.set_ylabel(PARAM_LABELS[j])
        if j == 0:
            ax.set_title("windowed Gelman-Rubin $\\hat R$", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")

        ax = axes[j, 1]
        for c in range(m):
            ax.plot(ks_grid, ks_stat[c, j], lw=0.6, label=f"chain {c + 1}")
        ax.axhline(ks_crit_value, color="r", ls=":", lw=0.9, label="KS critical value")
        ax.axvline(burn, color="k", ls="--", lw=0.8)
        if j == 0:
            ax.set_title("KS: segment $b$ vs. final segment", fontsize=8)
            ax.legend(fontsize=6, loc="upper right", ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("start $b$ (iteration)")
    fig.savefig(path)
    plt.close(fig)


def plot_posterior(kept: NDArray[np.float64], path: Path) -> None:
    """Thinned posterior draws against the prior density, with the MLE and its SE marked."""
    flat = kept.reshape(-1, kept.shape[2])
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))

    # Prior densities for overlay.
    w0_grid = np.linspace(0.0, flat[:, 0].max() * 1.05, 400)
    prior_w0 = LAM * np.exp(-LAM * w0_grid)
    w1_lo, w1_hi = np.quantile(flat[:, 1], [0.0005, 0.9995])
    w1_grid = np.linspace(min(w1_lo, MU - 0.3), max(w1_hi, MU + 0.3), 400)
    prior_w1 = np.exp(-np.abs(w1_grid - MU) / B_LAPLACE) / (2 * B_LAPLACE)
    grids = (w0_grid, w1_grid)
    priors = (prior_w0, prior_w1)

    for j, ax in enumerate(axes):
        ax.hist(flat[:, j], bins=70, density=True, alpha=0.6, label="posterior (thinned)")
        ax.plot(grids[j], priors[j], "k--", lw=0.9, label="prior")
        ax.axvline(MLE[j], color="r", lw=1.0, label="MLE")
        ax.axvspan(MLE[j] - MLE_SE[j], MLE[j] + MLE_SE[j], color="r", alpha=0.1, label="MLE ± SE")
        ax.set_xlabel(PARAM_LABELS[j])
        ax.set_ylabel("density")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def plot_acf(
    post_chain: NDArray[np.float64],
    kept_chain: NDArray[np.float64],
    path: Path,
    max_raw: int = 200,
    max_kept: int = 60,
) -> None:
    """Autocorrelation of one chain before (raw post-burn) and after thinning."""
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 3.6))
    for j in range(2):
        for col, (series, max_lag, title) in enumerate(
            [(post_chain[:, j], max_raw, "raw, post-burn-in"), (kept_chain[:, j], max_kept, "thinned")]
        ):
            rho = acf(series)[: max_lag + 1]
            ax = axes[j, col]
            ax.bar(np.arange(rho.size), rho, width=1.0, color="C0")
            band = 1.96 / np.sqrt(series.size)
            ax.axhline(band, color="r", ls=":", lw=0.8)
            ax.axhline(-band, color="r", ls=":", lw=0.8)
            ax.set_ylabel(f"ACF of {PARAM_LABELS[j]}")
            if j == 0:
                ax.set_title(title, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("lag")
    fig.savefig(path)
    plt.close(fig)
