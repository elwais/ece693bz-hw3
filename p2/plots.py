"""Figure for Problem 2: estimated p(x_i = 1) against node degree, one panel per case."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_marginals(cases: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(8, 9), sharey="row")
    for ax, c in zip(axes.ravel(), cases):
        deg = np.asarray(c["degree"])
        p = np.asarray(c["p"])
        se = np.asarray(c["se"])
        ax.errorbar(deg + 0.08 * np.random.default_rng(0).standard_normal(deg.size), p, yerr=2 * se, fmt=".", ms=4, lw=0.5)
        ex = c.get("exact")
        if ex is not None:
            ax.plot(deg, np.asarray(ex), "r_", ms=8, label="exact")
            ax.legend()
        ax.set_title(f"graph {c['graph']}: $\\lambda$={c['lam']}, $\\gamma$={c['gam']}")
        ax.set_xlabel("degree")
        ax.set_ylabel("$\\hat p(x_i=1)$")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
