"""Problem 2 driver: CFTP estimates of p(x_i = 1) for the six cases.

Run from hw3/code:  uv run python -m p2.run
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from p2.cftp import cftp
from p2.exact import exact_marginals
from p2.model import RESULTS, Adj, load_graph
from p2.plots import plot_marginals

SEED = 693
M = 5_000  # independent exact samples per case
CASES = [  # (lambda, gamma, graph), in the order of the assignment
    (0.8, 1.2, "A"),
    (0.8, 1.2, "B"),
    (0.2, 1.4, "A"),
    (0.2, 1.4, "B"),
    (1.4, 0.6, "A"),
    (1.4, 0.6, "B"),
]
LOG_PATH = RESULTS / "p2_run.log"
FIG_PATH = RESULTS / "figures" / "p2" / "marginals.pdf"

log = logging.getLogger("p2")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_PATH, mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.propagate = False


def estimate(adj: Adj, lam: float, gam: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """p_i = fraction of M exact samples with x_i = 1; SE_i = sqrt(p_i (1 - p_i) / M)."""
    counts = np.zeros(adj.shape[0])
    horizons: list[int] = []
    for _ in range(M):
        x, horizon = cftp(adj, lam, gam, rng)
        counts += x
        horizons.append(horizon)
    p = counts / M
    return p, np.sqrt(p * (1 - p) / M), horizons


def main() -> None:
    setup_logging()
    rng = np.random.default_rng(SEED)
    log.info("config: seed=%d M=%d samples per case", SEED, M)
    graphs = {g: load_graph(g) for g in "AB"}
    out: list[dict[str, object]] = []

    for lam, gam, g in CASES:
        adj = graphs[g]
        t0 = time.time()
        p, se, horizons = estimate(adj, lam, gam, rng)
        row: dict[str, object] = {
            "lam": lam, "gam": gam, "graph": g, "p": p.tolist(), "se": se.tolist(),
            "degree": adj.sum(1).astype(int).tolist(),
            "horizon_mean": float(np.mean(horizons)), "horizon_max": int(np.max(horizons)),
            "seconds": time.time() - t0,
        }
        log.info(
            "lam=%.1f gam=%.1f graph %s (N=%d): mean p=%.3f range [%.3f, %.3f] max SE=%.4f | "
            "horizon mean %.0f max %d | %.1f s",
            lam, gam, g, adj.shape[0], p.mean(), p.min(), p.max(), se.max(),
            row["horizon_mean"], row["horizon_max"], row["seconds"],
        )
        if g == "A":  # only graph A is small enough to enumerate all 2^20 states
            ex = exact_marginals(adj, lam, gam)
            z = (p - ex) / np.maximum(se, 1e-12)
            row["exact"] = ex.tolist()
            row["max_abs_err"] = float(np.max(np.abs(p - ex)))
            row["max_abs_z"] = float(np.max(np.abs(z)))
            log.info("  vs exact enumeration: max |p - exact| = %.4f, max |z| = %.2f, mean z^2 = %.2f (expect ~1)",
                     row["max_abs_err"], row["max_abs_z"], float(np.mean(z**2)))
        out.append(row)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p2_results.json").write_text(json.dumps(out, indent=1))
    plot_marginals(out, FIG_PATH)
    log.info("wrote %s and %s", RESULTS / "p2_results.json", FIG_PATH)


if __name__ == "__main__":
    main()
