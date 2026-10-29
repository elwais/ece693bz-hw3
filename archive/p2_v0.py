from pathlib import Path

import numpy as np
from numpy.typing import NDArray

M = 1_000                                   # independent CFTP runs per case
DATA = Path(__file__).resolve().parents[1] / "data"


def update(A: NDArray[np.float64], lam: float, gam: float, x: NDArray[np.int_], i: int, u: float) -> None:
    """Gibbs update of node i with shared uniform u (inversion). In place."""
    n_inf = float(np.dot(A[i], x))          # number of infected neighbours
    odds = lam * gam**n_inf                 # theta * gamma^(number of infected neighbours)
    x[i] = int(u < odds / (1 + odds))


def cftp(A: NDArray[np.float64], lam: float, gam: float, rng: np.random.Generator) -> NDArray[np.int_]:
    """Monotone CFTP (gamma >= 1). Returns one exact sample."""
    n_nodes = int(A.shape[0])
    nodes = np.empty(0, dtype=int)          # randomness for times -horizon..-1, ascending in time
    us = np.empty(0)
    horizon = 1
    while True:
        k = horizon - len(nodes)            # new randomness for the earlier times only
        nodes = np.concatenate([rng.integers(n_nodes, size=k), nodes])
        us = np.concatenate([rng.random(k), us])
        lo = np.zeros(n_nodes, dtype=int)   # bottom chain
        hi = np.ones(n_nodes, dtype=int)    # top chain
        for i, u in zip(nodes, us):
            update(A, lam, gam, lo, int(i), float(u))
            update(A, lam, gam, hi, int(i), float(u))
        if np.array_equal(lo, hi):
            return lo
        horizon *= 2


def marginals(A: NDArray[np.float64], lam: float, gam: float, rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate p(x_i = 1) and its SE from M independent exact samples."""
    counts = np.zeros(A.shape[0])
    for _ in range(M):
        counts += cftp(A, lam, gam, rng)
    p = counts / M
    return p, np.sqrt(p * (1 - p) / M)


rng = np.random.default_rng()
graphs = {
    "A": np.loadtxt(DATA / "GraphA_20ER.txt", delimiter=",", dtype=float),
    "B": np.loadtxt(DATA / "GraphB_193KPP.txt", delimiter=",", dtype=float),
}
cases = [
    (0.8, 1.2, "A"),
    (0.8, 1.2, "B"),
    (0.2, 1.4, "A"),
    (0.2, 1.4, "B"),
    # (1.4, 0.6, ...) is not monotone (gamma < 1); monotone CFTP does not apply.
]
results = {(lam, gam, g): marginals(graphs[g], lam, gam, rng) for lam, gam, g in cases}
