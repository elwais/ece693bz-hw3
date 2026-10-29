"""Infection model on a graph: data, constants, and the single-site Gibbs update."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]  # hw3/
DATA = ROOT / "data"
RESULTS = ROOT / "results"

type Adj = NDArray[np.float64]  # symmetric 0/1 adjacency matrix, zero diagonal
type State = NDArray[np.int_]  # x in {0,1}^N, 1 = infected


def load_graph(name: str) -> Adj:
    """name is 'A' (20 nodes) or 'B' (193 nodes)."""
    files = {"A": "GraphA_20ER.txt", "B": "GraphB_193KPP.txt"}
    return np.loadtxt(DATA / files[name], delimiter=",", dtype=float)


def prob_infected(n_inf: float, lam: float, gam: float) -> float:
    """P(x_i = 1 | neighbours) = odds / (1 + odds), odds = lam * gam^(number of infected neighbours)."""
    odds = lam * gam**n_inf
    return odds / (1 + odds)


def update(adj: Adj, lam: float, gam: float, x: State, i: int, u: float) -> None:
    """Gibbs update of node i, in place, using the uniform u (so the same u gives coupled chains)."""
    n_inf = float(np.dot(adj[i], x))
    x[i] = int(u < prob_infected(n_inf, lam, gam))
