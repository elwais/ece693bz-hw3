"""Coupling from the past for the infection model (random-scan Gibbs, bounding pair)."""

from __future__ import annotations

import numpy as np

from p2.model import Adj, State, prob_infected


def bounded_update(adj: Adj, lam: float, gam: float, lo: State, hi: State, i: int, u: float) -> None:
    """One coupled update of the pair (lo, hi), in place, using the same node i and uniform u.

    gam >= 1: P(x_i = 1 | neighbours) increases with the neighbours, so the lower chain
    looks at lo and the upper chain at hi (monotone case).
    gam < 1: it decreases with the neighbours, so the roles swap: the lower chain looks at hi
    and the upper chain at lo.  Either way lo <= x <= hi is preserved for every x in between.
    """
    n_lo = float(np.dot(adj[i], lo))
    n_hi = float(np.dot(adj[i], hi))
    if gam < 1:
        n_lo, n_hi = n_hi, n_lo
    new_lo = int(u < prob_infected(n_lo, lam, gam))
    new_hi = int(u < prob_infected(n_hi, lam, gam))
    lo[i] = new_lo
    hi[i] = new_hi


def cftp(adj: Adj, lam: float, gam: float, rng: np.random.Generator) -> tuple[State, int]:
    """Return one exact sample and the horizon T (steps into the past) that was needed."""
    n_nodes = adj.shape[0]
    nodes = np.empty(0, dtype=int)  # random nodes for times -T, ..., -1 (in time order)
    us = np.empty(0)  # uniforms for the same times
    horizon = 1
    while True:
        # Keep the old randomness for the recent times; draw new randomness only for earlier times.
        k = horizon - len(nodes)
        nodes = np.concatenate([rng.integers(n_nodes, size=k), nodes])
        us = np.concatenate([rng.random(k), us])
        lo = np.zeros(n_nodes, dtype=int)  # lower bound of the sandwich: all healthy
        hi = np.ones(n_nodes, dtype=int)  # upper bound: all infected
        for i, u in zip(nodes, us):
            bounded_update(adj, lam, gam, lo, hi, int(i), float(u))
        if np.array_equal(lo, hi):
            return lo, horizon
        horizon *= 2
