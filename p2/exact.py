"""Exact marginals p(x_i = 1) by enumerating all 2^N states (feasible for graph A only, N = 20)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from p2.model import Adj


def exact_marginals(adj: Adj, lam: float, gam: float) -> NDArray[np.float64]:
    n = adj.shape[0]
    codes = np.arange(2**n)
    x = ((codes[:, None] >> np.arange(n)) & 1).astype(np.float32)  # (2^N, N): every state
    a = adj.astype(np.float32)
    # phi(x) = (number infected) log(lam) + (x^T A x)/2 log(gam)
    phi = x.sum(1) * np.log(lam) + 0.5 * np.einsum("sn,sn->s", x @ a, x) * np.log(gam)
    w = np.exp(phi.astype(np.float64) - phi.max())
    return (w @ x) / w.sum()
