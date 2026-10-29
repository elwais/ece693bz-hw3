from pathlib import Path

import numpy as np
from numpy.typing import NDArray

N = 20000  # total iterations, including burn-in
S = 0.2  # proposal standard deviation (per coordinate)

data = np.loadtxt(
    Path(__file__).resolve().parents[1] / "data" / "orings.csv", delimiter=","
)
x, y = data[:, 0], data[:, 1]

lam = 1 / 7.3786
mu, b = -0.2322, 0.1082


def log_post(theta: NDArray[np.float64]) -> float:
    w0, w1 = float(theta[0]), float(theta[1])
    if w0 < 0:
        return -np.inf
    z: NDArray[np.float64] = w0 + w1 * x
    loglik = float(np.sum(y * z - np.logaddexp(0, z)))
    logprior = float(np.log(lam) - lam * w0 - np.log(2 * b) - abs(w1 - mu) / b)
    return logprior + loglik


rng = np.random.default_rng()
theta: NDArray[np.float64] = np.array([15.0429, -0.2322])
lp: float = log_post(theta)
chain: NDArray[np.float64] = np.empty((N, 2))

for t in range(N):
    prop: NDArray[np.float64] = theta + S * rng.standard_normal(2)
    lp_prop = log_post(prop)
    if np.log(rng.uniform()) < lp_prop - lp:
        theta, lp = prop, lp_prop
    chain[t] = theta
