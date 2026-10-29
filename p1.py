from pathlib import Path

import numpy as np
from numpy.typing import NDArray

B = 5_000  # burn-in iterations
THIN = 50  # keep every THIN-th draw
M = 10_000  # number of kept draws
S = np.array([3.0, 0.1])  # proposal sd for (w0, w1)

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
    # Why do w0 + w1 * x? I thought the random walk was centered at a sample of the posterior?
    # Oh wait, this is the laplacian or exponential perhaps
    z: NDArray[np.float64] = w0 + w1 * x

    # Why logaddexp?
    loglik = float(np.sum(y * z - np.logaddexp(0, z)))

    # I presume this is a direct plug-and-chug from the prior
    logprior = float(np.log(lam) - lam * w0 - np.log(2 * b) - abs(w1 - mu) / b)

    # Prior and likelihood is... just log theta?
    # p(theta) * p (y | x, theta)?
    return logprior + loglik


rng = np.random.default_rng()

# ML initializer? Prior? Sample from prior?
theta: NDArray[np.float64] = np.array([15.0429, -0.2322])

lp: float = log_post(theta)

# kept? what is kept?
kept: NDArray[np.float64] = np.empty((M, 2))
n_kept = 0

# The MH updates I presume
for it in range(B + THIN * M):
    # Okay this is the random walk part
    prop: NDArray[np.float64] = theta + S * rng.standard_normal(2)
    lp_prop = log_post(prop)

    # Acceptance ratio check
    if np.log(rng.uniform()) < lp_prop - lp:
        theta, lp = prop, lp_prop
    if it >= B and (it - B) % THIN == THIN - 1:
        kept[n_kept] = theta
        n_kept += 1

    # The loop mutates theta, lp,
    # so each theta is a (correlated) sample that (when stationary) is a
    # generator of the posterior?


# These were provided by the assignment
x_star = np.array([60.0, 50.0, 40.0])

# What is this? Why the different uses of z?
z = kept[:, :1] + kept[:, 1:] * x_star  # (M, 3): w0 + w1 x*

# What is sigma again? One of the laplacian or expoential? Is this the logistic model?
prob = 0.5 * (1 + np.tanh(z / 2))  # sigma(z), numerically stable

# Okay, we take different values of x_star to get different probabilities
# Then the means are Bayesian estimates on the posterior?
p_hat = prob.mean(axis=0)  # predictive failure prob

# This is just the usual stats?
se_hat = prob.std(axis=0, ddof=1) / np.sqrt(M)  # SE assuming independent draws
