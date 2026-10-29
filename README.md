# ECE 693B Homework 3: Monte Carlo Methods

Code for Homework 3 (Fall 2026).

- **Problem 1:** Bayesian logistic regression of the O-ring data with Metropolis–Hastings. Burn-in is chosen with windowed Gelman–Rubin and Kolmogorov–Smirnov monitors, and thinning with the integrated autocorrelation time. The code estimates posterior predictive failure probabilities and compares them with a Rao–Blackwellized estimator.
- **Problem 2:** Perfect sampling (coupling from the past) of an infection model on a graph. A bounding-chain update handles $\gamma < 1$. On the 20-node graph the estimates are validated against exact enumeration.

## Layout

| Path | Contents |
|---|---|
| `mcmc_core.py` | Metropolis–Hastings: `LogTarget` protocol, `Kernel` / `GaussianRW` proposals, `Chain`, `MetropolisHastings` |
| `mcmc_diagnostics.py` | ACF, integrated autocorrelation time, ESS, Gelman–Rubin $\hat R$, two-sample KS statistic |
| `p1/` | Problem 1: `model.py` (posterior), `run.py` (driver), `plots.py` |
| `p2/` | Problem 2: `model.py` (Gibbs update), `cftp.py` (CFTP), `exact.py` (enumeration), `run.py`, `plots.py` |
| `archive/` | Early single-file prototypes, kept for reference |

## Running

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 or later.

The drivers expect the homework directory layout: inputs come from `../data/` and outputs go to `../results/`.

```sh
uv sync
uv run p1   # or: uv run python -m p1.run   (about 1 minute)
uv run p2   # or: uv run python -m p2.run   (about 3–4 minutes)
```

Each run writes a log, a JSON summary and its figures to `../results/`.

### Data files

The following files go in `../data/`:

- `orings.csv`: 23 rows of (temperature °F, failure 0/1)
- `GraphA_20ER.txt`: 20×20 adjacency matrix
- `GraphB_193KPP.txt`: 193×193 adjacency matrix

The adjacency matrices are comma-separated 0/1 values.

Self-contained Colab notebooks with the data embedded are distributed separately.
