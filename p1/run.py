"""Problem 1 driver: MCMC for Bayesian logistic regression of O-ring failures.

Run from anywhere, after `uv sync` in hw3/code:   uv run p1
Writes results/p1_draws.npz, results/p1_summary.json, results/p1_run.log and
results/figures/p1/*.pdf.

Burn-in and monitoring follow Lec 9 (Monitoring MCMC): trace plots (slides 5-7),
the two-sample Kolmogorov-Smirnov test between segments (slides 16-17), and the
Gelman-Rubin R-hat over chains (slides 19-22).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mcmc_core import GaussianRW, MetropolisHastings, Theta
from mcmc_diagnostics import ess, gelman_rubin, ks_crit, ks_stat, mean_and_se, tau_multi
from p1.model import DATA_CSV, MLE, MLE_SE, RESULTS, X_STAR, LogisticPosterior, sigmoid
from p1.plots import plot_acf, plot_burn_in, plot_posterior, plot_trace

SEED = 693
N_CHAINS = 4
N_KEEP = 10_000  # independent draws required: kept draws have ESS ~ N_KEEP
N_RAW = 200_000  # raw iterations per chain (burn-in and thinning come out of this)
N_PILOT = 2_000  # tuning iterations per chain per round (discarded)
N_TUNE_ROUNDS = 8
ACCEPT_TARGET = 0.30
ACCEPT_BAND = (0.20, 0.40)
KS_SEG = 20_000  # raw draws per segment in the post-burn-in KS check (Lec 9, slides 16-17)
KS_LAG = 20  # samples taken every KS_LAG draws so that they are roughly IID
GR_WIN = 1_000  # window length for windowed Gelman-Rubin burn-in (Lec 9, slide 10); R-hat is noisy below this
GR_STEP = 250  # grid of window starts for the burn-in search
KS_STEP = 250  # grid of segment starts for the KS check
RHAT_MAX = 1.01  # Gelman-Rubin threshold on the post-burn-in draws (Lec 9, slides 19-22)
TAU_KEPT_MAX = 1.02  # thinning: smallest THIN with tau_int of kept draws below this
N_REP = 2_000  # replications of the simulated-outcome estimator in part (d)
PARAMS = ("w0", "w1")
FIG_DIR = RESULTS / "figures" / "p1"
LOG_PATH = RESULTS / "p1_run.log"

log = logging.getLogger("p1")


def setup_logging(path: Path) -> None:
    """Log to the console and to `path` (overwritten on each run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(path, mode="w")):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    log.propagate = False


def tune_kernel(
    model: LogisticPosterior,
    base: GaussianRW,
    starts: list[Theta],
    rng: np.random.Generator,
) -> GaussianRW:
    """Scale the proposal so the pilot acceptance rate lies in ACCEPT_BAND.

    Tuning uses separate pilot runs whose draws are discarded. The kernel is
    frozen afterwards, so the production chains are a valid MH chain.
    """
    kernel = base
    for rnd in range(N_TUNE_ROUNDS):
        rates = [MetropolisHastings(model, kernel).run(s, N_PILOT, rng).accept_rate for s in starts]
        acc = float(np.mean(rates))
        log.info("tuning round %d: sd=%s accept=%.3f", rnd, np.diag(kernel.chol).round(4), acc)
        if ACCEPT_BAND[0] <= acc <= ACCEPT_BAND[1]:
            break
        kernel = kernel.scaled(float(np.exp(2.0 * (acc - ACCEPT_TARGET))))
    return kernel


def burn_in_windows(draws: NDArray[np.float64]) -> tuple[int, NDArray[np.int_], NDArray[np.float64]]:
    """Burn-in from Gelman-Rubin over windows along the chain (Lec 9, slides 10 and 19-22).

    Slide 10 suggests looking at different windows along the chain. For each window
    [b, b + GR_WIN) we compute the Gelman-Rubin R-hat across the M chains, separately
    for each parameter. B is the first grid point where every parameter has R-hat
    at or below RHAT_MAX.
    """
    m, n, d = draws.shape
    grid = np.arange(0, n - GR_WIN + 1, GR_STEP)
    rhat = np.array(
        [[gelman_rubin(draws[:, b : b + GR_WIN, j]) for j in range(d)] for b in grid]
    )  # (grid, d)
    ok = np.flatnonzero((rhat <= RHAT_MAX).all(axis=1))
    if ok.size == 0:
        raise RuntimeError("windowed R-hat never reached the threshold; increase N_RAW")
    return int(grid[ok[0]]), grid, rhat


def ks_stats(draws: NDArray[np.float64]) -> tuple[NDArray[np.int_], NDArray[np.float64], float]:
    """Two-sample KS statistic of segment b against the final segment (Lec 9, slides 16-17).

    Each segment has KS_SEG raw draws, thinned to lag KS_LAG. The grid stops where the
    two segments would overlap. Used as a post-burn-in check, not to choose B.
    """
    m, n, d = draws.shape
    grid = np.arange(0, n - 2 * KS_SEG + 1, KS_STEP)
    n_seg = KS_SEG // KS_LAG
    crit = ks_crit(n_seg, n_seg)
    stat = np.empty((m, d, grid.size))
    for c in range(m):
        for j in range(d):
            last = draws[c, n - KS_SEG : n : KS_LAG, j]
            for k, b in enumerate(grid):
                stat[c, j, k] = ks_stat(draws[c, b : b + KS_SEG : KS_LAG, j], last)
    return grid, stat, crit


def compare_predictive_estimators(
    prob: NDArray[np.float64], rng: np.random.Generator
) -> list[dict[str, float]]:
    """Part (d): simulated-outcome estimator vs. its Rao-Blackwellization.

    prob has shape (m, n, k): sigma(z_*) for each kept draw and each temperature.
    Naive: draw y_m ~ Bernoulli(sigma(z_m)) for each kept draw and average the 0/1
    outcomes. Rao-Blackwellized: average sigma(z_m) itself, i.e. replace y_m by its
    conditional mean given theta_m. Both are unbiased for the same posterior
    predictive probability.

    By the law of total variance, Var(naive) = Var(RB) + E[sigma (1 - sigma)] / M. The
    second term is the Bernoulli noise that RB removes. It is measured here by
    repeating the simulation N_REP times on the same kept draws.
    """
    out = []
    for k in range(prob.shape[2]):
        pk = prob[:, :, k]
        flat = pk.ravel()
        m_draws = flat.size
        p_rb, se_rb, _ = mean_and_se(pk)
        reps = (rng.random((N_REP, m_draws)) < flat).mean(axis=1)  # (N_REP,)
        sd_sim = float(reps.std(ddof=1))  # Bernoulli noise only, theta draws held fixed
        theory_sd = float(np.sqrt(np.mean(flat * (1.0 - flat)) / m_draws))
        se_naive = float(np.hypot(se_rb, sd_sim))  # posterior MC error plus Bernoulli noise
        out.append(
            dict(
                p_rb=p_rb,
                se_rb=se_rb,
                p_naive_mean=float(reps.mean()),
                sd_bernoulli=sd_sim,
                sd_bernoulli_theory=theory_sd,
                se_naive=se_naive,
                se_ratio=se_naive / se_rb,
                var_ratio=(se_naive / se_rb) ** 2,
            )
        )
    return out


def choose_thin(post: NDArray[np.float64]) -> int:
    """Smallest THIN whose kept draws have tau_int <= TAU_KEPT_MAX for every parameter."""
    for k in range(1, 200):
        kept = post[:, k - 1 :: k]
        if max(tau_multi(kept[:, :, j]) for j in range(post.shape[2])) <= TAU_KEPT_MAX:
            return k
    raise RuntimeError("thinning never reached the threshold; increase N_RAW")


def main() -> None:
    setup_logging(LOG_PATH)
    rng = np.random.default_rng(SEED)
    model = LogisticPosterior.from_csv(DATA_CSV)
    summary: dict[str, object] = {}

    log.info(
        "config: seed=%d chains=%d N_RAW=%d N_PILOT=%d N_KEEP=%d GR_WIN=%d rhat_max=%.2f "
        "KS_SEG=%d KS_LAG=%d tau_kept_max=%.2f",
        SEED, N_CHAINS, N_RAW, N_PILOT, N_KEEP, GR_WIN, RHAT_MAX, KS_SEG, KS_LAG, TAU_KEPT_MAX,
    )
    log.info("data: %d launches from %s (assignment text says 24)", model.x.size, DATA_CSV.name)
    cov = model.fisher_cov(MLE)
    log.info("covariance-based SE at MLE: %s (glmfit: %s)", np.sqrt(np.diag(cov)).round(4), MLE_SE)

    # Over-dispersed starts: independent draws from the prior.
    starts = [model.sample_prior(rng) for _ in range(N_CHAINS)]
    log.info("starts (prior draws): %s", np.round(np.array(starts), 3).tolist())
    log.info("tuning proposal scale (pilot runs, discarded):")
    kernel = tune_kernel(model, GaussianRW.from_cov(cov), starts, rng)
    summary["proposal_sd"] = np.diag(kernel.chol).tolist()

    log.info("production: %d chains x %d iterations", N_CHAINS, N_RAW)
    raw = [MetropolisHastings(model, kernel).run(s, N_RAW, rng) for s in starts]
    draws = np.stack([c.draws for c in raw])  # (m, n, d)
    accept = [c.accept_rate for c in raw]
    summary["accept_rate_per_chain"] = accept
    log.info("acceptance rate per chain: %s", np.round(accept, 4).tolist())

    # Step 1: burn-in from windowed Gelman-Rubin; check with KS and R-hat on the kept draws.
    burn, win_grid, win_rhat = burn_in_windows(draws)
    post = draws[:, burn:]
    win_ok_after = float((win_rhat[win_grid >= burn] <= RHAT_MAX).all(axis=1).mean())
    log.info(
        "burn-in (windowed R-hat, window %d, threshold %.2f): B=%d; %.1f%% of later windows pass",
        GR_WIN, RHAT_MAX, burn, 100 * win_ok_after,
    )
    summary.update(burn=burn, gr_window=GR_WIN, rhat_max=RHAT_MAX)
    rhat_post = [gelman_rubin(post[:, :, j]) for j in range(2)]
    log.info("Gelman-Rubin R-hat on post-burn draws (w0, w1): %s", np.round(rhat_post, 4).tolist())
    summary["rhat_gelman_rubin_post"] = rhat_post

    ks_grid, ks_vals, ks_crit_value = ks_stats(draws)
    ks_after = ks_vals[:, :, ks_grid >= burn]
    frac = float((ks_after > ks_crit_value).mean())
    log.info(
        "KS check (segment %d, lag %d, crit %.4f): %.1f%% of post-burn comparisons reject",
        KS_SEG, KS_LAG, ks_crit_value, 100 * frac,
    )
    summary.update(ks_crit=ks_crit_value, ks_lag=KS_LAG, ks_seg=KS_SEG, ks_reject_frac_post=frac)

    # Step 2: thinning until kept draws are (nearly) uncorrelated.
    tau_post = [tau_multi(post[:, :, j]) for j in range(2)]
    thin = choose_thin(post)
    kept_probe = post[:, thin - 1 :: thin]
    tau_thin = max(tau_multi(kept_probe[:, :, j]) for j in range(2))
    # Keep enough thinned draws that the ESS is about N_KEEP.
    n_per_chain = math.ceil(N_KEEP * tau_thin / N_CHAINS)
    if post.shape[1] < thin * n_per_chain:
        raise RuntimeError(f"need {thin * n_per_chain} post-burn draws per chain; increase N_RAW")
    kept = post[:, thin - 1 :: thin][:, :n_per_chain]  # (m, n_per_chain, d)
    log.info(
        "thinning: raw tau_int=%s -> THIN=%d (tau_kept=%.3f); kept %d draws",
        np.round(tau_post, 2).tolist(), thin, tau_thin, kept.shape[0] * kept.shape[1],
    )
    summary.update(tau_int_raw=tau_post, thin=thin, tau_kept_max=tau_thin, n_kept=int(kept.size // 2))

    # Diagnostics table.
    ess_raw = [ess(post[:, :, j]) for j in range(2)]
    tau_kept = [tau_multi(kept[:, :, j]) for j in range(2)]
    ess_kept = [ess(kept[:, :, j]) for j in range(2)]
    for j, name in enumerate(PARAMS):
        log.info(
            "diagnostics %s: ESS(post)=%.0f tau(kept)=%.3f ESS(kept)=%.0f",
            name, ess_raw[j], tau_kept[j], ess_kept[j],
        )
    summary.update(ess_raw=ess_raw, tau_kept=tau_kept, ess_kept=ess_kept)

    # Posterior vs MLE.
    post_stats = {}
    for j, name in enumerate(PARAMS):
        mean, se, _ = mean_and_se(kept[:, :, j])
        sd = float(kept[:, :, j].std(ddof=1))
        q05, q95 = np.quantile(kept[:, :, j], [0.05, 0.95])
        post_stats[name] = dict(mean=mean, mc_se=se, sd=sd, q05=float(q05), q95=float(q95))
        log.info(
            "posterior %s: mean=%.4f (MC SE %.4f) sd=%.4f 90%% CI=(%.4f, %.4f) "
            "| MLE=%.4f (SE %.4f)",
            name, mean, se, sd, q05, q95, MLE[j], MLE_SE[j],
        )
    summary["posterior"] = post_stats

    # Predictive failure probabilities at X_STAR.
    z = kept[..., :1] + kept[..., 1:] * X_STAR  # (m, n_per_chain, 3)
    prob = sigmoid(z)
    plugin = sigmoid(MLE[0] + MLE[1] * X_STAR)
    preds = []
    for k, t in enumerate(X_STAR):
        p_hat, se, e = mean_and_se(prob[:, :, k])
        preds.append(dict(x_star=float(t), p_hat=p_hat, se=se, ess=e, plugin_mle=float(plugin[k])))
        log.info(
            "predictive %3.0f F: p_hat=%.4f SE=%.5f ESS=%.0f plug-in MLE=%.4f",
            t, p_hat, se, e, plugin[k],
        )
    summary["predictive"] = preds

    # Part (d): variance reduction by Rao-Blackwellization.
    comp = compare_predictive_estimators(prob, rng)
    for t, c in zip(X_STAR, comp, strict=True):
        log.info(
            "variance reduction %3.0f F: RB %.4f (SE %.5f) | simulated y_* %.4f (SE %.5f; "
            "Bernoulli sd %.5f, theory %.5f) | SE ratio %.2f, variance ratio %.2f",
            t, c["p_rb"], c["se_rb"], c["p_naive_mean"], c["se_naive"],
            c["sd_bernoulli"], c["sd_bernoulli_theory"], c["se_ratio"], c["var_ratio"],
        )
    summary["variance_reduction"] = [dict(x_star=float(t), **c) for t, c in zip(X_STAR, comp, strict=True)]

    # Figures.
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_trace(draws, burn, FIG_DIR / "trace.pdf")
    plot_burn_in(win_grid, win_rhat, ks_grid, ks_vals, ks_crit_value, burn, FIG_DIR / "burn_in.pdf")
    plot_posterior(kept, FIG_DIR / "posterior.pdf")
    plot_acf(post[0], kept[0], FIG_DIR / "acf.pdf")
    log.info("figures written to %s", FIG_DIR)

    # Save outputs.
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULTS / "p1_draws.npz",
        kept=kept,
        raw=draws,
        raw_log_p=np.stack([c.log_p for c in raw]),
        burn=burn,
        thin=thin,
    )
    with open(RESULTS / "p1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("saved %s and %s", RESULTS / "p1_draws.npz", RESULTS / "p1_summary.json")


if __name__ == "__main__":
    main()
