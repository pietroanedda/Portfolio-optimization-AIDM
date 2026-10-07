"""
Minimum variance and maximum expected return (under target volatility) portfolio optimization with analytic solutions.

Input data:
    mu    : vector of expected returns, shape (n,)
    sigma : vector of volatilities, shape (n,)
    corr  : correlation matrix, shape (n, n)

Functions:
    covariance_matrix           Sigma = DRD (R correlation matrix, D diagonal volatility matrix)
    check_correlation_matrix    checks for negative eigenvalues and cross-correlations equal to ±1 
    min_variance_portfolio      minimum variance portfolio (closed form solution)
    max_return_portfolio        maximum expected return portfolio under target volatility (soluzione analitica)
    optimize_portfolio          chooses optimization problem, imposes sparsity (at most K titoli) and long-only
    _plot_closed_form_frontier  computes the efficient frontier without constraints
    plot_efficient_frontier     plots efficient frontier
    plot_portfolio_weights      plots bar graphs of weights

"""

from itertools import combinations
from dataclasses import dataclass 

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Colors for the plots.
COLORS = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "frontier": "#2a78d6",
    "constrained": "#eb6834",
    "long": "#2a78d6",
    "short": "#e34948",
}

METHOD_LABELS = {"min_variance": "minimum variance", "max_return": "maximum expected return"}


@dataclass
class PortfolioResult:

    weights: np.ndarray
    expected_return: float
    volatility: float
    assets: tuple[int, ...] 
    method: str


def covariance_matrix(volatilities, correlation) -> np.ndarray:
    """Computes covariance matrix given a correlation matrix and an array of volatilities."""
    vol = np.asarray(volatilities, dtype=float)
    corr = np.asarray(correlation, dtype=float)
    return corr * np.outer(vol, vol)


def _perfect_correlation_mask(corr, tol = 1e-10) -> np.ndarray:
    """Boolean mask for off-diagonal correlations greater than or equal to 1 in modulus."""
    return (np.abs(corr) >= 1.0 - tol) & ~np.eye(len(corr), dtype=bool)


def check_correlation_matrix(
    correlation, tol = 1e-10, check_perfect = True
) -> tuple[bool, list[str]]:
    """
    Verify that the correlation matrix is valid.

    Checks:
      - square matrix, symmetric and diagonal elements equal to 1;
      - no off-diagonal values equal to ±1 in modulus (only if check_perfect);
      - no negative eigenvalues (the matrix must be positive semi-definite).

    Returns (valid, list_of_problems).
    """
    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        return False, [f"the matrix must be squared, received shape {corr.shape}"]

    problems = []

    if not np.allclose(corr, corr.T, atol=tol):
        problems.append("the matrix is not symmetric")

    if not np.allclose(np.diag(corr), 1.0, atol=tol):
        problems.append("the diagonal contains elements different from 1")

    if check_perfect:
        for i, j in np.argwhere(_perfect_correlation_mask(corr, tol)):
            if i < j:
                problems.append(f"off-diagonal correlation equal to {corr[i, j]:+.4f} in position ({i}, {j})")

    eigenvalues = np.linalg.eigvalsh((corr + corr.T) / 2)
    negative = eigenvalues[eigenvalues < -tol]
    if negative.size:
        values = ", ".join(f"{v:.4g}" for v in negative)
        problems.append(f"the matrix has {negative.size} negative eigenvalues: {values}")

    return not problems, problems


def _frontier_constants(expected_returns, cov):
    """Constants of the Markowitz efficient frontier."""

    mu = np.asarray(expected_returns, dtype=float)
    ones = np.ones_like(mu)
    solved = np.linalg.solve(cov, np.column_stack([ones, mu]))
    inv_ones, inv_mu = solved[:, 0], solved[:, 1]
    a = ones @ inv_ones
    b = ones @ inv_mu
    c = mu @ inv_mu
    return inv_ones, inv_mu, a, b, c, a * c - b**2


def min_variance_portfolio(cov) -> np.ndarray:
    """Minimum variance portfolio (GMV), analytic solution"""
    inv_ones = np.linalg.solve(cov, np.ones(len(cov)))
    return inv_ones / inv_ones.sum()


def max_return_portfolio(expected_returns, cov, target_vol: float) -> np.ndarray:
    """
    Maximum expected return under target volatility portfolio;
    Admissible problem only if target_vol >= 1/sqrt(A). 
    If all expected returns are equal (D = 0, e.g. only one asset) each portfolio
    has the same return: it returns GMV, with volatility <= target_vol.
    """
    mu = np.asarray(expected_returns, dtype=float)
    inv_ones, inv_mu, a, b, c, d = _frontier_constants(mu, cov)

    excess = a * target_vol**2 - 1.0  
    if excess < -1e-10:
        raise ValueError(
            f"volatility target {target_vol:.4%} lower than feasible minimum "
            f"volatility {1 / np.sqrt(a):.4%}"
        )

    w_gmv = inv_ones / a
    if d <= 1e-12 * a * c:  # equal expected returns: degenerate frontier
        return w_gmv

    t = np.sqrt(max(excess, 0.0) / d)
    return w_gmv + t * (inv_mu - (b / a) * inv_ones)


def _validate_inputs(expected_returns, volatilities, correlation):
    mu = np.asarray(expected_returns, dtype=float)
    vol = np.asarray(volatilities, dtype=float)
    corr = np.asarray(correlation, dtype=float)

    if mu.ndim != 1 or vol.shape != mu.shape:
        raise ValueError("expected returns and volatilities must be arrays of same length")
    if corr.shape != (mu.size, mu.size):
        raise ValueError(f"the correlation matrix must have shape ({mu.size}, {mu.size})")
    if np.any(vol <= 0):
        raise ValueError("volatilities must be strictly positive")

    valid, problems = check_correlation_matrix(corr, check_perfect=False)
    if not valid:
        raise ValueError("correlation matrix not valid:\n  - " + "\n  - ".join(problems))
    return mu, vol, corr


def optimize_portfolio(
    expected_returns,
    volatilities,
    correlation,
    method: str = "min_variance",
    target_vol: float | None = None,
    max_assets: int | None = None,
    long_only: bool = True,
    tol: float = 1e-10,
) -> PortfolioResult:
    
    if method not in ("min_variance", "max_return"):
        raise ValueError("method must be 'min_variance' or 'max_return'")
    if method == "max_return" and (target_vol is None or target_vol <= 0):
        raise ValueError("for 'max_return' target_vol must be positive")

    mu, vol, corr = _validate_inputs(expected_returns, volatilities, correlation)
    cov = covariance_matrix(vol, corr)
    n = mu.size
    k_max = n if max_assets is None else min(int(max_assets), n)
    if k_max < 1:
        raise ValueError("max_assets must be at least 1")

    perfect = _perfect_correlation_mask(corr, tol)
    best = None
    best_objective = best_variance = np.inf

    for k in range(1, k_max + 1):
        for subset in combinations(range(n), k):
            idx = list(subset)
            if perfect[np.ix_(idx, idx)].any():
                continue  # couple with correlation +-1: singular covariance matrix of subset

            sub_cov = cov[np.ix_(idx, idx)]
            try:
                if method == "min_variance":
                    w_sub = min_variance_portfolio(sub_cov)
                else:
                    w_sub = max_return_portfolio(mu[idx], sub_cov, target_vol)
            except (ValueError, np.linalg.LinAlgError):
                continue  # unfeasible subset or singular covariance matrix

            if long_only:
                if np.any(w_sub < -tol):
                    continue
                w_sub = np.clip(w_sub, 0.0, None)
                w_sub /= w_sub.sum()

            weights = np.zeros(n)
            weights[idx] = w_sub
            variance = weights @ cov @ weights
            objective = variance if method == "min_variance" else -(mu @ weights)

            if objective < best_objective - tol or (
                abs(objective - best_objective) <= tol and variance < best_variance - tol
            ):
                best, best_objective, best_variance = weights, objective, variance

    if best is None:
        raise ValueError("no feasible portfolio with requested constraints")

    return PortfolioResult(
        weights=best,
        expected_return=float(mu @ best),
        volatility=float(np.sqrt(best_variance)),
        assets=tuple(int(i) for i in np.flatnonzero(np.abs(best) > tol)),
        method=method,
    )


def _plot_closed_form_frontier(ax, mu, cov) -> None:

    try:
        _, _, a, b, c, d = _frontier_constants(mu, cov)
    except np.linalg.LinAlgError:
        return
    if d <= 1e-12 * a * c:  
        return
    
    span = mu.max() - mu.min()
    m = np.linspace(mu.min() - 0.3 * span, mu.max() + 0.3 * span, 400)
    m = np.union1d(m, [b / a])  
    s = np.sqrt((a * m**2 - 2 * b * m + c) / d)
    efficient = m >= b / a
    ax.plot(s[efficient], m[efficient], color=COLORS["frontier"], lw=2,
            solid_capstyle="round", label="Frontier without constraints")
    ax.plot(s[~efficient], m[~efficient], color=COLORS["frontier"], lw=1.5,
            ls=(0, (4, 3)), alpha=0.6, label="Inefficient branch")
    ax.plot(1 / np.sqrt(a), b / a, "o", ms=8, color=COLORS["frontier"],
            mec=COLORS["surface"], mew=2, zorder=4)


def plot_efficient_frontier(
    expected_returns,
    volatilities,
    correlation,
    max_assets: int | None = None,
    long_only: bool = True,
    asset_names=None,
    portfolios: dict[str, PortfolioResult] | None = None,
    n_points: int = 100,
    ax=None,
):
  
    mu, vol, corr = _validate_inputs(expected_returns, volatilities, correlation)
    cov = covariance_matrix(vol, corr)
    n = mu.size
    names = list(asset_names) if asset_names is not None else [f"Asset {i + 1}" for i in range(n)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=COLORS["surface"])
    else:
        fig = ax.figure
    ax.set_facecolor(COLORS["surface"])

    has_perfect = _perfect_correlation_mask(corr).any()
    if not has_perfect:
        _plot_closed_form_frontier(ax, mu, cov)

    k = n if max_assets is None else min(int(max_assets), n)
    if long_only or k < n or has_perfect:
        gmv = optimize_portfolio(mu, vol, corr, "min_variance", max_assets=k, long_only=long_only)
        top_vol = vol.max() if long_only else 1.3 * vol.max()
        points = [(gmv.volatility, gmv.expected_return)]
        for target in np.linspace(gmv.volatility, max(top_vol, gmv.volatility), n_points)[1:]:
            p = optimize_portfolio(mu, vol, corr, "max_return", target_vol=target,
                                   max_assets=k, long_only=long_only)
            points.append((p.volatility, p.expected_return))
        cs, cm = np.array(points).T

        parts = ((["long-only"] if long_only else []) + ([f"K ≤ {k}"] if k < n else [])
                 + (["without couples with ρ = ±1"] if has_perfect else []))
        ax.plot(cs, cm, color=COLORS["constrained"], lw=2, solid_capstyle="round",
                label=f"Frontier with constraints ({', '.join(parts)})")
        ax.plot(cs[0], cm[0], "o", ms=8, color=COLORS["constrained"],
                mec=COLORS["surface"], mew=2, zorder=4)

    ax.scatter(vol, mu, s=64, color=COLORS["muted"], edgecolors=COLORS["surface"],
               linewidths=2, zorder=5, label="Single assets")
    for x, y, name in zip(vol, mu, names):
        ax.annotate(name, (x, y), xytext=(7, -3), textcoords="offset points",
                    fontsize=9, color=COLORS["ink_secondary"])

    for label, p in (portfolios or {}).items():
        ax.scatter(p.volatility, p.expected_return, s=110, marker="D", color=COLORS["ink"],
                   edgecolors=COLORS["surface"], linewidths=2, zorder=6)
        ax.annotate(label, (p.volatility, p.expected_return), xytext=(-8, 8),
                    textcoords="offset points", ha="right", fontsize=9,
                    fontweight="bold", color=COLORS["ink"])

    ax.set_title("Efficient frontier", loc="left", fontsize=13, color=COLORS["ink"], pad=12)
    ax.set_xlabel("Volatility", color=COLORS["ink_secondary"])
    ax.set_ylabel("Expected return", color=COLORS["ink_secondary"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(left=0)
    ax.grid(True, color=COLORS["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLORS["axis"])
    legend = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(COLORS["ink_secondary"])
    fig.tight_layout()
    return fig, ax


def plot_portfolio_weights(
    result: PortfolioResult,
    asset_names=None,
    title: str | None = None,
    hide_zero: bool = False,
    sort: bool = False,
    ax=None,
):
   
    weights = np.asarray(result.weights, dtype=float)
    n = weights.size
    names = np.array(list(asset_names) if asset_names is not None else [f"Asset {i + 1}" for i in range(n)])

    order = np.array(result.assets) if hide_zero else np.arange(n)
    if sort:
        order = order[np.argsort(-weights[order], kind="stable")]
    values = weights[order]
    y = np.arange(order.size)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 1.4 + 0.45 * order.size), facecolor=COLORS["surface"])
    else:
        fig = ax.figure
    ax.set_facecolor(COLORS["surface"])

    ax.barh(y, values, height=0.5, color=np.where(values < 0, COLORS["short"], COLORS["long"]), zorder=3)
    ax.axvline(0, color=COLORS["axis"], lw=1, zorder=4)

    for yi, v in zip(y, values):
        ax.annotate(f"{v:.1%}", (v, yi), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9,
                    color=COLORS["ink_secondary"] if v != 0 else COLORS["muted"])

    lo, hi = min(values.min(), 0.0), max(values.max(), 0.0)
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo - pad if lo < 0 else 0.0, hi + pad)

    ax.set_yticks(y, names[order])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(True, axis="x", color=COLORS["grid"], lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["muted"], length=0)
    ax.tick_params(axis="y", colors=COLORS["ink_secondary"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is None:
        title = f"Portfolio weights · {METHOD_LABELS.get(result.method, result.method)}"
    ax.set_title(title, loc="left", fontsize=12, color=COLORS["ink"], pad=24)
    ax.annotate(f"μ = {result.expected_return:.2%}   σ = {result.volatility:.2%}",
                (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                fontsize=9, color=COLORS["ink_secondary"])
    fig.tight_layout()
    return fig, ax


def print_portfolio(title: str, result: PortfolioResult, asset_names) -> None:
    print(f"\n{title}")
    print(f"  Expected return: {result.expected_return:8.4%}")
    print(f"  Volatility:        {result.volatility:8.4%}")
    for i in result.assets:
        print(f"  {asset_names[i]:<20s} {result.weights[i]:8.2%}")


def main() -> None:
    names = ["USA Equities", "EU Equities", "Emerging markets",
             "Gov. Bonds", "Corp. Bonds", "Raw Materials"]
    mu = np.array([0.080, 0.070, 0.095, 0.025, 0.040, 0.050])
    sigma = np.array([0.16, 0.18, 0.24, 0.05, 0.08, 0.21])
    corr = np.array([
        [1.00, 0.80, 0.70, -0.20, 0.30, 0.30],
        [0.80, 1.00, 0.70, -0.10, 0.35, 0.30],
        [0.70, 0.70, 1.00, -0.10, 0.40, 0.40],
        [-0.20, -0.10, -0.10, 1.00, 0.50, -0.10],
        [0.30, 0.35, 0.40, 0.50, 1.00, 0.10],
        [0.30, 0.30, 0.40, -0.10, 0.10, 1.00],
    ])

    valid, problems = check_correlation_matrix(corr)
    print("Valid correlation matrix:", valid)

    bad_corr = corr.copy()
    bad_corr[0, 1] = bad_corr[1, 0] = 1.0
    bad_corr[0, 3] = bad_corr[3, 0] = 0.9
    print("\nExample of nonvalid matrix:")
    for problem in check_correlation_matrix(bad_corr)[1]:
        print("  -", problem)

    print("\nCovariance matrix:")
    print(np.array2string(covariance_matrix(sigma, corr), precision=5, suppress_small=True))

    K = 3
    target_vol = 0.12

    gmv = optimize_portfolio(mu, sigma, corr, "min_variance", max_assets=K, long_only=True)
    print_portfolio(f"Minimum variance (long-only, al più {K} titoli)", gmv, names)

    best = optimize_portfolio(mu, sigma, corr, "max_return", target_vol=target_vol,
                              max_assets=K, long_only=True)
    print_portfolio(f"Maximum expected return with volatility ≤ {target_vol:.0%} "
                    f"(long-only, at most {K} assets)", best, names)

    fig, _ = plot_efficient_frontier(
        mu, sigma, corr, max_assets=K, long_only=True, asset_names=names,
        portfolios={"Minimum variance": gmv, f"Max expected return (σ ≤ {target_vol:.0%})": best},
    )
    fig.savefig("efficient_frontier.png", dpi=150, facecolor=fig.get_facecolor())
    print("\nPlot saved in efficient_frontier.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=COLORS["surface"])
    plot_portfolio_weights(gmv, names, ax=axes[0])
    plot_portfolio_weights(best, names, ax=axes[1])
    fig.savefig("portfolio_weights.png", dpi=150, facecolor=fig.get_facecolor())
    print("Plot saved in portfolio_weights.png")

    if plt.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()