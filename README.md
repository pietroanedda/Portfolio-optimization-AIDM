# Portfolio Optimization

Markowitz mean–variance portfolio optimization in Python, built on **closed-form (analytic) solutions**.

Given each asset's expected return, volatility and the correlation matrix, the script computes:

- the **minimum variance portfolio** (global minimum variance, GMV);
- the **maximum expected return portfolio** for a target volatility;
- both under optional **long-only** and **sparsity** (at most *K* assets) constraints;
- the **efficient frontier**, with and without constraints, and bar charts of the portfolio weights.

## Requirements

- Python 3.10 or later
- `numpy`
- `matplotlib`

```bash
pip install -r requirements.txt
```

## Usage

Run the built-in example:

```bash
python portfolio_opt_Pietro_Anedda.py
```

The example uses six asset classes (USA equities, EU equities, emerging markets, government bonds, corporate bonds, raw materials) and:

1. checks that the correlation matrix is valid and shows the problems found in a deliberately broken one;
2. prints the covariance matrix;
3. computes the minimum variance portfolio and the maximum expected return portfolio with volatility ≤ 12%, both long-only and with at most 3 assets;
4. saves two charts in the current working directory:
   - `efficient_frontier.png`: unconstrained frontier, constrained frontier, single assets and the two optimal portfolios;
   - `portfolio_weights.png`: weights of the two portfolios.

Output of the example:

```
Minimum variance (long-only, al più 3 titoli)
  Expected return:  3.2138%
  Volatility:         4.3639%
  USA Equities           11.27%
  Gov. Bonds             84.97%
  Raw Materials           3.77%

Maximum expected return with volatility ≤ 12% (long-only, at most 3 assets)
  Expected return:  6.7992%
  Volatility:        12.0000%
  USA Equities           56.64%
  Emerging markets        9.70%
  Corp. Bonds            33.66%
```

### Using the functions on your own data

```python
import numpy as np
from portfolio_opt_Pietro_Anedda import (
    check_correlation_matrix,
    optimize_portfolio,
    plot_efficient_frontier,
    plot_portfolio_weights,
)

names = ["Equities", "Bonds", "Gold"]
mu    = np.array([0.07, 0.03, 0.04])   # expected returns
sigma = np.array([0.15, 0.05, 0.12])   # volatilities
corr  = np.array([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.0],
    [0.1, 0.0, 1.0],
])

valid, problems = check_correlation_matrix(corr)

gmv = optimize_portfolio(mu, sigma, corr, method="min_variance")
best = optimize_portfolio(mu, sigma, corr, method="max_return",
                          target_vol=0.08, max_assets=2, long_only=True)

print(best.weights, best.expected_return, best.volatility, best.assets)

fig, ax = plot_efficient_frontier(mu, sigma, corr, max_assets=2, asset_names=names,
                                  portfolios={"GMV": gmv, "Max return": best})
fig, ax = plot_portfolio_weights(best, names)
```

`optimize_portfolio` returns a `PortfolioResult` with these fields:

| Field             | Description                                        |
|-------------------|----------------------------------------------------|
| `weights`         | weight vector, shape `(n,)`, summing to 1          |
| `expected_return` | portfolio expected return                          |
| `volatility`      | portfolio volatility                               |
| `assets`          | indices of the assets with non-zero weight         |
| `method`          | `"min_variance"` or `"max_return"`                 |

## Main functions

| Function                    | Description |
|-----------------------------|-------------|
| `covariance_matrix`         | Builds the covariance matrix Σ = D R D from volatilities and correlations. |
| `check_correlation_matrix`  | Checks that the matrix is square, symmetric, has a unit diagonal, has no off-diagonal correlations equal to ±1 and no negative eigenvalues. Returns `(valid, problems)`. |
| `min_variance_portfolio`    | Global minimum variance portfolio, closed form. |
| `max_return_portfolio`      | Maximum expected return portfolio for a target volatility, closed form. |
| `optimize_portfolio`        | Chooses the problem (`min_variance` / `max_return`) and applies the long-only and at-most-*K*-assets constraints. |
| `plot_efficient_frontier`   | Plots the efficient frontier (unconstrained and constrained), the single assets and any given portfolios. |
| `plot_portfolio_weights`    | Horizontal bar chart of the portfolio weights (long in blue, short in red). |

## Method

### Covariance matrix

With *D* the diagonal matrix of volatilities and *R* the correlation matrix:

$$\Sigma = D\,R\,D$$

### Analytic solutions

Define the frontier constants:

$$A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}, \quad B = \mathbf{1}^\top \Sigma^{-1} \mu, \quad C = \mu^\top \Sigma^{-1} \mu, \quad \Delta = AC - B^2$$

**Minimum variance** (fully invested, $\mathbf{1}^\top w = 1$):

$$w_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf{1}}{A}, \qquad \sigma_{\text{GMV}} = \frac{1}{\sqrt{A}}, \qquad \mu_{\text{GMV}} = \frac{B}{A}$$

**Maximum expected return** with $w^\top \Sigma w \le \sigma_{\text{target}}^2$ (feasible only if $\sigma_{\text{target}} \ge 1/\sqrt{A}$):

$$w^{\star} = w_{\text{GMV}} + t \left(\Sigma^{-1}\mu - \frac{B}{A}\,\Sigma^{-1}\mathbf{1}\right), \qquad t = \sqrt{\frac{A\,\sigma_{\text{target}}^2 - 1}{\Delta}}$$

If all expected returns are equal ($\Delta = 0$), every portfolio has the same return and the GMV portfolio is returned.

**Unconstrained efficient frontier**:

$$\sigma^2(m) = \frac{A m^2 - 2 B m + C}{\Delta}, \qquad m \ge \frac{B}{A}$$

### Long-only and sparsity constraints

With the constraints enabled, `optimize_portfolio` enumerates every subset of at most *K* assets and, for each one:

- skips it if it contains a pair with correlation ±1 (singular covariance matrix);
- computes the analytic solution restricted to that subset;
- with `long_only=True`, discards it if any weight is negative;
- keeps the subset with the best objective (lowest variance, or highest expected return), breaking ties by lower variance.

The result is the exact optimum of the constrained problem. The number of subsets grows combinatorially with *n* and *K*, so this approach suits a small number of assets.

## Repository structure

```
.
├── portfolio_opt_Pietro_Anedda.py   # optimization, plotting functions and example (main)
├── requirements.txt                 # dependencies
└── README.md
```

## Author

Pietro Anedda
