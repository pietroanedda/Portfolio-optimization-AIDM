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
