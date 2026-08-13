# From TD-Gammon to Portfolio Allocation

Reproducing Tesauro's TD-Gammon (1992) from scratch, then transferring the same
Temporal-Difference learning framework to dynamic multi-asset portfolio allocation.

**Why this project exists:** most RL-for-finance demos import a pre-built
library and skip straight to results. This project builds the algorithm layer
first — MLP, backpropagation, TD(λ) with eligibility traces, all in raw NumPy
— to demonstrate the mechanics, not just the API. The same trained approach is
then transferred to a genuine asset-management problem: choosing between
allocation regimes under changing market conditions.

## Status
**Complete.** Both stages delivered: a TD-Gammon agent trained on Hypergammon
(3 checkers per side), and a portfolio allocation agent trained on 10 liquid
ETFs, backtested out-of-sample.

## Key results

**Backgammon agent** — TD(λ)=0.7, 40,000 self-play games:
- **85.8% winrate vs. random play** (milestone: 70%)
- Config aligned with Tesauro's reference parameters (α: 0.1 → 0.01)

**Portfolio allocation agent** — out-of-sample test, 2020–2024:

| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Calmar |
|---|---|---|---|---|---|---|
| **Agent TD(λ)** | +11.1% | 20.7% | 0.44 | 0.53 | -31.8% | 0.35 |
| S&P 500 (SPY) | +14.4% | 21.1% | 0.59 | 0.70 | -33.7% | 0.43 |
| 60/40 | +5.3% | 13.4% | 0.24 | 0.31 | -28.3% | 0.19 |
| Equal-weight | +4.7% | 13.8% | 0.19 | 0.24 | -26.6% | 0.18 |

The agent beats both static benchmarks (60/40, equal-weight) on every
risk-adjusted metric, and reduces max drawdown vs. a pure S&P 500 position.
It does not beat SPY's Calmar ratio on this specific bull-market test window
— a limitation identified, diagnosed, and documented rather than hidden (see
`notebooks/04_portfolio_allocation.ipynb`, final section).

## Why this maps to asset management

| RL concept | AM equivalent |
|---|---|
| State (market features: returns, EWMA volatility, rolling correlations) | Regime indicators a multi-asset PM tracks |
| Action space (8 allocation profiles, incl. risk-parity and momentum) | Regime selection — a hierarchical, top-down allocation pattern described in the CFA Institute monograph *AI in Asset Management* (Halperin, Kolm & Ritter, 2025) |
| Reward (return net of transaction costs, drawdown penalty, turnover penalty) | Risk-adjusted mandate: not just return, but path quality and implementation cost |
| Temporal-difference credit assignment | Conceptually parallel to discounting/actualization in valuation |

## Architecture

```
src/
  backgammon_env.py    Game environment (rules, moves, capture, bear-off)
  portfolio_env.py     Features, allocation profiles, environment, tearsheet
notebooks/
  01_td_learning_random_walk.ipynb   TD(0) vs TD(lambda) foundations
  02_backgammon_environment.ipynb    Environment build + validation
  03_td_gammon_agent.ipynb           Agent training (NumPy MLP, TD(lambda))
  04_portfolio_allocation.ipynb      Transfer to portfolio allocation, backtest
outputs/                Exported charts (tracked); model weights (.pkl, not tracked — see below)
data/                   Cached market data (not tracked in git)
tests/                  Reserved for unit tests (not yet implemented)
```

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Run the notebooks in order (01 → 04). Each notebook is seeded
(`SEED = 42`) and self-contained: running it end-to-end regenerates the
trained weights, the exported charts, and reproduces the results reported
above. Model weights (`outputs/*.pkl`) are intentionally excluded from
version control (see `.gitignore`) — they are fully reproducible from the
notebooks rather than shipped as binary artifacts.

## Stack

- Python 3.14
- NumPy — neural network, backpropagation, and TD(λ) implemented from scratch
  (no ML framework: the point of this project is to demonstrate the mechanics)
- pandas — feature engineering on price data
- matplotlib — training curves, backtest visualizations
- yfinance — market data retrieval

See `requirements.txt` for exact versions.

## Reproducibility

- All environments seeded via `np.random.default_rng(seed)`, propagated
  explicitly through training and evaluation — no dependency on global
  `np.random` state
- Model weights verified non-empty on save (`os.path.getsize`) before any
  further processing
- Train/test split strictly chronological (train ≤ 2019, test 2020–2024);
  feature normalization statistics computed on train only

## Limitations & next steps

- Discrete action space (8 profiles) rather than continuous weights — a
  deliberate trade-off to stay within value-based TD(λ), at a cost in
  allocation granularity
- Single out-of-sample window (2020–2024, bull-market-dominated); a prolonged
  bear-market test (e.g. 2000–2002) would better stress-test the drawdown
  protection thesis
- Natural extension: policy-gradient methods (Actor-Critic, PPO) for
  continuous allocation weights

## References

- Tesauro, G. (1995). *Temporal Difference Learning and TD-Gammon*.
  Communications of the ACM.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*.
- Halperin, I., Kolm, P., Ritter, G. (2025). *AI in Asset Management*.
  CFA Institute Research Foundation.

## Author

M1 Finance, Université Paris-Dauphine