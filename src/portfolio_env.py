"""
portfolio_env.py — Multi-asset allocation via profile selection.

Version history (kept for methodological traceability):

v1 -> v2: REWARD RECALIBRATION. In v1, reward = 100*(r - risk_aversion*r^2)
with r as a decimal: at daily frequency (r ~ 0.01), the penalty only
carried 1-3% of the signal. The agent effectively maximized raw return ->
100% SPY. In v2, returns are converted to percentage points BEFORE the
quadratic penalty is applied.

v2 -> v3 (current version): RICHER REWARD SHAPING. The simple quadratic
penalty on returns (v2) punished intra-step volatility but ignored two
risks that matter to a portfolio manager: cumulative DRAWDOWN (the risk of
being forced to sell at the worst moment) and TURNOVER (the real cost of
frequent rebalancing). v3 adds:
  - a penalty proportional to the square of the current drawdown (lambda_dd)
  - a penalty proportional to the action's turnover (lambda_turnover)
  - a terminal bonus proportional to the episode's realized Sharpe ratio,
    rewarding path quality rather than just the terminal value
This version (v3) is the one used for all results documented in
notebooks/04_portfolio_allocation.ipynb.

ACTIONS = ALLOCATION PROFILES, not a raw weight grid. With 10 assets, a
25%-step grid would produce hundreds of corner combinations — unmanageable.
Instead, the agent chooses among 8 professional profiles (from most
defensive to most aggressive, plus 2 DYNAMIC profiles whose weights adapt
to current market conditions). The agent becomes a "regime selector" — the
hierarchical pattern described in the CFA Institute monograph (Halperin,
Kolm & Ritter 2025): a high-level, strategic agent.

COMPACT CORRELATION FEATURES: with 10 assets, the 45 pairwise correlations
would saturate the state. We keep 2 aggregates: average market correlation
(a risk-on/risk-off regime signal) and equity-bond correlation (SPY-TLT,
THE diversification signal).

Universe (10 liquid ETFs, full history since 2010):
  SPY (US large cap)   QQQ (US tech)      IWM (US small cap)
  EFA (dev. intl.)     EEM (emerging)     TLT (20y+ Treasuries)
  IEF (7-10y Treasuries) LQD (IG credit)  GLD (gold)  VNQ (US REITs)
"""

import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "VNQ"]


# ============================================================
# 1. FEATURES
# ============================================================

def compute_features(prices, corr_window=60, momentum_window=60, ewma_lambda=0.94):
    """
    Per-asset features: 5d return, 20d return, 60d momentum, EWMA vol (RiskMetrics).
    Market-level features: average cross-asset correlation + SPY-TLT correlation.
    Also returns RAW vols and momentum (needed for the dynamic profiles).
    """
    returns = np.log(prices / prices.shift(1))
    assets = list(prices.columns)

    feats = {}
    raw_vols = {}
    for a in assets:
        r = returns[a]
        feats[f"ret5_{a}"] = r.rolling(5).mean()
        feats[f"ret20_{a}"] = r.rolling(20).mean()
        feats[f"mom_{a}"] = np.log(prices[a] / prices[a].shift(momentum_window))
        ewma_var = (r ** 2).ewm(alpha=1 - ewma_lambda).mean()
        raw_vols[a] = np.sqrt(ewma_var * 252)
        feats[f"vol_{a}"] = raw_vols[a]

    # Average market correlation (risk-on/off regime)
    rolling_corr = returns.rolling(corr_window).corr()
    avg_corr = rolling_corr.groupby(level=0).apply(
        lambda m: (m.values.sum() - len(m)) / (len(m) ** 2 - len(m)))
    feats["avg_corr"] = avg_corr

    # Equity-bond correlation (SPY-TLT): THE diversification signal
    feats["corr_SPY_TLT"] = returns["SPY"].rolling(corr_window).corr(returns["TLT"])

    features = pd.DataFrame(feats, index=prices.index)
    vols = pd.DataFrame(raw_vols, index=prices.index)
    momentum = pd.DataFrame({a: feats[f"mom_{a}"] for a in assets}, index=prices.index)
    return features, returns, vols, momentum


def normalize_features(features_train, features):
    """Z-score using TRAIN-only statistics (prevents look-ahead bias)."""
    mu = features_train.mean()
    sigma = features_train.std().replace(0, 1.0)
    return (features - mu) / sigma


# ============================================================
# 2. ALLOCATION PROFILES (the action space)
# ============================================================
# Asset order: SPY QQQ IWM EFA EEM | TLT IEF LQD | GLD | VNQ

STATIC_PROFILES = {
    # name: weights across the 10 assets (sum to 1.0)
    "Ultra-defensive": [0.10, 0.00, 0.00, 0.00, 0.00, 0.25, 0.30, 0.15, 0.15, 0.05],
    "Defensive":        [0.15, 0.05, 0.00, 0.10, 0.00, 0.20, 0.25, 0.10, 0.10, 0.05],
    "Balanced":         [0.25, 0.10, 0.00, 0.10, 0.05, 0.15, 0.10, 0.10, 0.10, 0.05],
    "Growth":           [0.30, 0.20, 0.05, 0.10, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05],
    "Aggressive":       [0.35, 0.25, 0.10, 0.10, 0.10, 0.00, 0.00, 0.00, 0.05, 0.05],
    "Equal-weight":     [0.10] * 10,
}
PROFILE_NAMES = list(STATIC_PROFILES.keys()) + ["Risk-parity", "Momentum"]


def risk_parity_weights(daily_vols):
    """Inverse-vol weights: proportional to 1/volatility (naive risk parity)."""
    inv = 1.0 / np.maximum(daily_vols, 1e-6)
    return inv / inv.sum()


def momentum_weights(daily_momentum):
    """Weights proportional to positive 60d momentum; equal-weight if all <= 0."""
    m = np.maximum(daily_momentum, 0.0)
    if m.sum() < 1e-9:
        return np.ones(len(m)) / len(m)
    return m / m.sum()


# ============================================================
# 3. ENVIRONMENT (v3 — canonical version used for all results)
# ============================================================

class PortfolioEnv:
    """
    At each step, the agent picks ONE profile out of 8 (see PROFILE_NAMES).
    Effective weights are those of the profile (static, or computed
    dynamically for Risk-parity and Momentum from the current day's
    conditions).

    Reward per step:
        reward = r_step - dd_penalty - turnover_penalty [+ terminal_bonus]

        r_step           = net return of the step, in PERCENTAGE POINTS
        dd_penalty        = lambda_dd * (current_drawdown_%)^2 / 100
                             (sharpens caution around tail risk)
        turnover_penalty = lambda_turnover * turnover of the action
        terminal_bonus    = only on the episode's last step, proportional
                             to the realized Sharpe ratio (clipped to
                             [-3, 3]) — rewards path quality, not just the
                             terminal value

    Increasing lambda_dd (1.0 -> 3.0) sharpens the drawdown penalty, pushing
    the agent toward more caution at the potential cost of a lower CAGR.
    This is a genuine return/protection trade-off, not a free parameter —
    see the systematic sweep over lambda_dd in the notebook.
    """

    def __init__(self, returns, features_norm, vols, momentum,
                 episode_length=60, transaction_cost=0.0005,
                 lambda_dd=2.0, lambda_turnover=0.5, seed=None):
        common = (features_norm.dropna().index
                  .intersection(returns.dropna().index)
                  .intersection(vols.dropna().index)
                  .intersection(momentum.dropna().index))
        self.returns = returns.loc[common].to_numpy()
        self.features = features_norm.loc[common].to_numpy()
        self.vols = vols.loc[common].to_numpy()
        self.momentum = momentum.loc[common].to_numpy()
        self.dates = common
        self.n_assets = self.returns.shape[1]
        self.n_actions = len(PROFILE_NAMES)
        self.episode_length = episode_length
        self.transaction_cost = transaction_cost
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.rng = np.random.default_rng(seed)

        self._static_profiles = np.array(list(STATIC_PROFILES.values()))
        # State = market features + current weights + time-remaining signal
        self.state_dim = self.features.shape[1] + self.n_assets + 1

    def profile_weights(self, k, t=None):
        """Effective weights of profile k under day t's conditions."""
        t = self.t if t is None else t
        if k < len(self._static_profiles):
            return self._static_profiles[k]
        if PROFILE_NAMES[k] == "Risk-parity":
            return risk_parity_weights(self.vols[t])
        return momentum_weights(self.momentum[t])

    def reset(self, t_start=None):
        t_max = len(self.returns) - self.episode_length - 1
        self.t_start = self.rng.integers(0, t_max) if t_start is None else t_start
        self.t = self.t_start
        self.steps_remaining = self.episode_length
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.episode_returns = []
        self.episode_equity = [1.0]
        return self._state(self.weights)

    def _state(self, weights):
        return np.concatenate([self.features[self.t], weights,
                               [self.steps_remaining / self.episode_length]])

    def candidate_states(self):
        """Afterstates: resulting state for each profile + the certain cost of switching."""
        X = np.empty((self.n_actions, self.state_dim))
        costs = np.empty(self.n_actions)
        for k in range(self.n_actions):
            w = self.profile_weights(k)
            X[k] = self._state(w)
            costs[k] = self.transaction_cost * np.abs(w - self.weights).sum()
        return X, costs

    def step(self, k):
        w = self.profile_weights(k)
        turnover = np.abs(w - self.weights).sum()
        cost = self.transaction_cost * turnover
        r_net = float(w @ self.returns[self.t + 1]) - cost
        self.episode_returns.append(r_net)

        new_equity = self.episode_equity[-1] * np.exp(r_net)
        self.episode_equity.append(new_equity)
        peak = max(self.episode_equity)
        current_dd = new_equity / peak - 1.0

        growth = np.exp(self.returns[self.t + 1])
        w_drifted = w * growth
        self.weights = w_drifted / w_drifted.sum()

        self.t += 1
        self.steps_remaining -= 1
        done = self.steps_remaining == 0

        r_step = r_net * 100.0
        dd_penalty = self.lambda_dd * (current_dd * 100) ** 2 / 100.0
        turnover_penalty = self.lambda_turnover * turnover

        if done:
            r_arr = np.array(self.episode_returns)
            episode_sharpe = r_arr.mean() / max(r_arr.std(), 1e-8) * np.sqrt(252)
            terminal_bonus = np.clip(episode_sharpe, -3, 3) * 2.0
            reward = r_step - dd_penalty - turnover_penalty + terminal_bonus
        else:
            reward = r_step - dd_penalty - turnover_penalty

        return self._state(self.weights), reward, done

    def check_invariant(self):
        assert abs(self.weights.sum() - 1.0) < 1e-8, f"Weights sum to {self.weights.sum()}"


# ============================================================
# 4. TEARSHEET — single source of truth (do not redefine in notebooks)
# ============================================================

def tearsheet(log_returns, risk_free_rate=0.02, trading_days=252):
    """
    Computes standard risk-adjusted performance metrics.

    Convention on degenerate cases (few/no negative returns, etc.): returns
    NaN rather than an arbitrary default (0 or 1), so these cases stay
    visible instead of being silently confused with a genuine zero.
    """
    r = np.asarray(log_returns, dtype=float)
    n = len(r)
    equity = np.exp(np.cumsum(r))
    cagr = float(equity[-1] ** (trading_days / n) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else np.nan
    r_neg = r[r < 0]
    vol_neg = float(r_neg.std(ddof=1) * np.sqrt(trading_days)) if len(r_neg) > 1 else np.nan
    sortino = (cagr - risk_free_rate) / vol_neg if vol_neg and vol_neg > 0 else np.nan
    peaks = np.maximum.accumulate(equity)
    max_dd = float((equity / peaks - 1).min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": max_dd, "Calmar": calmar}