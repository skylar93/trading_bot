"""
Statistical Tests for Strategy Validation.

Implements bootstrap confidence intervals, permutation tests, deflated
Sharpe ratio (Bailey & López de Prado 2014), and regime-conditional
performance reporting.

Usage
-----
    from training.analysis.statistical_tests import StrategyStatisticalTests
    import numpy as np

    st = StrategyStatisticalTests()
    lo, mid, hi = st.bootstrap_sharpe_ci(returns)
    p_value = st.permutation_test(returns)
    dsr = st.deflated_sharpe_ratio(sharpe=1.2, n_trials=50, ...)
    report = st.regime_conditional_report(returns, regimes)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def _sharpe_from_returns(returns: np.ndarray, annualize: bool = True) -> float:
    """Annualized Sharpe ratio from a 1-D array of period returns."""
    if len(returns) < 2:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-12:
        return 0.0
    raw = mu / sigma
    return float(raw * np.sqrt(252)) if annualize else float(raw)


class StrategyStatisticalTests:
    """Statistical significance tools for backtested trading strategies."""

    # ------------------------------------------------------------------
    # 1. Bootstrap Sharpe CI
    # ------------------------------------------------------------------

    def bootstrap_sharpe_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 10_000,
        ci: float = 0.95,
        random_state: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for the annualised Sharpe ratio.

        Parameters
        ----------
        returns : array-like
            Period returns (e.g. daily).
        n_bootstrap : int
            Number of bootstrap resamples.
        ci : float
            Confidence level (default 0.95 → 95 % CI).
        random_state : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        (lower, point_estimate, upper) : tuple of float
        """
        returns = np.asarray(returns, dtype=float)
        rng = np.random.default_rng(random_state)

        point_estimate = _sharpe_from_returns(returns)

        boot_sharpes = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=len(returns), replace=True)
            boot_sharpes[i] = _sharpe_from_returns(sample)

        alpha = 1.0 - ci
        lower = float(np.percentile(boot_sharpes, 100 * alpha / 2))
        upper = float(np.percentile(boot_sharpes, 100 * (1 - alpha / 2)))
        return lower, point_estimate, upper

    # ------------------------------------------------------------------
    # 2. Permutation test
    # ------------------------------------------------------------------

    def permutation_test(
        self,
        returns: np.ndarray,
        n_permutations: int = 10_000,
        random_state: Optional[int] = None,
    ) -> float:
        """One-sided sign-flip permutation test: H0 = mean return = 0.

        Tests whether the strategy's mean return is significantly positive.
        Under H0, randomly flipping the sign of each return is a valid
        permutation (each return is equally likely positive or negative), so
        we can build the null distribution of the Sharpe ratio.

        Parameters
        ----------
        returns : array-like
            Period returns of the strategy.
        n_permutations : int
            Number of sign-flip permutations.
        random_state : int, optional
            RNG seed.

        Returns
        -------
        float
            p-value.  Values below 0.05 indicate the strategy's returns are
            unlikely to have arisen from a zero-mean process.
        """
        returns = np.asarray(returns, dtype=float)
        rng = np.random.default_rng(random_state)

        observed_sharpe = _sharpe_from_returns(returns)

        null_sharpes = np.empty(n_permutations)
        for i in range(n_permutations):
            signs = rng.choice([-1.0, 1.0], size=len(returns))
            flipped = returns * signs
            null_sharpes[i] = _sharpe_from_returns(flipped)

        # Fraction of null sharpes >= observed (one-sided upper tail)
        p_value = float(np.mean(null_sharpes >= observed_sharpe))
        return p_value

    # ------------------------------------------------------------------
    # 3. Deflated Sharpe Ratio (Bailey & López de Prado 2014)
    # ------------------------------------------------------------------

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        var_sharpe: float,
        skew: float,
        kurt: float,
    ) -> float:
        """Deflated Sharpe Ratio corrected for multiple testing.

        Adjusts the observed Sharpe ratio for the expected maximum Sharpe
        that would arise by chance when testing ``n_trials`` strategies.

        Parameters
        ----------
        sharpe : float
            Observed (in-sample) annualised Sharpe ratio.
        n_trials : int
            Number of strategy/hyperparameter combinations tried.
        var_sharpe : float
            Variance of the Sharpe ratio estimate.
        skew : float
            Skewness of returns.
        kurt : float
            Excess kurtosis of returns.

        Returns
        -------
        float
            Probability that the strategy beats a random benchmark
            (DSR ∈ [0, 1]).
        """
        if n_trials <= 1:
            # No multiple-testing correction needed
            z = sharpe / (np.sqrt(var_sharpe) + 1e-12)
            return float(scipy_stats.norm.cdf(z))

        # Expected maximum SR under multiple testing (Equation 8, BLP 2014)
        # gamma_euler ≈ 0.5772
        gamma = 0.5772156649
        e_max_sr = (
            (1.0 - gamma) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
            + gamma * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
        # Adjust for non-normality
        sr_star = e_max_sr * np.sqrt(var_sharpe + 1e-12)

        # Variance of Sharpe under non-normality (Lo 2002)
        t = max(n_trials, 2)
        sigma_sr = np.sqrt(
            (1.0 + 0.5 * sharpe ** 2 - skew * sharpe + ((kurt - 3) / 4) * sharpe ** 2) / t
        )

        z = (sharpe - sr_star) / (sigma_sr + 1e-12)
        dsr = float(scipy_stats.norm.cdf(z))
        return dsr

    # ------------------------------------------------------------------
    # 4. Regime-conditional performance report
    # ------------------------------------------------------------------

    def regime_conditional_report(
        self,
        returns: np.ndarray,
        regimes: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """Decompose strategy performance by market regime.

        Parameters
        ----------
        returns : array-like, shape (T,)
            Period returns.
        regimes : array-like, shape (T,)
            Integer regime label for each period (e.g. 0=bear, 1=sideways,
            2=bull from HMM).

        Returns
        -------
        dict
            ``{regime_id: {sharpe, max_drawdown, win_rate, n_trades, mean_return}}``
        """
        returns = np.asarray(returns, dtype=float)
        regimes = np.asarray(regimes, dtype=int)

        if len(returns) != len(regimes):
            raise ValueError(
                f"returns length {len(returns)} != regimes length {len(regimes)}"
            )

        report: Dict[int, Dict[str, float]] = {}
        for regime_id in np.unique(regimes):
            mask = regimes == regime_id
            r = returns[mask]

            if len(r) < 2:
                report[int(regime_id)] = {
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": float(np.mean(r > 0)) if len(r) else 0.0,
                    "n_trades": int(len(r)),
                    "mean_return": float(np.mean(r)) if len(r) else 0.0,
                }
                continue

            sharpe = _sharpe_from_returns(r)

            # Max drawdown from cumulative returns
            cum = np.cumprod(1.0 + r)
            running_max = np.maximum.accumulate(cum)
            drawdowns = (cum - running_max) / running_max
            max_dd = float(np.min(drawdowns))

            win_rate = float(np.mean(r > 0))
            mean_ret = float(np.mean(r))

            report[int(regime_id)] = {
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "n_trades": int(len(r)),
                "mean_return": mean_ret,
            }

        return report
