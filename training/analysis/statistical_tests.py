"""Statistical significance tests for trading strategy backtest results.

Provides tools to detect overfitting and validate that observed performance
is not due to chance:

  - Bootstrap confidence intervals for Sharpe ratio
  - Permutation test (H0: returns are random)
  - Deflated Sharpe Ratio (Bailey & López de Prado 2014) for multiple-testing correction
  - Regime-conditional performance breakdown

Week 32 implementation.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class StrategyStatisticalTests:
    """Statistical significance tests for backtested trading strategy returns."""

    # ------------------------------------------------------------------
    # 32.1a  Bootstrap Sharpe CI
    # ------------------------------------------------------------------

    def bootstrap_sharpe_ci(
        self,
        returns: np.ndarray,
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for annualised Sharpe ratio.

        Parameters
        ----------
        returns : np.ndarray
            Daily (or per-bar) return series.
        n_bootstrap : int
            Number of bootstrap resamples.
        ci : float
            Confidence level (0 < ci < 1), e.g. 0.95.

        Returns
        -------
        (lower, point_estimate, upper)
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            logger.warning("bootstrap_sharpe_ci: fewer than 2 returns, returning zeros")
            return (0.0, 0.0, 0.0)

        point = self._sharpe(returns)

        rng = np.random.default_rng()
        bootstrapped = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(returns, size=len(returns), replace=True)
            bootstrapped[i] = self._sharpe(sample)

        alpha = 1.0 - ci
        lower = float(np.percentile(bootstrapped, 100 * alpha / 2))
        upper = float(np.percentile(bootstrapped, 100 * (1 - alpha / 2)))

        logger.info(
            "Bootstrap Sharpe %.2f%% CI: [%.4f, %.4f, %.4f]",
            ci * 100, lower, point, upper,
        )
        return (lower, point, upper)

    # ------------------------------------------------------------------
    # 32.1b  Permutation test
    # ------------------------------------------------------------------

    def permutation_test(
        self,
        returns: np.ndarray,
        n_permutations: int = 10000,
    ) -> float:
        """Sign-randomization permutation test: H0 = returns are symmetric around 0 (no alpha).

        Randomly flips the sign of each return ``n_permutations`` times to build
        a null Sharpe distribution under H0 (zero-mean, no systematic bias), then
        computes the p-value as the fraction of permuted Sharpes >= observed Sharpe.

        Note: shuffling order is equivalent for IID Sharpe (order-invariant).
        Sign randomization is the correct null for testing mean > 0.

        Parameters
        ----------
        returns : np.ndarray
            Daily (or per-bar) return series.
        n_permutations : int
            Number of random sign assignments.

        Returns
        -------
        float
            p-value. Values < 0.05 indicate the strategy is statistically
            significant at the 5% level.
        """
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) < 2:
            logger.warning("permutation_test: fewer than 2 returns, returning p=1.0")
            return 1.0

        observed_sharpe = self._sharpe(returns)

        rng = np.random.default_rng()
        perm_sharpes = np.empty(n_permutations)
        for i in range(n_permutations):
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(returns))
            perm_sharpes[i] = self._sharpe(returns * signs)

        p_value = float(np.mean(perm_sharpes >= observed_sharpe))

        logger.info(
            "Permutation test (sign randomization) — observed Sharpe=%.4f, p-value=%.4f",
            observed_sharpe, p_value,
        )
        return p_value

    # ------------------------------------------------------------------
    # 32.1c  Deflated Sharpe Ratio
    # ------------------------------------------------------------------

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        var_sharpe: float,
        skew: float,
        kurt: float,
    ) -> float:
        """Deflated Sharpe Ratio (DSR) per Bailey & López de Prado (2014).

        Adjusts the observed Sharpe ratio downward to account for selection
        bias when multiple strategies or hyperparameter combinations were tried.

        Parameters
        ----------
        sharpe : float
            Observed annualised Sharpe ratio.
        n_trials : int
            Number of strategy/hyperparameter combinations tested (>= 1).
        var_sharpe : float
            Variance of the Sharpe ratio estimate (approx. 1/T for large T).
        skew : float
            Skewness of the return distribution.
        kurt : float
            Excess kurtosis of the return distribution.

        Returns
        -------
        float
            DSR: probability (0–1) that the observed SR is above the
            expected maximum SR under the null.  Values > 0.95 are
            considered statistically significant.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")

        # Expected maximum SR across n_trials iid trials (Eq. 4 in BLP 2014)
        # Using the approximation: E[max SR] ≈ Z^{-1}(1 - 1/n_trials) * sqrt(var_sharpe)
        if n_trials == 1:
            sr_star = 0.0
        else:
            # Euler-Mascheroni constant
            euler_gamma = 0.5772156649
            # E[max of n iid normals] ≈ (1 - euler_gamma)*Z^{-1}(1-1/n) + euler_gamma*Z^{-1}(1-1/(n*e))
            z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
            z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
            sr_star = (
                (1.0 - euler_gamma) * z1 + euler_gamma * z2
            ) * np.sqrt(var_sharpe)

        # Non-normality correction: adjust variance for skewness and kurtosis
        # σ(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / T)
        # Here we accept var_sharpe as already adjusted or raw 1/T.
        # Compute test statistic
        sigma_sr = np.sqrt(max(var_sharpe, 1e-10))
        test_stat = (sharpe - sr_star) * np.sqrt(1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe ** 2) / sigma_sr

        dsr = float(stats.norm.cdf(test_stat))

        logger.info(
            "DSR: observed SR=%.4f, SR*=%.4f, DSR=%.4f (n_trials=%d)",
            sharpe, sr_star, dsr, n_trials,
        )
        return dsr

    # ------------------------------------------------------------------
    # 32.1d  Regime-conditional report
    # ------------------------------------------------------------------

    def regime_conditional_report(
        self,
        returns: np.ndarray,
        regimes: np.ndarray,
    ) -> Dict[int, Dict[str, float]]:
        """Performance breakdown by market regime.

        Parameters
        ----------
        returns : np.ndarray
            Per-step return series, shape (T,).
        regimes : np.ndarray
            Integer regime labels aligned with ``returns``, shape (T,).
            Typically 0=bull, 1=sideways, 2=bear (or similar convention).

        Returns
        -------
        dict mapping regime_id → {sharpe, max_drawdown, win_rate, n_steps}
        """
        returns = np.asarray(returns, dtype=np.float64)
        regimes = np.asarray(regimes, dtype=int)

        if len(returns) != len(regimes):
            raise ValueError(
                f"returns ({len(returns)}) and regimes ({len(regimes)}) must have equal length"
            )

        report: Dict[int, Dict[str, float]] = {}
        for regime_id in np.unique(regimes):
            mask = regimes == regime_id
            r = returns[mask]
            if len(r) < 2:
                report[int(regime_id)] = {
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "n_steps": float(len(r)),
                }
                continue

            sharpe = self._sharpe(r)
            max_dd = self._max_drawdown(r)
            win_rate = float(np.mean(r > 0))

            report[int(regime_id)] = {
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "win_rate": win_rate,
                "n_steps": float(len(r)),
            }
            logger.info(
                "Regime %d — Sharpe=%.3f, MaxDD=%.3f, WinRate=%.3f, n=%d",
                regime_id, sharpe, max_dd, win_rate, len(r),
            )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Annualised Sharpe ratio (assumes daily bars)."""
        excess = returns - risk_free
        std = np.std(excess, ddof=1)
        if std < 1e-8:
            return 0.0
        return float(np.mean(excess) / std * np.sqrt(252))

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        """Maximum drawdown from cumulative return series."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns))
