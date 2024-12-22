"""
Advanced backtesting scenarios for trading bot.
Implements complex market conditions and edge cases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from envs.trading_env import TradingEnvironment
from agents.strategies.single.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    def __init__(self, base_price: float = 100.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility

    def generate_flash_crash(
        self,
        length: int = 100,
        crash_idx: Optional[int] = None,
        crash_size: float = 0.15,
    ) -> pd.DataFrame:
        """Generate a flash crash scenario

        Args:
            length: Number of periods
            crash_idx: When the crash occurs
            crash_size: Size of the crash as a percentage

        Returns:
            DataFrame with OHLCV data
        """
        # Ensure crash_idx is valid
        if crash_idx is None:
            crash_idx = length // 2
        crash_idx = min(crash_idx, length - 3)  # Ensure room for recovery

        # Generate base prices
        prices = np.random.normal(0, self.volatility, length).cumsum()
        prices = self.base_price * np.exp(prices)

        # Add flash crash
        crash_impact = prices[crash_idx] * crash_size
        prices[crash_idx : crash_idx + 3] -= crash_impact
        prices[crash_idx + 3 :] -= crash_impact * 0.7  # Partial recovery

        # Generate OHLCV data
        df = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    start="2024-01-01", periods=length, freq="1min"
                ),
                "$open": prices,
                "$high": prices * (1 + np.random.uniform(0, 0.01, length)),
                "$low": prices * (1 - np.random.uniform(0, 0.01, length)),
                "$close": prices,
                "$volume": np.random.uniform(100, 1000, length),
            }
        )

        return df

    def generate_low_liquidity(
        self,
        length: int = 100,
        low_liq_start: int = 30,
        low_liq_length: int = 20,
    ) -> pd.DataFrame:
        """Generate a low liquidity scenario

        Args:
            length: Number of periods
            low_liq_start: When low liquidity starts
            low_liq_length: How long low liquidity lasts

        Returns:
            DataFrame with OHLCV data
        """
        # Generate base prices with higher volatility during low liquidity
        prices = np.zeros(length)
        volatility = np.ones(length) * self.volatility
        volatility[low_liq_start : low_liq_start + low_liq_length] *= 3

        for i in range(1, length):
            prices[i] = prices[i - 1] + np.random.normal(0, volatility[i])

        prices = self.base_price * np.exp(prices)

        # Generate volumes with low liquidity period
        volumes = np.random.uniform(100, 1000, length)
        volumes[low_liq_start : low_liq_start + low_liq_length] *= 0.1

        df = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    start="2024-01-01", periods=length, freq="1min"
                ),
                "$open": prices,
                "$high": prices * (1 + np.random.uniform(0, 0.01, length)),
                "$low": prices * (1 - np.random.uniform(0, 0.01, length)),
                "$close": prices,
                "$volume": volumes,
            }
        )

        return df

    def generate_choppy_market(
        self, length: int = 100, chop_intensity: float = 2.0
    ) -> pd.DataFrame:
        """Generate a choppy market scenario with rapid price reversals

        Args:
            length: Number of periods
            chop_intensity: Intensity of the choppy behavior

        Returns:
            DataFrame with OHLCV data
        """
        # Generate oscillating prices with higher amplitude
        t = np.linspace(0, 4 * np.pi, length)
        trend = (
            self.base_price
            + np.sin(t) * self.base_price * 0.1 * chop_intensity
        )
        noise = np.random.normal(
            0, self.volatility * chop_intensity * 2, length
        )
        prices = trend + noise

        df = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    start="2024-01-01", periods=length, freq="1min"
                ),
                "$open": prices,
                "$high": prices * (1 + np.random.uniform(0, 0.01, length)),
                "$low": prices * (1 - np.random.uniform(0, 0.01, length)),
                "$close": prices,
                "$volume": np.random.uniform(100, 1000, length),
            }
        )

        return df

    def combine_scenarios(self, scenarios: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple scenario DataFrames into one"""
        return pd.concat(scenarios, ignore_index=True)


class ScenarioTester:
    """Advanced scenario testing for trading strategies"""

    def __init__(self, env=None, agent=None):
        """Initialize scenario tester

        Args:
            env: Optional TradingEnvironment instance
            agent: Optional trading agent instance
        """
        self.logger = logging.getLogger(__name__)
        self.env = env
        self.agent = agent

    def test_scenario(
        self, data: pd.DataFrame, scenario_type: str = "normal"
    ) -> Dict[str, Any]:
        """Test strategy under specific scenario

        Args:
            data: Market data
            scenario_type: Type of scenario to test

        Returns:
            Dictionary with test results
        """
        # Create environment if not provided
        if self.env is None:
            self.env = TradingEnvironment(
                df=data,
                initial_balance=10000.0,
                trading_fee=0.001,
                window_size=20,
            )

        # Create agent if not provided
        if self.agent is None:
            self.agent = PPOAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                c1=1.0,
                c2=0.01,
                batch_size=64,
                n_epochs=10,
            )

        # Run scenario
        if scenario_type == "flash_crash":
            return self._test_flash_crash()
        elif scenario_type == "low_liquidity":
            return self._test_low_liquidity()
        elif scenario_type == "high_volatility":
            return self._test_high_volatility()
        elif scenario_type == "trend_following":
            return self._test_trend_following()
        elif scenario_type == "mean_reversion":
            return self._test_mean_reversion()
        else:
            return self._test_normal()

    def _test_flash_crash(self) -> Dict[str, Any]:
        """Test flash crash scenario"""
        portfolio_values = []
        trades = []
        timestamps = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            # Convert scalar action to array if needed
            if isinstance(action, (int, float)):
                action = np.array([action])
            next_state, reward, done, truncated, info = self.env.step(action)

            portfolio_values.append(info["portfolio_value"])
            timestamps.append(info["step"])
            if info.get("trade"):
                trades.append(info["trade"])

            if done or truncated:
                break
            state = next_state

        # Calculate metrics
        max_drawdown_idx = np.argmax(
            np.maximum.accumulate(portfolio_values) - portfolio_values
        )

        return {
            "metrics": {
                "final_value": portfolio_values[-1],
                "total_return": (portfolio_values[-1] - portfolio_values[0])
                / portfolio_values[0],
                "max_drawdown": (
                    max(portfolio_values[:max_drawdown_idx])
                    - min(portfolio_values[max_drawdown_idx:])
                )
                / max(portfolio_values[:max_drawdown_idx]),
                "recovery_time": len(portfolio_values) - max_drawdown_idx,
                "survived_crash": portfolio_values[-1]
                > portfolio_values[0] * 0.5,
            },
            "trades": trades,
            "portfolio_values": portfolio_values,
            "timestamps": timestamps,
        }

    def _test_low_liquidity(self) -> Dict[str, Any]:
        """Test low liquidity scenario"""
        portfolio_values = []
        trades = []
        timestamps = []
        slippage = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            # Convert scalar action to array if needed
            if isinstance(action, (int, float)):
                action = np.array([action])
            next_state, reward, done, truncated, info = self.env.step(action)

            portfolio_values.append(info["portfolio_value"])
            timestamps.append(info["step"])
            if info.get("trade"):
                trades.append(info["trade"])
            if "slippage" in info:
                slippage.append(info["slippage"])

            if done or truncated:
                break
            state = next_state

        return {
            "metrics": {
                "final_value": portfolio_values[-1],
                "total_return": (portfolio_values[-1] - portfolio_values[0])
                / portfolio_values[0],
                "avg_slippage": np.mean(slippage) if slippage else 0,
                "max_slippage": np.max(slippage) if slippage else 0,
                "trade_count": len(trades),
                "trade_count_low_liq": len(
                    [
                        t
                        for t in trades
                        if 300 <= timestamps.index(t["entry_time"]) < 400
                    ]
                ),
            },
            "trades": trades,
            "portfolio_values": portfolio_values,
            "timestamps": timestamps,
        }

    def _test_high_volatility(self) -> Dict[str, Any]:
        """Test high volatility scenario"""
        returns = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            # Convert scalar action to array if needed
            if isinstance(action, (int, float)):
                action = np.array([action])
            next_state, reward, done, truncated, info = self.env.step(action)

            returns.append(reward)

            if done or truncated:
                break
            state = next_state

        volatility = np.std(returns)

        return {
            "volatility": volatility,
            "sharpe_ratio": (
                np.mean(returns) / volatility if volatility > 0 else 0
            ),
        }

    def _test_trend_following(self) -> Dict[str, Any]:
        """Test trend following scenario"""
        positions = []
        returns = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)

            positions.append(info["position"])
            returns.append(reward)

            if done or truncated:
                break
            state = next_state

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(positions, returns)

        return {
            "trend_strength": trend_strength,
            "avg_position": np.mean(np.abs(positions)),
        }

    def _test_mean_reversion(self) -> Dict[str, Any]:
        """Test mean reversion scenario"""
        positions = []
        prices = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)

            positions.append(info["position"])
            prices.append(info.get("price", 0))

            if done or truncated:
                break
            state = next_state

        # Calculate mean reversion strength
        mr_strength = self._calculate_mean_reversion_strength(
            positions, prices
        )

        return {
            "mean_reversion_strength": mr_strength,
            "position_changes": len(
                [
                    i
                    for i in range(1, len(positions))
                    if positions[i] != positions[i - 1]
                ]
            ),
        }

    def _test_normal(self) -> Dict[str, Any]:
        """Test normal market conditions"""
        portfolio_values = []
        trades = []
        timestamps = []
        state, _ = self.env.reset()

        for _ in range(1000):
            action = self.agent.get_action(state)
            # Convert scalar action to array if needed
            if isinstance(action, (int, float)):
                action = np.array([action])
            next_state, reward, done, truncated, info = self.env.step(action)

            portfolio_values.append(info["portfolio_value"])
            timestamps.append(info["step"])
            if info.get("trade"):
                trades.append(info["trade"])

            if done or truncated:
                break
            state = next_state

        # Calculate metrics
        returns = pd.Series(portfolio_values).pct_change().dropna()
        volatility = returns.std()
        sharpe = (
            np.sqrt(252) * returns.mean() / volatility if volatility > 0 else 0
        )
        negative_returns = returns[returns < 0]
        sortino = (
            np.sqrt(252) * returns.mean() / negative_returns.std()
            if len(negative_returns) > 0
            else 0
        )

        # Calculate drawdown
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (rolling_max - portfolio_values) / rolling_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Calculate win rate
        winning_trades = sum(
            1
            for t in trades
            if (t["type"] == "buy" and t["price"] < portfolio_values[-1])
            or (t["type"] == "sell" and t["price"] > portfolio_values[-1])
        )
        win_rate = winning_trades / len(trades) if trades else 0

        return {
            "metrics": {
                "final_value": portfolio_values[-1],
                "total_return": (portfolio_values[-1] - portfolio_values[0])
                / portfolio_values[0],
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "total_trades": len(trades),
            },
            "trades": trades,
            "portfolio_values": portfolio_values,
            "timestamps": timestamps,
        }

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from list of values"""
        peak = values[0]
        max_drawdown = 0

        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_trend_strength(
        self, positions: List[int], returns: List[float]
    ) -> float:
        """Calculate trend following strength"""
        if len(positions) < 2 or len(returns) < 2:
            return 0

        # Calculate position-return correlation
        correlation = np.corrcoef(positions[:-1], returns[1:])[0, 1]
        return abs(correlation)

    def _calculate_mean_reversion_strength(
        self, positions: List[int], prices: List[float]
    ) -> float:
        """Calculate mean reversion strength"""
        if len(positions) < 2 or len(prices) < 2:
            return 0

        # Calculate position-price correlation
        correlation = np.corrcoef(positions, prices)[0, 1]
        return -correlation  # Negative correlation indicates mean reversion
