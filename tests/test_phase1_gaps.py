"""Tests for Phase 1: Critical Gap Fixes.

Tests SAC/TD3 agents, Walk-Forward validation, Regime Detection,
and VAE OOD Detection.
"""

import numpy as np
import pandas as pd
import pytest
import gymnasium as gym


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_space():
    """Standard 2D observation space (window=20, features=5)."""
    return gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(20, 5), dtype=np.float32
    )


@pytest.fixture
def act_space():
    """Continuous action space [-1, 1]."""
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)


@pytest.fixture
def flat_obs_space():
    """Flat 1D observation space for SB3 compatibility."""
    return gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32
    )


@pytest.fixture
def sample_price_df():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    prices = np.maximum(prices, 1.0)  # ensure positive

    df = pd.DataFrame({
        "$open": prices + np.random.randn(n) * 0.1,
        "$high": prices + abs(np.random.randn(n) * 0.5),
        "$low": prices - abs(np.random.randn(n) * 0.5),
        "$close": prices,
        "$volume": np.random.randint(1000, 100000, n).astype(float),
    })
    return df


# ---------------------------------------------------------------------------
# SAC Agent Tests
# ---------------------------------------------------------------------------

class TestSACAgent:
    def test_import(self):
        from agents.strategies.single.sac_agent import SACAgent
        assert SACAgent is not None

    def test_init(self, obs_space, act_space):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        assert agent.observation_space == obs_space
        assert agent.action_space == act_space

    def test_get_action_shape(self, obs_space, act_space):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        obs = obs_space.sample()
        action = agent.get_action(obs)
        assert action.shape == act_space.shape

    def test_get_action_deterministic(self, obs_space, act_space):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        obs = obs_space.sample()
        a1 = agent.get_action(obs, deterministic=True)
        a2 = agent.get_action(obs, deterministic=True)
        np.testing.assert_array_equal(a1, a2)

    def test_action_in_bounds(self, obs_space, act_space):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        for _ in range(10):
            obs = obs_space.sample()
            action = agent.get_action(obs)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_train_step(self, obs_space, act_space):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        obs = obs_space.sample()
        action = act_space.sample()
        next_obs = obs_space.sample()
        metrics = agent.train_step(obs, action, 1.0, next_obs, False)
        assert isinstance(metrics, dict)

    def test_factory_creates_sac(self, obs_space, act_space):
        from agents.strategies.agent_factory import create_agent
        agent = create_agent("sac", config={}, observation_space=obs_space, action_space=act_space)
        assert "SAC" in type(agent).__name__ or "Dummy" in type(agent).__name__

    def test_save_load(self, obs_space, act_space, tmp_path):
        from agents.strategies.single.sac_agent import SACAgent
        agent = SACAgent(obs_space, act_space, learning_starts=10)
        _ = agent.get_action(obs_space.sample())  # ensure model init
        path = str(tmp_path / "sac_model")
        agent.save(path)
        agent2 = SACAgent(obs_space, act_space, learning_starts=10)
        agent2.load(path)
        obs = obs_space.sample()
        a1 = agent.get_action(obs, deterministic=True)
        a2 = agent2.get_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


# ---------------------------------------------------------------------------
# TD3 Agent Tests
# ---------------------------------------------------------------------------

class TestTD3Agent:
    def test_import(self):
        from agents.strategies.single.td3_agent import TD3Agent
        assert TD3Agent is not None

    def test_init(self, obs_space, act_space):
        from agents.strategies.single.td3_agent import TD3Agent
        agent = TD3Agent(obs_space, act_space, learning_starts=10)
        assert agent.observation_space == obs_space

    def test_get_action_shape(self, obs_space, act_space):
        from agents.strategies.single.td3_agent import TD3Agent
        agent = TD3Agent(obs_space, act_space, learning_starts=10)
        obs = obs_space.sample()
        action = agent.get_action(obs)
        assert action.shape == act_space.shape

    def test_action_in_bounds(self, obs_space, act_space):
        from agents.strategies.single.td3_agent import TD3Agent
        agent = TD3Agent(obs_space, act_space, learning_starts=10)
        for _ in range(10):
            obs = obs_space.sample()
            action = agent.get_action(obs)
            assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_train_step(self, obs_space, act_space):
        from agents.strategies.single.td3_agent import TD3Agent
        agent = TD3Agent(obs_space, act_space, learning_starts=10)
        obs = obs_space.sample()
        action = act_space.sample()
        next_obs = obs_space.sample()
        metrics = agent.train_step(obs, action, 0.5, next_obs, False)
        assert isinstance(metrics, dict)

    def test_factory_creates_td3(self, obs_space, act_space):
        from agents.strategies.agent_factory import create_agent
        agent = create_agent("td3", config={}, observation_space=obs_space, action_space=act_space)
        assert "TD3" in type(agent).__name__ or "Dummy" in type(agent).__name__

    def test_save_load(self, obs_space, act_space, tmp_path):
        from agents.strategies.single.td3_agent import TD3Agent
        agent = TD3Agent(obs_space, act_space, learning_starts=10)
        _ = agent.get_action(obs_space.sample())
        path = str(tmp_path / "td3_model")
        agent.save(path)
        agent2 = TD3Agent(obs_space, act_space, learning_starts=10)
        agent2.load(path)
        obs = obs_space.sample()
        a1 = agent.get_action(obs, deterministic=True)
        a2 = agent2.get_action(obs, deterministic=True)
        np.testing.assert_allclose(a1, a2, atol=1e-5)


# ---------------------------------------------------------------------------
# Walk-Forward Validation Tests
# ---------------------------------------------------------------------------

class TestWalkForwardValidator:
    def test_import(self):
        from training.validation.walk_forward import WalkForwardValidator
        assert WalkForwardValidator is not None

    def test_split_creates_folds(self, sample_price_df):
        from training.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_splits=5, train_ratio=0.3, gap_days=2)
        splits = validator.split(sample_price_df)
        assert len(splits) == 5
        for train_df, test_df in splits:
            assert len(train_df) > 0
            assert len(test_df) > 0

    def test_split_no_future_leak(self, sample_price_df):
        from training.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_splits=5, train_ratio=0.3, gap_days=5)
        splits = validator.split(sample_price_df)
        for train_df, test_df in splits:
            train_end_idx = train_df.index[-1]
            test_start_idx = test_df.index[0]
            assert test_start_idx > train_end_idx + 4  # gap_days=5

    def test_expanding_window_grows(self, sample_price_df):
        from training.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_splits=5, train_ratio=0.3, gap_days=2, mode="expanding")
        splits = validator.split(sample_price_df)
        train_sizes = [len(t) for t, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_result_metrics(self, sample_price_df):
        from training.validation.walk_forward import WalkForwardResult, FoldResult
        folds = [
            FoldResult(fold_idx=0, train_size=100, test_size=50, is_sharpe=1.5, oos_sharpe=0.8),
            FoldResult(fold_idx=1, train_size=150, test_size=50, is_sharpe=1.2, oos_sharpe=1.0),
        ]
        result = WalkForwardResult(folds=folds)
        assert result.oos_sharpe == pytest.approx(0.9, abs=0.01)
        assert result.is_sharpe == pytest.approx(1.35, abs=0.01)
        assert result.stability_ratio == pytest.approx(0.9 / 1.35, abs=0.01)

    def test_too_few_rows_raises(self):
        from training.validation.walk_forward import WalkForwardValidator
        validator = WalkForwardValidator(n_splits=10, train_ratio=0.9, gap_days=50)
        tiny_df = pd.DataFrame({"$close": [1, 2, 3], "$volume": [100, 200, 300]})
        with pytest.raises(ValueError):
            validator.split(tiny_df)

    def test_sharpe_computation(self):
        from training.validation.walk_forward import WalkForwardValidator
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        sharpe = WalkForwardValidator._compute_sharpe(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # positive returns → positive Sharpe


# ---------------------------------------------------------------------------
# Regime Detection Tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def test_import(self):
        from training.signals.regime_detector import RegimeDetector
        assert RegimeDetector is not None

    def test_fit_predict(self, sample_price_df):
        from training.signals.regime_detector import RegimeDetector
        detector = RegimeDetector(n_regimes=3, lookback=30)
        detector.fit(sample_price_df)
        assert detector.is_fitted

        probs = detector.predict_proba(sample_price_df.tail(60))
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 0.01

    def test_predict_returns_valid_regime(self, sample_price_df):
        from training.signals.regime_detector import RegimeDetector
        detector = RegimeDetector(n_regimes=3)
        detector.fit(sample_price_df)
        regime = detector.predict(sample_price_df.tail(60))
        assert regime in [0, 1, 2]

    def test_unfitted_returns_uniform(self, sample_price_df):
        from training.signals.regime_detector import RegimeDetector
        detector = RegimeDetector(n_regimes=3)
        probs = detector.predict_proba(sample_price_df.tail(60))
        expected = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        np.testing.assert_allclose(probs, expected, atol=0.01)

    def test_save_load(self, sample_price_df, tmp_path):
        from training.signals.regime_detector import RegimeDetector
        detector = RegimeDetector(n_regimes=3)
        detector.fit(sample_price_df)
        path = str(tmp_path / "regime.pkl")
        detector.save(path)

        loaded = RegimeDetector.load(path)
        assert loaded.is_fitted

        probs1 = detector.predict_proba(sample_price_df.tail(60))
        probs2 = loaded.predict_proba(sample_price_df.tail(60))
        np.testing.assert_allclose(probs1, probs2, atol=1e-5)

    def test_regime_labels(self):
        from training.signals.regime_detector import RegimeDetector
        detector = RegimeDetector()
        assert detector.get_regime_label(0) == "bear"
        assert detector.get_regime_label(1) == "sideways"
        assert detector.get_regime_label(2) == "bull"

    def test_no_volume_column(self):
        from training.signals.regime_detector import RegimeDetector
        df = pd.DataFrame({"$close": np.cumsum(np.random.randn(200)) + 100})
        df["$close"] = df["$close"].clip(lower=1.0)
        detector = RegimeDetector(n_regimes=2)
        detector.fit(df)
        probs = detector.predict_proba(df.tail(60))
        assert probs.shape == (2,)


# ---------------------------------------------------------------------------
# VAE OOD Detection Tests
# ---------------------------------------------------------------------------

class TestVAEOODDetector:
    def test_import(self):
        from agents.risk.ood_detector import VAEOODDetector
        assert VAEOODDetector is not None

    def test_fit(self):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 50
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=8, hidden_dim=32)
        data = np.random.randn(200, obs_dim).astype(np.float32)
        detector.fit(data, epochs=5, batch_size=64)
        assert detector.is_fitted

    def test_in_distribution_low_signal(self):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 50
        np.random.seed(42)
        data = np.random.randn(500, obs_dim).astype(np.float32)
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=8, hidden_dim=32)
        detector.fit(data, epochs=20, batch_size=64)

        # In-distribution sample should have low abstain signal
        in_dist = data[0]
        signal = detector.get_abstain_signal(in_dist)
        assert 0.0 <= signal <= 1.0
        # Most training samples should be in-distribution
        signals = [detector.get_abstain_signal(data[i]) for i in range(50)]
        assert np.mean(signals) < 0.5

    def test_ood_high_signal(self):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 50
        np.random.seed(42)
        data = np.random.randn(500, obs_dim).astype(np.float32)
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=8, hidden_dim=32)
        detector.fit(data, epochs=20, batch_size=64)

        # Way OOD observation
        ood_obs = np.ones(obs_dim, dtype=np.float32) * 100.0
        signal = detector.get_abstain_signal(ood_obs)
        assert signal > 0.5  # should be flagged as OOD

    def test_is_ood_tuple(self):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 20
        data = np.random.randn(100, obs_dim).astype(np.float32)
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=4, hidden_dim=16)
        detector.fit(data, epochs=5)
        is_ood, error = detector.is_ood(data[0])
        assert isinstance(is_ood, bool)
        assert isinstance(error, float)

    def test_unfitted_returns_zero(self):
        from agents.risk.ood_detector import VAEOODDetector
        detector = VAEOODDetector(obs_dim=10)
        signal = detector.get_abstain_signal(np.zeros(10))
        assert signal == 0.0

    def test_2d_obs_flattening(self):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 20 * 5  # window=20, features=5
        data_3d = np.random.randn(100, 20, 5).astype(np.float32)
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=8, hidden_dim=32)
        detector.fit(data_3d, epochs=5)
        assert detector.is_fitted

        # Test with 2D input
        obs_2d = np.random.randn(20, 5).astype(np.float32)
        signal = detector.get_abstain_signal(obs_2d)
        assert 0.0 <= signal <= 1.0

    def test_save_load(self, tmp_path):
        from agents.risk.ood_detector import VAEOODDetector
        obs_dim = 20
        data = np.random.randn(100, obs_dim).astype(np.float32)
        detector = VAEOODDetector(obs_dim=obs_dim, latent_dim=4, hidden_dim=16)
        detector.fit(data, epochs=5)

        path = str(tmp_path / "ood.pt")
        detector.save(path)
        loaded = VAEOODDetector.load(path)
        assert loaded.is_fitted

        obs = data[0]
        s1 = detector.get_abstain_signal(obs)
        s2 = loaded.get_abstain_signal(obs)
        assert abs(s1 - s2) < 1e-5


# ---------------------------------------------------------------------------
# Integration: Agent Factory creates real SAC/TD3
# ---------------------------------------------------------------------------

class TestAgentFactoryIntegration:
    def test_sac_not_dummy(self, obs_space, act_space):
        """SAC factory should return real SACAgent, not DummyAgent."""
        from agents.strategies.agent_factory import create_agent
        agent = create_agent("sac", config={}, observation_space=obs_space, action_space=act_space)
        assert type(agent).__name__ == "SACAgent"

    def test_td3_not_dummy(self, obs_space, act_space):
        """TD3 factory should return real TD3Agent, not DummyAgent."""
        from agents.strategies.agent_factory import create_agent
        agent = create_agent("td3", config={}, observation_space=obs_space, action_space=act_space)
        assert type(agent).__name__ == "TD3Agent"

    def test_all_ensemble_agents_creatable(self, obs_space, act_space):
        """All 3 RL agents (PPO, SAC, TD3) should be creatable."""
        from agents.strategies.agent_factory import create_agent
        for agent_type in ["ppo", "sac", "td3"]:
            agent = create_agent(agent_type, config={}, observation_space=obs_space, action_space=act_space)
            assert agent is not None
            action = agent.get_action(obs_space.sample())
            assert action.shape == act_space.shape
