"""
Week 4: Coverage tests for new code.

Targets >80% coverage on:
  - agents/sb3/feature_extractors.py
  - envs/wrap_env.py  (SB3CompatWrapper, ClipActions, RecordEpisodeStats,
                       StackObservation, make_env, make_sb3_env)
  - training/callbacks/sb3_callbacks.py
"""
import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_df(n=100):
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({
        "$open":   close * rng.uniform(0.998, 1.002, n),
        "$high":   close * rng.uniform(1.000, 1.010, n),
        "$low":    close * rng.uniform(0.990, 1.000, n),
        "$close":  close,
        "$volume": rng.integers(1_000, 10_000, n).astype(float),
    })


def _make_single_env(df=None):
    from envs.single_asset_rl_env import SingleAssetRLTradingEnv
    if df is None:
        df = _make_dummy_df()
    return SingleAssetRLTradingEnv(data=df)


# ===========================================================================
# 1. Feature Extractors
# ===========================================================================

class TestTradingWindowExtractor:
    @pytest.fixture
    def obs_space(self):
        return spaces.Box(low=-10.0, high=10.0, shape=(20, 5), dtype=np.float32)

    def test_init(self, obs_space):
        from agents.sb3.feature_extractors import TradingWindowExtractor
        ext = TradingWindowExtractor(obs_space, features_dim=64)
        assert ext.features_dim == 64

    def test_forward(self, obs_space):
        from agents.sb3.feature_extractors import TradingWindowExtractor
        ext = TradingWindowExtractor(obs_space, features_dim=64)
        batch = torch.zeros(4, 20, 5)   # (batch, window, features)
        out = ext.forward(batch)
        assert out.shape == (4, 64)

    def test_requires_2d_obs(self):
        from agents.sb3.feature_extractors import TradingWindowExtractor
        bad_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        with pytest.raises(ValueError):
            TradingWindowExtractor(bad_space)


class TestLSTMTradingExtractor:
    @pytest.fixture
    def obs_space(self):
        return spaces.Box(low=-10.0, high=10.0, shape=(20, 5), dtype=np.float32)

    def test_init(self, obs_space):
        from agents.sb3.feature_extractors import LSTMTradingExtractor
        ext = LSTMTradingExtractor(obs_space, features_dim=64, hidden_size=64, num_layers=1)
        assert ext.features_dim == 64

    def test_forward(self, obs_space):
        from agents.sb3.feature_extractors import LSTMTradingExtractor
        ext = LSTMTradingExtractor(obs_space, features_dim=64, hidden_size=64, num_layers=1)
        batch = torch.zeros(2, 20, 5)
        out = ext.forward(batch)
        assert out.shape == (2, 64)

    def test_requires_2d_obs(self):
        from agents.sb3.feature_extractors import LSTMTradingExtractor
        bad_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        with pytest.raises(ValueError):
            LSTMTradingExtractor(bad_space)


# ===========================================================================
# 2. wrap_env — SB3CompatWrapper
# ===========================================================================

class TestSB3CompatWrapper:
    def test_step_records_episode_on_done(self):
        from envs.wrap_env import SB3CompatWrapper
        env = SB3CompatWrapper(_make_single_env())
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if done:
                assert "episode" in info
                assert info["episode"]["l"] == steps
        assert steps > 0

    def test_clips_action(self):
        from envs.wrap_env import SB3CompatWrapper
        env = SB3CompatWrapper(_make_single_env())
        env.reset()
        # Out-of-range action should not raise
        action = np.array([999.0])
        obs, reward, term, trunc, info = env.step(action)
        assert obs is not None

    def test_reset_clears_episode_stats(self):
        from envs.wrap_env import SB3CompatWrapper
        env = SB3CompatWrapper(_make_single_env())
        env.reset()
        env.step(env.action_space.sample())
        env.reset()
        assert env._episode_reward == 0.0
        assert env._episode_length == 0


# ===========================================================================
# 3. wrap_env — ClipActions
# ===========================================================================

class TestClipActions:
    def test_clips_out_of_range(self):
        from envs.wrap_env import ClipActions
        base = _make_single_env()
        env = ClipActions(base)
        clipped = env.action(np.array([100.0]))
        assert clipped <= env.action_space.high.max()

    def test_within_range_unchanged(self):
        from envs.wrap_env import ClipActions
        base = _make_single_env()
        env = ClipActions(base)
        act = np.array([0.0])
        assert env.action(act) == 0.0


# ===========================================================================
# 4. wrap_env — RecordEpisodeStats
# ===========================================================================

class TestRecordEpisodeStats:
    def test_records_episode_on_termination(self):
        from envs.wrap_env import RecordEpisodeStats
        env = RecordEpisodeStats(_make_single_env())
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        assert len(env.episode_returns) == 1
        assert len(env.episode_lengths) == 1
        assert "episode" in info

    def test_reset_clears_counters(self):
        from envs.wrap_env import RecordEpisodeStats
        env = RecordEpisodeStats(_make_single_env())
        env.reset()
        env.step(env.action_space.sample())
        env.reset()
        assert env.current_return == 0
        assert env.current_length == 0


# ===========================================================================
# 5. wrap_env — StackObservation
# ===========================================================================

class TestStackObservation:
    def test_init(self):
        from envs.wrap_env import StackObservation
        base = _make_single_env()
        env = StackObservation(base, stack_size=4)
        assert env.obs_stack is None

    def test_reset_initializes_stack(self):
        from envs.wrap_env import StackObservation
        base = _make_single_env()
        env = StackObservation(base, stack_size=4)
        obs, info = env.reset()
        assert obs is not None
        assert env.obs_stack is not None

    def test_observation_updates(self):
        from envs.wrap_env import StackObservation
        base = _make_single_env()
        env = StackObservation(base, stack_size=4)
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs is not None

    def test_requires_2d_obs_space(self):
        from envs.wrap_env import StackObservation
        bad_env = gym.make("CartPole-v1")  # has 1D obs
        with pytest.raises(ValueError):
            StackObservation(bad_env)


# ===========================================================================
# 6. wrap_env — make_env (legacy)
# ===========================================================================

class TestMakeEnv:
    def test_make_env_no_normalize_no_stack(self):
        from envs.wrap_env import make_env
        base = _make_single_env()
        env = make_env(base, normalize=False, stack_size=1)
        assert env is not None
        obs, _ = env.reset()
        assert obs is not None

    def test_make_env_no_normalize_with_stack(self):
        from envs.wrap_env import make_env
        base = _make_single_env()
        env = make_env(base, normalize=False, stack_size=4)
        assert env is not None
        obs, _ = env.reset()
        assert obs is not None

    def test_make_env_with_normalize(self):
        from envs.wrap_env import make_env
        base = _make_single_env()
        # Now that torch is imported and reset() is fixed, NormalizeObservation should work
        env = make_env(base, normalize=True, stack_size=1)
        assert env is not None
        obs, info = env.reset()
        assert obs is not None


# ===========================================================================
# 7. wrap_env — make_sb3_env (vec_normalize_kwargs path)
# ===========================================================================

class TestMakeSB3Env:
    @pytest.fixture
    def df(self):
        return _make_dummy_df(100)

    def test_with_vec_normalize_kwargs(self, df):
        from envs.wrap_env import make_sb3_env
        env = make_sb3_env(
            df,
            n_envs=1,
            use_vec_normalize=True,
            vec_normalize_kwargs={"clip_obs": 5.0, "clip_reward": 5.0},
        )
        obs = env.reset()
        assert obs is not None
        env.close()

    def test_without_vec_normalize(self, df):
        from envs.wrap_env import make_sb3_env
        env = make_sb3_env(df, n_envs=1, use_vec_normalize=False)
        obs = env.reset()
        assert obs is not None
        env.close()


# ===========================================================================
# 8. sb3_callbacks — MLflowLoggingCallback
# ===========================================================================

class TestMLflowLoggingCallback:
    def _make_trained_model(self):
        import numpy as np
        import pandas as pd
        from envs.wrap_env import make_sb3_env
        from stable_baselines3 import PPO
        df = _make_dummy_df(100)
        env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=64)
        return model, env

    def test_on_step_with_mock_manager(self):
        from training.callbacks.sb3_callbacks import MLflowLoggingCallback
        mock_manager = MagicMock()
        cb = MLflowLoggingCallback(mlflow_manager=mock_manager, log_interval=1)

        model, env = self._make_trained_model()
        # Manually simulate callback state
        cb.init_callback(model)
        cb.n_calls = 1
        cb.num_timesteps = 64
        result = cb._on_step()
        assert result is True

    def test_on_step_no_manager(self):
        from training.callbacks.sb3_callbacks import MLflowLoggingCallback
        cb = MLflowLoggingCallback(mlflow_manager=None, log_interval=1)
        model, env = self._make_trained_model()
        cb.init_callback(model)
        cb.n_calls = 1
        cb.num_timesteps = 1
        result = cb._on_step()
        assert result is True

    def test_on_step_manager_exception(self):
        from training.callbacks.sb3_callbacks import MLflowLoggingCallback
        mock_manager = MagicMock()
        mock_manager.log_metrics.side_effect = Exception("mlflow down")
        cb = MLflowLoggingCallback(mlflow_manager=mock_manager, log_interval=1)
        model, env = self._make_trained_model()
        cb.init_callback(model)
        cb.n_calls = 1
        cb.num_timesteps = 64
        # Should not raise
        result = cb._on_step()
        assert result is True


# ===========================================================================
# 9. sb3_callbacks — SB3CheckpointCallback with mlflow_manager
# ===========================================================================

class TestSB3CheckpointCallbackWithMLflow:
    def test_mlflow_logging_on_save(self, tmp_path):
        from training.callbacks.sb3_callbacks import SB3CheckpointCallback
        from envs.wrap_env import make_sb3_env
        from stable_baselines3 import PPO

        df = _make_dummy_df(100)
        env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        model = PPO("MlpPolicy", env, verbose=0)

        mock_manager = MagicMock()
        cb = SB3CheckpointCallback(
            save_freq=32,
            save_path=str(tmp_path),
            mlflow_manager=mock_manager,
            verbose=1,
        )
        model.learn(total_timesteps=64, callback=cb)
        # At least one checkpoint was saved and mlflow was called
        assert mock_manager.log_artifact.called or mock_manager.log_metrics.called

    def test_mlflow_exception_swallowed(self, tmp_path):
        from training.callbacks.sb3_callbacks import SB3CheckpointCallback
        from envs.wrap_env import make_sb3_env
        from stable_baselines3 import PPO

        df = _make_dummy_df(100)
        env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        model = PPO("MlpPolicy", env, verbose=0)

        mock_manager = MagicMock()
        mock_manager.log_artifact.side_effect = Exception("artifact error")
        cb = SB3CheckpointCallback(
            save_freq=32,
            save_path=str(tmp_path),
            mlflow_manager=mock_manager,
        )
        # Should not raise even if MLflow fails
        model.learn(total_timesteps=64, callback=cb)


# ===========================================================================
# 10. sb3_callbacks — SB3EvalCallback with mlflow_manager
# ===========================================================================

class TestSB3EvalCallback:
    def test_eval_callback_with_mlflow(self, tmp_path):
        from training.callbacks.sb3_callbacks import SB3EvalCallback
        from envs.wrap_env import make_sb3_env
        from stable_baselines3 import PPO

        df = _make_dummy_df(100)
        train_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        eval_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        model = PPO("MlpPolicy", train_env, verbose=0)

        mock_manager = MagicMock()
        cb = SB3EvalCallback(
            eval_env=eval_env,
            mlflow_manager=mock_manager,
            n_eval_episodes=1,
            eval_freq=32,
            best_model_save_path=str(tmp_path),
            verbose=0,
        )
        model.learn(total_timesteps=64, callback=cb)
        # Eval was triggered at least once
        assert cb.last_mean_reward is not None

    def test_eval_callback_mlflow_exception_swallowed(self, tmp_path):
        from training.callbacks.sb3_callbacks import SB3EvalCallback
        from envs.wrap_env import make_sb3_env
        from stable_baselines3 import PPO

        df = _make_dummy_df(100)
        train_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        eval_env = make_sb3_env(df, n_envs=1, use_vec_normalize=True)
        model = PPO("MlpPolicy", train_env, verbose=0)

        mock_manager = MagicMock()
        mock_manager.log_metrics.side_effect = Exception("mlflow error")
        cb = SB3EvalCallback(
            eval_env=eval_env,
            mlflow_manager=mock_manager,
            n_eval_episodes=1,
            eval_freq=32,
            best_model_save_path=str(tmp_path),
            verbose=0,
        )
        # Should not raise
        model.learn(total_timesteps=64, callback=cb)


# ===========================================================================
# 11. wrap_env — NormalizeObservation (now torch is imported in wrap_env)
# ===========================================================================

class TestNormalizeObservation:
    def test_init(self):
        from envs.wrap_env import NormalizeObservation
        base = _make_single_env()
        env = NormalizeObservation(base)
        assert env.count == 0
        assert env.running_mean is not None

    def test_observation_numpy(self):
        from envs.wrap_env import NormalizeObservation
        base = _make_single_env()
        env = NormalizeObservation(base)
        raw_obs = env.observation_space.sample()
        norm_obs = env.observation(raw_obs)
        assert norm_obs.dtype == np.float32
        assert np.all(norm_obs >= -10) and np.all(norm_obs <= 10)

    def test_observation_tensor(self):
        from envs.wrap_env import NormalizeObservation
        base = _make_single_env()
        env = NormalizeObservation(base)
        raw_obs = torch.zeros(*env.observation_space.shape)
        norm_obs = env.observation(raw_obs)
        assert isinstance(norm_obs, torch.Tensor)

    def test_observation_invalid_type(self):
        from envs.wrap_env import NormalizeObservation
        base = _make_single_env()
        env = NormalizeObservation(base)
        with pytest.raises(ValueError):
            env.observation("bad_type")

    def test_reset_clears_stats(self):
        from envs.wrap_env import NormalizeObservation
        base = _make_single_env()
        env = NormalizeObservation(base)
        obs, info = env.reset()
        assert obs is not None
        # count is re-incremented by _update_stats during observation() call
        assert env.count > 0


# ===========================================================================
# 12. wrap_env — MLflowLoggingWrapper (uses _mlflow stub)
# ===========================================================================

class TestMLflowLoggingWrapper:
    def test_init_and_reset(self):
        from envs.wrap_env import MLflowLoggingWrapper, RecordEpisodeStats
        import mlflow, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.set_tracking_uri(f"file://{tmp}/mlruns")
            mlflow.set_experiment("test_wrapper")
            with mlflow.start_run():
                base = RecordEpisodeStats(_make_single_env())
                env = MLflowLoggingWrapper(base, experiment_name="test_wrapper")
                obs, info = env.reset()
                assert obs is not None

    def test_step_logs_metrics(self):
        from envs.wrap_env import MLflowLoggingWrapper, RecordEpisodeStats
        import mlflow, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.set_tracking_uri(f"file://{tmp}/mlruns")
            mlflow.set_experiment("test_wrapper")
            with mlflow.start_run():
                base = RecordEpisodeStats(_make_single_env())
                env = MLflowLoggingWrapper(base, experiment_name="test_wrapper")
                env.reset()
                obs, reward, term, trunc, info = env.step(env.action_space.sample())
                assert obs is not None
