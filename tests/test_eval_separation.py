"""C3: verify eval environment uses separate data."""
import pytest
import numpy as np
import pandas as pd


class TestEvalDataSeparation:
    def test_train_single_agent_accepts_eval_data(self):
        """train_single_agent should accept eval_data parameter."""
        import inspect
        from training.train_pipeline import train_single_agent
        sig = inspect.signature(train_single_agent)
        assert "eval_data" in sig.parameters, (
            "train_single_agent missing eval_data parameter"
        )

    def test_eval_env_uses_different_data(self):
        """create_eval_env should create env with provided data, not training data."""
        from training.env_factory import create_eval_env, create_env
        # Minimal config
        config = {
            "env": {"type": "single_asset_rl", "window_size": 5, "initial_capital": 10000},
        }
        n = 50
        train_data = pd.DataFrame({
            "$open": [100.0]*n, "$high": [101.0]*n,
            "$low": [99.0]*n, "$close": [100.0]*n,
            "$volume": [1000.0]*n,
        })
        eval_data = pd.DataFrame({
            "$open": [200.0]*n, "$high": [201.0]*n,
            "$low": [199.0]*n, "$close": [200.0]*n,
            "$volume": [2000.0]*n,
        })
        train_env = create_env(config, train_data, validate=False)
        eval_env = create_eval_env(config, eval_data)
        # Verify they use different data
        assert train_env.data["$close"].iloc[0] != eval_env.data["$close"].iloc[0]
