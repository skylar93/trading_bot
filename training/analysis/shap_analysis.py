"""
SHAP-based feature importance analysis for RL trading policies.

Provides a model-agnostic explainability layer that works with:
  - Any callable (lambda obs: values)
  - SB3 policies (PPO, SAC, TD3, etc.) via predict()
  - PyTorch neural networks via DeepExplainer (optional)

Core class: SHAPAnalyzer
  - Uses KernelExplainer for model-agnostic analysis (any policy)
  - Uses DeepExplainer for PyTorch models (faster, gradient-based)
  - Returns SHAPResult with shap_values, expected_value, feature_names
  - Methods: ranking(), plot_bar(), plot_summary(), plot_waterfall()

Week 25 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result dataclass
# ------------------------------------------------------------------

@dataclass
class SHAPResult:
    """Container for SHAP analysis output.

    Attributes
    ----------
    shap_values:
        Array of shape (n_samples, n_features) with SHAP values.
        Each value represents the contribution of that feature to
        the model output for that sample.
    expected_value:
        The base value (model output when all features are at their
        average). Scalar or array depending on output dimension.
    feature_names:
        List of feature names corresponding to columns.
    explainer_type:
        "kernel", "deep", or "tree" — which explainer was used.
    data:
        The input data used for explanation (n_samples, n_features).
    """
    shap_values: np.ndarray
    expected_value: Union[float, np.ndarray]
    feature_names: List[str]
    explainer_type: str
    data: np.ndarray

    def ranking(self) -> List[Tuple[str, float]]:
        """Return features sorted by mean |SHAP value| descending.

        Returns
        -------
        List of (feature_name, mean_abs_shap) tuples, most important first.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        indices = np.argsort(mean_abs)[::-1]
        return [(self.feature_names[i], float(mean_abs[i])) for i in indices]

    def importance_dict(self) -> Dict[str, float]:
        """Return {feature_name: mean_abs_shap} dictionary."""
        return dict(self.ranking())

    def top_k(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return the top-k most important features."""
        return self.ranking()[:k]

    def plot_bar(self, max_display: int = 16, show: bool = True) -> None:
        """Bar plot of mean |SHAP| per feature (requires matplotlib + shap)."""
        if not _SHAP_AVAILABLE:  # pragma: no cover
            raise ImportError("shap is required for plotting. pip install shap")
        exp = shap.Explanation(
            values=self.shap_values,
            base_values=self.expected_value,
            data=self.data,
            feature_names=self.feature_names,
        )
        shap.plots.bar(exp, max_display=max_display, show=show)

    def plot_summary(self, max_display: int = 16, show: bool = True) -> None:
        """Beeswarm summary plot (requires matplotlib + shap)."""
        if not _SHAP_AVAILABLE:  # pragma: no cover
            raise ImportError("shap is required for plotting. pip install shap")
        shap.summary_plot(
            self.shap_values,
            features=self.data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=show,
        )

    def plot_waterfall(self, sample_idx: int = 0, show: bool = True) -> None:
        """Waterfall plot for a single sample (requires matplotlib + shap)."""
        if not _SHAP_AVAILABLE:  # pragma: no cover
            raise ImportError("shap is required for plotting. pip install shap")
        exp = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=self.data[sample_idx],
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(exp, show=show)


# ------------------------------------------------------------------
# Main analyser
# ------------------------------------------------------------------

@dataclass
class SHAPConfig:
    """Configuration for SHAPAnalyzer.

    Parameters
    ----------
    explainer_type:
        "kernel" (model-agnostic, default) or "deep" (PyTorch only).
    n_background:
        Number of background samples for KernelExplainer (more = slower but
        more accurate). For DeepExplainer this is used as batch size.
    n_explain:
        Maximum number of test samples to explain per call to .explain().
        None = explain all provided samples.
    link:
        Link function for KernelExplainer ("identity" or "logit").
    seed:
        Random seed for reproducibility.
    """
    explainer_type: str = "kernel"   # "kernel" | "deep"
    n_background: int = 100
    n_explain: Optional[int] = 200
    link: str = "identity"
    seed: int = 42


class SHAPAnalyzer:
    """Model-agnostic SHAP feature importance for RL trading policies.

    Supports any callable that maps (n_samples, n_features) → (n_samples,)
    or SB3 policies via the ``from_sb3_policy`` factory.

    Examples
    --------
    **Kernel explainer (any model):**

    >>> analyzer = SHAPAnalyzer(
    ...     model_fn=lambda obs: policy.predict(obs)[0].flatten(),
    ...     feature_names=["rsi", "macd", "bb_width", ...],
    ...     background_data=X_train[:100],
    ... )
    >>> result = analyzer.explain(X_test[:50])
    >>> print(result.top_k(5))

    **From SB3 policy:**

    >>> analyzer = SHAPAnalyzer.from_sb3_policy(
    ...     policy=model.policy,
    ...     feature_names=feature_names,
    ...     background_data=X_train[:100],
    ... )
    >>> result = analyzer.explain(X_test[:50])
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        background_data: np.ndarray,
        config: Optional[SHAPConfig] = None,
        torch_model: Optional[Any] = None,  # nn.Module for DeepExplainer
    ):
        if not _SHAP_AVAILABLE:  # pragma: no cover
            raise ImportError("Install shap: pip install shap")

        self.model_fn = model_fn
        self.feature_names = list(feature_names)
        self.background_data = np.asarray(background_data, dtype=np.float32)
        self.config = config or SHAPConfig()
        self.torch_model = torch_model
        self._explainer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_sb3_policy(
        cls,
        policy: Any,
        feature_names: List[str],
        background_data: np.ndarray,
        config: Optional[SHAPConfig] = None,
        use_value_fn: bool = True,
    ) -> "SHAPAnalyzer":
        """Build from a Stable-Baselines3 policy object.

        Parameters
        ----------
        policy:
            An SB3 policy (e.g. model.policy from PPO/SAC/TD3).
        feature_names:
            Feature names for the flat observation vector.
        background_data:
            Background samples, shape (n_bg, n_obs_features).
        config:
            Optional SHAPConfig.
        use_value_fn:
            If True, explain the value function output (scalar per sample).
            If False, explain the action mean output.
        """
        def _policy_fn(obs: np.ndarray) -> np.ndarray:
            obs_t = obs.astype(np.float32)
            results = []
            for i in range(len(obs_t)):
                row = obs_t[i : i + 1]
                try:
                    if use_value_fn and hasattr(policy, "predict_values"):
                        import torch
                        with torch.no_grad():
                            t = torch.FloatTensor(row)
                            val = policy.predict_values(t)
                        results.append(float(val.cpu().numpy().ravel()[0]))
                    else:
                        action, _ = policy.predict(row, deterministic=True)
                        results.append(float(np.asarray(action).ravel()[0]))
                except Exception:
                    action, _ = policy.predict(row, deterministic=True)
                    results.append(float(np.asarray(action).ravel()[0]))
            return np.array(results, dtype=np.float32)

        return cls(
            model_fn=_policy_fn,
            feature_names=feature_names,
            background_data=background_data,
            config=config,
        )

    @classmethod
    def from_torch_model(
        cls,
        model: Any,
        feature_names: List[str],
        background_data: np.ndarray,
        config: Optional[SHAPConfig] = None,
    ) -> "SHAPAnalyzer":
        """Build using DeepExplainer for a PyTorch nn.Module.

        The model must accept (batch, n_features) float tensors.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise ImportError("PyTorch is required for DeepExplainer.")
        cfg = config or SHAPConfig()
        cfg.explainer_type = "deep"

        def _model_fn(obs: np.ndarray) -> np.ndarray:
            import torch
            with torch.no_grad():
                t = torch.FloatTensor(obs)
                out = model(t)
            return out.cpu().numpy().reshape(len(obs), -1).mean(axis=1)

        return cls(
            model_fn=_model_fn,
            feature_names=feature_names,
            background_data=background_data,
            config=cfg,
            torch_model=model,
        )

    # ------------------------------------------------------------------
    # Core explain method
    # ------------------------------------------------------------------

    def explain(
        self,
        data: np.ndarray,
        n_explain: Optional[int] = None,
    ) -> SHAPResult:
        """Compute SHAP values for *data*.

        Parameters
        ----------
        data:
            Test samples to explain, shape (n_samples, n_features).
        n_explain:
            Override config.n_explain for this call.

        Returns
        -------
        SHAPResult with shap_values (n_samples, n_features).
        """
        data = np.asarray(data, dtype=np.float32)
        limit = n_explain if n_explain is not None else self.config.n_explain
        if limit is not None and len(data) > limit:
            rng = np.random.default_rng(self.config.seed)
            idx = rng.choice(len(data), size=limit, replace=False)
            data = data[idx]

        explainer = self._get_explainer()

        if self.config.explainer_type == "kernel":
            shap_vals = explainer.shap_values(data)
            if isinstance(shap_vals, list):
                # Multi-output: average over outputs
                shap_vals = np.mean(shap_vals, axis=0)
            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = float(np.mean(expected))
        elif self.config.explainer_type == "deep":
            import torch
            data_t = torch.FloatTensor(data)
            shap_vals = explainer.shap_values(data_t)
            if isinstance(shap_vals, list):
                shap_vals = np.mean(shap_vals, axis=0)
            if hasattr(shap_vals, "numpy"):
                shap_vals = shap_vals.numpy()
            expected = float(np.mean(explainer.expected_value))
        else:
            raise ValueError(f"Unknown explainer_type: {self.config.explainer_type}")

        shap_vals = np.asarray(shap_vals, dtype=np.float32)

        return SHAPResult(
            shap_values=shap_vals,
            expected_value=expected,
            feature_names=self.feature_names,
            explainer_type=self.config.explainer_type,
            data=data,
        )

    # ------------------------------------------------------------------
    # Convenience: explain from environment trajectories
    # ------------------------------------------------------------------

    def explain_from_obs_buffer(
        self,
        obs_buffer: np.ndarray,
        flatten: bool = True,
    ) -> SHAPResult:
        """Explain from a raw observation buffer (T, window, features).

        Parameters
        ----------
        obs_buffer:
            Shape (T, window_size, n_features) or (T, n_features).
        flatten:
            If True and obs_buffer is 3-D, flatten to (T, window*features).
            Feature names are repeated for each time step as
            "rsi_t0", "rsi_t1", etc.
        """
        if obs_buffer.ndim == 3 and flatten:
            T, W, F = obs_buffer.shape
            data = obs_buffer.reshape(T, W * F)
            names: List[str] = []
            for t in range(W):
                for fname in self.feature_names:
                    names.append(f"{fname}_t{t}")
            orig_names = self.feature_names
            self.feature_names = names
            result = self.explain(data)
            self.feature_names = orig_names
            return result
        return self.explain(obs_buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_explainer(self) -> Any:
        """Lazily build and cache the SHAP explainer."""
        if self._explainer is not None:
            return self._explainer

        cfg = self.config
        bg = self.background_data

        if cfg.n_background is not None and len(bg) > cfg.n_background:
            rng = np.random.default_rng(cfg.seed)
            idx = rng.choice(len(bg), size=cfg.n_background, replace=False)
            bg = bg[idx]

        if cfg.explainer_type == "kernel":
            # Use kmeans summary to speed up KernelExplainer
            n_km = min(50, len(bg))
            bg_summary = shap.kmeans(bg, n_km)
            self._explainer = shap.KernelExplainer(
                self.model_fn,
                bg_summary,
                link=cfg.link,
            )
        elif cfg.explainer_type == "deep":
            if not _TORCH_AVAILABLE or self.torch_model is None:  # pragma: no cover
                raise ValueError("DeepExplainer requires a torch_model and PyTorch.")
            import torch
            bg_t = torch.FloatTensor(bg)
            self._explainer = shap.DeepExplainer(self.torch_model, bg_t)
        else:
            raise ValueError(f"Unknown explainer_type: {cfg.explainer_type!r}")

        return self._explainer

    def reset_explainer(self) -> None:
        """Force re-creation of the explainer on next call (e.g. after refit)."""
        self._explainer = None


# ------------------------------------------------------------------
# Convenience function: quick importance ranking
# ------------------------------------------------------------------

def compute_feature_importance(
    model_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: List[str],
    background_data: np.ndarray,
    explain_data: np.ndarray,
    n_background: int = 50,
    n_explain: int = 100,
) -> List[Tuple[str, float]]:
    """One-shot convenience wrapper: compute and return feature ranking.

    Returns
    -------
    List of (feature_name, mean_abs_shap) sorted descending.
    """
    cfg = SHAPConfig(n_background=n_background, n_explain=n_explain)
    analyzer = SHAPAnalyzer(
        model_fn=model_fn,
        feature_names=feature_names,
        background_data=background_data,
        config=cfg,
    )
    result = analyzer.explain(explain_data)
    return result.ranking()
