"""
Hyperparameter optimization module.

Exports the Optuna-based implementation (Week 8 rebuild).
The legacy Ray Tune module (hyperopt_ray.py) is kept for reference
but is no longer imported at package level.
"""

from .hyperopt_optuna import (
    OptunaHyperopt,
    HyperoptResult,
    TrialResult,
    run_hyperopt,
)

__all__ = [
    "OptunaHyperopt",
    "HyperoptResult",
    "TrialResult",
    "run_hyperopt",
]
