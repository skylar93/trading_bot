"""
Continual learning pipeline for the trading bot.

Week 23 — Phase 8: Production Readiness & Advanced Integration

Modules:
    experience_store   — RegimeAwareExperienceStore with EWC support
    adaptive_trainer   — Drift-triggered auto-retraining pipeline
"""

from training.continual.experience_store import EWCRegularizer, RegimeAwareExperienceStore
from training.continual.adaptive_trainer import AdaptiveTrainer

__all__ = [
    "RegimeAwareExperienceStore",
    "EWCRegularizer",
    "AdaptiveTrainer",
]
