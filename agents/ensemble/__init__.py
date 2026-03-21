"""Ensemble agents and meta-controller."""

from .meta_controller import MetaController, MetaControllerConfig
from .regime_detector import MarketRegimeDetector, RegimeDetectorConfig, _HMMLEARN_AVAILABLE
from .communication import AgentIntention, CommunicationBus, INTENTION_DIM

__all__ = [
    "MetaController",
    "MetaControllerConfig",
    "MarketRegimeDetector",
    "RegimeDetectorConfig",
    "_HMMLEARN_AVAILABLE",
    "AgentIntention",
    "CommunicationBus",
    "INTENTION_DIM",
]
