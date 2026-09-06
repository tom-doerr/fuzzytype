"""fuzzytype -- predictive text input from a base LM plus a noisy channel.

A base model decodes the probable continuations of what you have written; a
fuzzy match reads your keystrokes as a noisy transmission of one of them. The
two combine into a posterior over "what you meant", which both ranks the
suggestions and decides which branches are worth decoding further.
"""

from .channel import ChannelCosts, MatchResult, match
from .engine import DEFAULT_PREAMBLE, Engine, EngineConfig
from .lm import DEFAULT_MODEL, LanguageModel, TopK
from .search import Candidate, PredictConfig, PredictStats, predict
from .rank import Suggestion, rerank

__all__ = [
    "ChannelCosts",
    "MatchResult",
    "match",
    "Engine",
    "EngineConfig",
    "DEFAULT_PREAMBLE",
    "DEFAULT_MODEL",
    "LanguageModel",
    "TopK",
    "Candidate",
    "PredictConfig",
    "PredictStats",
    "predict",
    "Suggestion",
    "rerank",
]
__version__ = "0.1.0"
