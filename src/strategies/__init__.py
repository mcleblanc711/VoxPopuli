"""Trading strategy implementations."""

from .base_strategy import BaseStrategy, Signal, Position
from .attention_momentum import AttentionMomentumStrategy
from .sentiment_divergence import SentimentDivergenceStrategy
from .contrarian import ContrarianStrategy
from .cross_subreddit import CrossSubredditStrategy
from .velocity_sentiment import VelocitySentimentStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "Position",
    "AttentionMomentumStrategy",
    "SentimentDivergenceStrategy",
    "ContrarianStrategy",
    "CrossSubredditStrategy",
    "VelocitySentimentStrategy",
]

STRATEGY_REGISTRY = {
    "attention_momentum": AttentionMomentumStrategy,
    "sentiment_divergence": SentimentDivergenceStrategy,
    "contrarian": ContrarianStrategy,
    "cross_subreddit": CrossSubredditStrategy,
    "velocity_sentiment": VelocitySentimentStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Get a strategy by name.

    Args:
        name: Strategy name.
        **kwargs: Strategy parameters.

    Returns:
        Strategy instance.
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](**kwargs)
