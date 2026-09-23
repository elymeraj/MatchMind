"""Embedding utilities for MatchMind."""

from .sentence_embedder import (
    DEFAULT_MODEL_NAME,
    SentenceEmbedder,
    SentenceEmbedderConfig,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "SentenceEmbedder",
    "SentenceEmbedderConfig",
]