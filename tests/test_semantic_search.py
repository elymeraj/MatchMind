"""Tests for the MatchMind semantic search engine."""

import numpy as np
import pytest

from src.retrieval import SemanticSearcher


class FakeEmbedder:
    """Small deterministic embedder for retrieval unit tests."""

    embedding_dimension = 3

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        normalized_text = (
            text.lower()
            .replace("é", "e")
            .replace("è", "e")
            .replace("ê", "e")
            .replace("ë", "e")
        )

        if "mbappe" in normalized_text:
            return np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )

        return np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )


def build_searcher() -> SemanticSearcher:
    """Create a deterministic search engine fixture."""

    chunks = [
        {
            "chunk_id": "chunk_1",
            "possession_id": 10,
            "text": "Mbappé takes a shot.",
        },
        {
            "chunk_id": "chunk_2",
            "possession_id": 20,
            "text": "Messi carries the ball.",
        },
        {
            "chunk_id": "chunk_3",
            "possession_id": 10,
            "text": "France attacks again.",
        },
    ]

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.8, 0.2, 0.0],
    ], dtype=np.float32)

    return SemanticSearcher(
        chunks=chunks,
        embeddings=embeddings,
        embedder=FakeEmbedder(),
    )


def test_search_returns_best_chunk() -> None:
    """The highest similarity result should be ranked first."""

    searcher = build_searcher()

    results = searcher.search(
        "Mbappé shoots",
        top_k=2,
    )

    assert len(results) == 2

    assert (
        results[0]["chunk"]["chunk_id"]
        == "chunk_1"
    )

    assert (
        results[0]["score"]
        >= results[1]["score"]
    )


def test_possession_deduplication() -> None:
    """Possession search should return unique possessions."""

    searcher = build_searcher()

    results = searcher.search_possessions(
        "Mbappé shoots",
        top_k=2,
    )

    possession_ids = [
        result["chunk"]["possession_id"]
        for result in results
    ]

    assert len(possession_ids) == 2

    assert len(
        set(possession_ids)
    ) == 2


def test_invalid_top_k() -> None:
    """Non-positive Top-K values should be rejected."""

    searcher = build_searcher()

    with pytest.raises(ValueError):
        searcher.search(
            "Mbappé",
            top_k=0,
        )


def test_empty_query() -> None:
    """Empty queries should be rejected."""

    searcher = build_searcher()

    with pytest.raises(ValueError):
        searcher.search(
            "",
            top_k=5,
        )


def test_embedding_count_must_match_chunks() -> None:
    """Every chunk must have exactly one embedding row."""

    chunks = [
        {
            "text": "Test",
            "possession_id": 1,
        }
    ]

    embeddings = np.zeros(
        (2, 3),
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        SemanticSearcher(
            chunks=chunks,
            embeddings=embeddings,
            embedder=FakeEmbedder(),
        )