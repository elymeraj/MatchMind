"""Tests for MatchMind sentence embeddings."""

import numpy as np
import pytest

from src.embeddings import SentenceEmbedder


@pytest.fixture(scope="module")
def embedder() -> SentenceEmbedder:
    """Load the embedding model once for this test module."""

    return SentenceEmbedder()


@pytest.mark.integration
def test_single_embedding_shape(
    embedder: SentenceEmbedder,
) -> None:
    """One sentence should produce one 384-dimensional vector."""

    embedding = embedder.encode_text(
        "Lionel Messi takes a shot."
    )

    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32


@pytest.mark.integration
def test_embedding_is_normalized(
    embedder: SentenceEmbedder,
) -> None:
    """Default embeddings should have approximately unit norm."""

    embedding = embedder.encode_text(
        "Argentina attacks through the centre."
    )

    norm = np.linalg.norm(
        embedding
    )

    assert np.isclose(
        norm,
        1.0,
        atol=1e-5,
    )


@pytest.mark.integration
def test_multiple_embeddings_shape(
    embedder: SentenceEmbedder,
) -> None:
    """Multiple texts should produce a two-dimensional matrix."""

    embeddings = embedder.encode_texts([
        "Lionel Messi takes a shot.",
        "Kylian Mbappé attacks for France.",
        "Hugo Lloris completes a pass.",
    ])

    assert embeddings.shape == (
        3,
        384,
    )


@pytest.mark.integration
def test_semantic_similarity(
    embedder: SentenceEmbedder,
) -> None:
    """Semantically similar football actions should be closer."""

    embeddings = embedder.encode_texts([
        "Kylian Mbappé takes a shot for France.",
        "France attacks and Mbappé attempts a shot.",
        "Hugo Lloris completes a goalkeeper pass.",
    ])

    similar_score = float(
        embeddings[0]
        @ embeddings[1]
    )

    different_score = float(
        embeddings[0]
        @ embeddings[2]
    )

    assert (
        similar_score
        > different_score
    )


@pytest.mark.integration
def test_empty_text_is_rejected(
    embedder: SentenceEmbedder,
) -> None:
    """Empty strings should not be encoded."""

    with pytest.raises(ValueError):
        embedder.encode_text("")