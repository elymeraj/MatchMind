"""Integration test for the MatchMind retrieval pipeline."""

import numpy as np
import pytest

from transformers import AutoTokenizer

from src.chunking import FootballChunker
from src.data.statsbomb_loader import load_events
from src.embeddings import SentenceEmbedder
from src.retrieval import SemanticSearcher
from src.text.event_formatter import build_event_documents


MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

MATCH_ID = 3869685


@pytest.mark.integration
def test_world_cup_final_retrieval_pipeline() -> None:
    """Run the validated MatchMind retrieval pipeline end to end."""

    events = load_events(
        MATCH_ID
    )

    event_documents = (
        build_event_documents(
            events,
            include_location=False,
            include_shootout=False,
        )
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    chunker = FootballChunker(
        tokenizer
    )

    chunks = chunker.build_chunks(
        event_documents
    )

    embedder = SentenceEmbedder()

    embeddings = (
        embedder.encode_chunks(
            chunks,
            show_progress_bar=False,
        )
    )

    searcher = SemanticSearcher(
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder,
    )

    results = (
        searcher.search_possessions(
            (
                "A France attack where "
                "Kylian Mbappé takes a shot"
            ),
            top_k=5,
        )
    )

    assert len(events) == 4407
    assert len(event_documents) == 3252
    assert len(chunks) == 428

    assert embeddings.shape == (
        428,
        384,
    )

    assert embeddings.dtype == np.float32

    assert len(results) == 5

    possession_ids = [
        result["chunk"][
            "possession_id"
        ]
        for result in results
    ]

    assert len(
        set(possession_ids)
    ) == 5

    assert all(
        results[index]["score"]
        >= results[index + 1]["score"]
        for index in range(
            len(results) - 1
        )
    )