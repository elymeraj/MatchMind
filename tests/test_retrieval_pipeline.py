"""Integration test for the MatchMind retrieval pipeline."""

import numpy as np
import pytest

from transformers import AutoTokenizer

from src.chunking import FootballChunker
from src.data.statsbomb_loader import load_events
from src.embeddings import SentenceEmbedder
from src.retrieval import SemanticSearcher
from src.text.event_formatter import build_event_documents
from src.pipeline import RetrievalPipeline

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
    
def test_pipeline_requires_index_before_search() -> None:
    """Searching before indexing should raise a clear error."""

    pipeline = RetrievalPipeline()

    with pytest.raises(
        RuntimeError,
        match="No match has been indexed",
    ):
        pipeline.search(
            "Mbappé takes a shot"
        )


@pytest.mark.integration
def test_high_level_retrieval_pipeline() -> None:
    """The high-level pipeline should index and search a real match."""

    pipeline = RetrievalPipeline()

    summary = pipeline.index_match(
        MATCH_ID,
        show_progress_bar=False,
    )

    assert summary["indexed"] is True
    assert summary["match_id"] == MATCH_ID
    assert summary["events"] == 4407
    assert summary["event_documents"] == 3252
    assert summary["chunks"] == 428
    assert summary["embedding_dimension"] == 384

    results = pipeline.search(
        (
            "A France attack where "
            "Kylian Mbappé takes a shot"
        ),
        top_k=5,
        unique_possessions=True,
    )

    assert len(results) == 5

    possession_ids = [
        result["chunk"]["possession_id"]
        for result in results
    ]

    assert len(
        set(possession_ids)
    ) == 5