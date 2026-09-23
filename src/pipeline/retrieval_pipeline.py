"""High-level retrieval pipeline for MatchMind.

This module orchestrates the validated MatchMind retrieval components:

StatsBomb events
    -> textual event documents
    -> football-aware chunks
    -> dense sentence embeddings
    -> semantic search

The pipeline provides a simple interface for indexing one match and
retrieving relevant football possessions or chunks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import AutoTokenizer

from src.chunking import (
    FootballChunker,
    FootballChunkerConfig,
)
from src.data.statsbomb_loader import load_events
from src.embeddings import (
    SentenceEmbedder,
    SentenceEmbedderConfig,
)
from src.retrieval import SemanticSearcher
from src.text.event_formatter import build_event_documents


JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


class RetrievalPipeline:
    """High-level semantic retrieval pipeline for one football match.

    Parameters
    ----------
    chunker_config:
        Configuration controlling football-aware chunking.
    embedder_config:
        Configuration controlling sentence embeddings.

    Notes
    -----
    A match must be indexed with :meth:`index_match` before calling
    :meth:`search`.
    """

    def __init__(
        self,
        *,
        chunker_config: FootballChunkerConfig | None = None,
        embedder_config: SentenceEmbedderConfig | None = None,
    ) -> None:
        self.embedder = SentenceEmbedder(
            config=embedder_config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedder.config.model_name
        )

        self.chunker = FootballChunker(
            tokenizer=self.tokenizer,
            config=chunker_config,
        )

        self.match_id: int | None = None

        self.events: JSONRecords = []
        self.event_documents: JSONRecords = []
        self.chunks: JSONRecords = []

        self.embeddings: np.ndarray | None = None
        self.searcher: SemanticSearcher | None = None

    def index_match(
        self,
        match_id: int,
        *,
        include_location: bool = False,
        include_shootout: bool = False,
        show_progress_bar: bool = False,
    ) -> dict[str, Any]:
        """Build the complete retrieval index for one match.

        Parameters
        ----------
        match_id:
            StatsBomb match identifier.
        include_location:
            Whether event text should contain pitch coordinates.
        include_shootout:
            Whether penalty-shootout events should be included.
        show_progress_bar:
            Whether embedding generation should display a progress bar.

        Returns
        -------
        dict[str, Any]
            Summary describing the indexed match.
        """

        if not isinstance(match_id, int):
            raise TypeError(
                "match_id must be an integer."
            )

        if match_id <= 0:
            raise ValueError(
                "match_id must be greater than zero."
            )

        events = load_events(
            match_id
        )

        event_documents = build_event_documents(
            events,
            include_location=include_location,
            include_shootout=include_shootout,
        )

        chunks = self.chunker.build_chunks(
            event_documents
        )

        embeddings = self.embedder.encode_chunks(
            chunks,
            show_progress_bar=show_progress_bar,
        )

        searcher = SemanticSearcher(
            chunks=chunks,
            embeddings=embeddings,
            embedder=self.embedder,
        )

        self.match_id = match_id
        self.events = events
        self.event_documents = event_documents
        self.chunks = chunks
        self.embeddings = embeddings
        self.searcher = searcher

        return self.get_index_summary()

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        unique_possessions: bool = True,
    ) -> JSONRecords:
        """Search the currently indexed match.

        Parameters
        ----------
        query:
            Natural-language football query.
        top_k:
            Maximum number of results returned.
        unique_possessions:
            If True, return at most one result per possession.
            If False, return the highest-scoring chunks regardless
            of whether several belong to the same possession.

        Returns
        -------
        list[dict[str, Any]]
            Ranked semantic retrieval results.
        """

        searcher = self._require_searcher()

        if unique_possessions:
            return searcher.search_possessions(
                query,
                top_k=top_k,
            )

        return searcher.search(
            query,
            top_k=top_k,
        )

    def get_index_summary(
        self,
    ) -> dict[str, Any]:
        """Return summary information about the current index."""

        if self.match_id is None:
            return {
                "indexed": False,
                "match_id": None,
                "events": 0,
                "event_documents": 0,
                "chunks": 0,
                "embedding_dimension": (
                    self.embedder.embedding_dimension
                ),
            }

        return {
            "indexed": True,
            "match_id": self.match_id,
            "events": len(self.events),
            "event_documents": len(
                self.event_documents
            ),
            "chunks": len(self.chunks),
            "embedding_dimension": (
                self.embedder.embedding_dimension
            ),
        }

    def _require_searcher(
        self,
    ) -> SemanticSearcher:
        """Return the searcher or raise if no match has been indexed."""

        if self.searcher is None:
            raise RuntimeError(
                "No match has been indexed. "
                "Call index_match() before search()."
            )

        return self.searcher