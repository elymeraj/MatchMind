"""In-memory semantic retrieval utilities for MatchMind.

This module performs exact semantic search over precomputed dense
embeddings.

It intentionally remains independent from vector databases such as
Qdrant. The same retrieval concepts implemented here will later be
delegated to a persistent vector store.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.embeddings import SentenceEmbedder


JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


class SemanticSearcher:
    """Perform exact semantic search over football chunks.

    Parameters
    ----------
    chunks:
        Retrieval chunks corresponding to the embedding matrix.
    embeddings:
        Dense vector representation of every chunk.
    embedder:
        Sentence embedder used to encode user queries.

    Notes
    -----
    The row order of ``embeddings`` must match the order of ``chunks``.

    For example:

    ``embeddings[10]`` must represent ``chunks[10]``.
    """

    def __init__(
        self,
        chunks: Sequence[JSONRecord],
        embeddings: np.ndarray,
        embedder: SentenceEmbedder,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder

        if not self.chunks:
            raise ValueError(
                "chunks cannot be empty."
            )

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self._validate_embeddings()

        self.normalized_embeddings = (
            self._normalize_matrix(
                self.embeddings
            )
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        deduplicate_by: str | None = None,
    ) -> JSONRecords:
        """Retrieve the most semantically similar chunks.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results returned.
        deduplicate_by:
            Optional chunk metadata key used to remove duplicate
            retrieval results.

            For example, using ``"possession_id"`` returns at most one
            chunk for each football possession.

        Returns
        -------
        list[dict[str, Any]]
            Ranked retrieval results containing similarity scores and
            their corresponding chunks.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = (
            self.embedder.encode_text(
                query
            )
        )

        query_embedding = (
            self._normalize_vector(
                query_embedding
            )
        )

        scores = (
            self.normalized_embeddings
            @ query_embedding
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results: JSONRecords = []
        seen_values: set[Any] = set()

        for raw_rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            chunk_index = int(index)
            chunk = self.chunks[
                chunk_index
            ]

            if deduplicate_by is not None:
                if deduplicate_by not in chunk:
                    raise KeyError(
                        "Cannot deduplicate retrieval "
                        f"results because chunk "
                        f"{chunk_index} does not contain "
                        f"the key '{deduplicate_by}'."
                    )

                value = chunk[
                    deduplicate_by
                ]

                if value in seen_values:
                    continue

                seen_values.add(value)

            results.append(
                {
                    "rank": len(results) + 1,
                    "raw_rank": raw_rank,
                    "score": float(
                        scores[chunk_index]
                    ),
                    "chunk_index": chunk_index,
                    "chunk": chunk,
                }
            )

            if len(results) >= top_k:
                break

        return results

    def search_possessions(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> JSONRecords:
        """Retrieve unique football possessions.

        This is a convenience wrapper around :meth:`search` using
        ``possession_id`` as the deduplication key.
        """

        return self.search(
            query,
            top_k=top_k,
            deduplicate_by="possession_id",
        )

    def _validate_embeddings(
        self,
    ) -> None:
        """Validate the chunk embedding matrix."""

        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must be a "
                "two-dimensional matrix."
            )

        if (
            self.embeddings.shape[0]
            != len(self.chunks)
        ):
            raise ValueError(
                "The number of embedding rows "
                "must match the number of chunks. "
                f"Received {self.embeddings.shape[0]} "
                "embeddings for "
                f"{len(self.chunks)} chunks."
            )

        expected_dimension = (
            self.embedder.embedding_dimension
        )

        if (
            self.embeddings.shape[1]
            != expected_dimension
        ):
            raise ValueError(
                "Unexpected embedding dimension. "
                f"Expected {expected_dimension}, "
                f"received "
                f"{self.embeddings.shape[1]}."
            )

        if not np.all(
            np.isfinite(
                self.embeddings
            )
        ):
            raise ValueError(
                "embeddings contain non-finite values."
            )

    @staticmethod
    def _normalize_matrix(
        matrix: np.ndarray,
    ) -> np.ndarray:
        """L2-normalize every embedding vector."""

        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "embeddings contain a zero vector."
            )

        return (
            matrix
            / norms
        ).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _normalize_vector(
        vector: np.ndarray,
    ) -> np.ndarray:
        """L2-normalize one embedding vector."""

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Query embedding must be "
                "one-dimensional."
            )

        norm = np.linalg.norm(
            vector
        )

        if norm == 0:
            raise ValueError(
                "Query embedding cannot be "
                "a zero vector."
            )

        return (
            vector
            / norm
        ).astype(
            np.float32,
            copy=False,
        )