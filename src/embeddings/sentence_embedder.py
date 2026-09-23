"""Sentence embedding utilities for MatchMind.

This module wraps a SentenceTransformer model and provides a small,
reusable interface for encoding football text, user queries, and
retrieval chunks.

The embedder is intentionally independent from chunking and retrieval.
Its only responsibility is converting text into dense vector
representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


@dataclass(frozen=True)
class SentenceEmbedderConfig:
    """Configuration for sentence embedding.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier used by SentenceTransformer.
    batch_size:
        Number of texts encoded together.
    normalize_embeddings:
        Whether returned vectors should have unit L2 norm.
        Normalized embeddings allow cosine similarity to be computed
        efficiently with a dot product.
    device:
        PyTorch device used for inference.
    """

    model_name: str = DEFAULT_MODEL_NAME
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )


class SentenceEmbedder:
    """Encode text into dense sentence embeddings.

    Parameters
    ----------
    config:
        Embedding model configuration.
    """

    def __init__(
        self,
        config: SentenceEmbedderConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else SentenceEmbedderConfig()
        )

        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

        embedding_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if embedding_dimension is None:
            raise RuntimeError(
                "The embedding model did not expose "
                "its embedding dimension."
            )

        self.embedding_dimension = int(
            embedding_dimension
        )

    def encode_text(
        self,
        text: str,
    ) -> np.ndarray:
        """Encode one text into a one-dimensional embedding vector."""

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        if not text.strip():
            raise ValueError(
                "text cannot be empty."
            )

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=(
                self.config.normalize_embeddings
            ),
            show_progress_bar=False,
        )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        self._validate_single_embedding(
            embedding
        )

        return embedding

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode multiple texts into a two-dimensional matrix."""

        text_list = list(texts)

        if not text_list:
            raise ValueError(
                "texts cannot be empty."
            )

        for index, text in enumerate(
            text_list
        ):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"texts[{index}] cannot be empty."
                )

        embeddings = self.model.encode(
            text_list,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=(
                self.config.normalize_embeddings
            ),
            show_progress_bar=show_progress_bar,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self._validate_embedding_matrix(
            embeddings,
            expected_rows=len(text_list),
        )

        return embeddings

    def encode_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        *,
        text_key: str = "text",
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode retrieval chunks using their text field.

        The order of the returned embeddings matches the order of the
        input chunks. This makes it possible to recover chunk metadata
        using the same array index during retrieval.

        Parameters
        ----------
        chunks:
            Retrieval chunks containing textual content.
        text_key:
            Dictionary key containing the text to embed.
        show_progress_bar:
            Whether SentenceTransformer should display an encoding
            progress bar.

        Returns
        -------
        numpy.ndarray
            Matrix with shape
            ``(number_of_chunks, embedding_dimension)``.
        """

        chunk_list = list(chunks)

        if not chunk_list:
            raise ValueError(
                "chunks cannot be empty."
            )

        texts: list[str] = []

        for index, chunk in enumerate(
            chunk_list
        ):
            if text_key not in chunk:
                raise KeyError(
                    f"Chunk at index {index} does not "
                    f"contain the key '{text_key}'."
                )

            text = chunk[text_key]

            if not isinstance(text, str):
                raise TypeError(
                    f"Chunk text at index {index} "
                    "must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"Chunk text at index {index} "
                    "cannot be empty."
                )

            texts.append(text)

        return self.encode_texts(
            texts,
            show_progress_bar=show_progress_bar,
        )

    def _validate_single_embedding(
        self,
        embedding: np.ndarray,
    ) -> None:
        """Validate the shape of one embedding."""

        expected_shape = (
            self.embedding_dimension,
        )

        if embedding.shape != expected_shape:
            raise RuntimeError(
                "Unexpected embedding shape. "
                f"Expected {expected_shape}, "
                f"received {embedding.shape}."
            )

    def _validate_embedding_matrix(
        self,
        embeddings: np.ndarray,
        *,
        expected_rows: int,
    ) -> None:
        """Validate the shape of an embedding matrix."""

        expected_shape = (
            expected_rows,
            self.embedding_dimension,
        )

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                "Unexpected embedding matrix shape. "
                f"Expected {expected_shape}, "
                f"received {embeddings.shape}."
            )