"""Football-aware chunking utilities for MatchMind.

This module groups textual football events into retrieval chunks while
preserving complete event boundaries and respecting a tokenizer-based
maximum sequence length.

The chunker does not create embeddings. Its responsibility is limited to
building retrieval-ready text chunks and preserving structured metadata.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


PERIOD_NAMES = {
    1: "First Half",
    2: "Second Half",
    3: "First Extra Time",
    4: "Second Extra Time",
    5: "Penalty Shootout",
}


MATCH_PHASES = {
    1: "Regulation Time",
    2: "Regulation Time",
    3: "Extra Time",
    4: "Extra Time",
    5: "Penalty Shootout",
}


def get_period_name(period: int | None) -> str:
    """Return a human-readable football period name."""

    if period is None:
        return "Unknown"

    return PERIOD_NAMES.get(
        int(period),
        "Unknown",
    )


def get_match_phase(period: int | None) -> str:
    """Return the broader football match phase."""

    if period is None:
        return "Unknown"

    return MATCH_PHASES.get(
        int(period),
        "Unknown",
    )


@dataclass(frozen=True)
class FootballChunkerConfig:
    """Configuration for football-aware chunking.

    Parameters
    ----------
    max_tokens:
        Maximum number of tokenizer tokens allowed in one chunk.
    overlap_events:
        Number of complete football events repeated between neighboring
        chunks from the same possession.
    """

    max_tokens: int = 220
    overlap_events: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values."""

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        if self.overlap_events < 0:
            raise ValueError(
                "overlap_events cannot be negative."
            )


class FootballChunker:
    """Build football-aware retrieval chunks.

    The chunker preserves complete event descriptions and splits a
    possession only between events.

    Structured metadata such as the match period and play pattern is
    stored alongside the chunk but is not systematically injected into
    the embedding text.

    Parameters
    ----------
    tokenizer:
        Hugging Face compatible tokenizer used to measure chunk length.
    config:
        Football-aware chunking configuration.
    """

    def __init__(
        self,
        tokenizer: Any,
        config: FootballChunkerConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = (
            config
            if config is not None
            else FootballChunkerConfig()
        )

    def count_tokens(
        self,
        text: str,
    ) -> int:
        """Count tokenizer tokens without truncating the input."""

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        return len(
            encoded["input_ids"]
        )

    def build_chunks(
        self,
        event_documents: JSONRecords,
    ) -> JSONRecords:
        """Build retrieval chunks from textual football events.

        Events are first grouped by StatsBomb possession. Complete event
        descriptions are then accumulated until adding another event
        would exceed the configured token budget.

        Neighboring chunks can share a configurable number of complete
        events.

        Parameters
        ----------
        event_documents:
            Event-level documents produced by
            ``build_event_documents``.

        Returns
        -------
        list[dict[str, Any]]
            Football-aware chunks containing text and structured
            metadata.
        """

        events_by_possession = self._group_by_possession(
            event_documents
        )

        chunks: JSONRecords = []

        for (
            possession_id,
            possession_events,
        ) in events_by_possession.items():
            chunks.extend(
                self._chunk_possession(
                    possession_id,
                    possession_events,
                )
            )

        return chunks

    @staticmethod
    def _group_by_possession(
        event_documents: JSONRecords,
    ) -> dict[int, JSONRecords]:
        """Group event documents by possession ID."""

        grouped: dict[int, JSONRecords] = defaultdict(list)

        for document in event_documents:
            possession_id = document.get(
                "possession_id"
            )

            if possession_id is None:
                continue

            grouped[
                int(possession_id)
            ].append(document)

        for possession_events in grouped.values():
            possession_events.sort(
                key=lambda document: document.get(
                    "index",
                    0,
                )
            )

        return dict(grouped)

    def _chunk_possession(
        self,
        possession_id: int,
        possession_events: JSONRecords,
    ) -> JSONRecords:
        """Split one possession into football-aware chunks."""

        chunks: JSONRecords = []

        position = 0
        chunk_number = 0

        while position < len(
            possession_events
        ):
            first_event = possession_events[
                position
            ]

            possession_team = first_event.get(
                "possession_team"
            ) or "Unknown"

            header = self._build_text_header(
                possession_id=possession_id,
                possession_team=possession_team,
            )

            selected_events: JSONRecords = []
            cursor = position

            while cursor < len(
                possession_events
            ):
                candidate_events = (
                    selected_events
                    + [
                        possession_events[
                            cursor
                        ]
                    ]
                )

                candidate_text = (
                    self._build_chunk_text(
                        header,
                        candidate_events,
                    )
                )

                if (
                    self.count_tokens(
                        candidate_text
                    )
                    <= self.config.max_tokens
                ):
                    selected_events = (
                        candidate_events
                    )
                    cursor += 1
                else:
                    break

            if not selected_events:
                event = possession_events[
                    position
                ]

                event_id = event.get(
                    "event_id",
                    "unknown",
                )

                event_text = event.get(
                    "text",
                    "",
                )

                event_token_count = (
                    self.count_tokens(
                        self._build_chunk_text(
                            header,
                            [event],
                        )
                    )
                )

                raise ValueError(
                    "A single football event exceeds the "
                    "configured chunk token budget. "
                    f"Event ID: {event_id}. "
                    f"Tokens: {event_token_count}. "
                    f"Budget: {self.config.max_tokens}. "
                    f"Text: {event_text}"
                )

            chunk_text = (
                self._build_chunk_text(
                    header,
                    selected_events,
                )
            )

            chunks.append(
                self._build_chunk_record(
                    possession_id=possession_id,
                    chunk_number=chunk_number,
                    possession_team=possession_team,
                    selected_events=selected_events,
                    text=chunk_text,
                )
            )

            chunk_number += 1

            if cursor >= len(
                possession_events
            ):
                break

            position = (
                self._next_position(
                    current_position=position,
                    cursor=cursor,
                    selected_event_count=len(
                        selected_events
                    ),
                )
            )

        return chunks

    @staticmethod
    def _build_text_header(
        *,
        possession_id: int,
        possession_team: str,
    ) -> str:
        """Build the minimal text header used for embeddings."""

        return (
            f"Possession {possession_id}. "
            f"{possession_team}."
        )

    @staticmethod
    def _build_chunk_text(
        header: str,
        events: JSONRecords,
    ) -> str:
        """Combine a possession header with complete event texts."""

        event_text = " ".join(
            event["text"]
            for event in events
        )

        return (
            f"{header}\n\n"
            f"{event_text}"
        )

    def _next_position(
        self,
        *,
        current_position: int,
        cursor: int,
        selected_event_count: int,
    ) -> int:
        """Calculate the next event position while preserving overlap."""

        effective_overlap = min(
            self.config.overlap_events,
            max(
                selected_event_count - 1,
                0,
            ),
        )

        next_position = (
            cursor
            - effective_overlap
        )

        return max(
            current_position + 1,
            next_position,
        )

    def _build_chunk_record(
        self,
        *,
        possession_id: int,
        chunk_number: int,
        possession_team: str,
        selected_events: JSONRecords,
        text: str,
    ) -> JSONRecord:
        """Build the final retrieval chunk and its metadata."""

        first_event = selected_events[0]
        last_event = selected_events[-1]

        period = first_event.get(
            "period"
        )

        players = sorted({
            event["player"]
            for event in selected_events
            if event.get("player")
        })

        event_types = sorted({
            event["event_type"]
            for event in selected_events
            if event.get("event_type")
        })

        source_event_ids = [
            event["event_id"]
            for event in selected_events
            if event.get("event_id")
        ]

        source_event_indices = [
            event["index"]
            for event in selected_events
            if event.get("index") is not None
        ]

        return {
            "chunk_id": (
                f"possession_{possession_id}"
                f"_football_{chunk_number}"
            ),
            "strategy": "football_aware",
            "possession_id": possession_id,
            "period": period,
            "period_name": get_period_name(
                period
            ),
            "match_phase": get_match_phase(
                period
            ),
            "possession_team": (
                possession_team
            ),
            "play_pattern": first_event.get(
                "play_pattern"
            ),
            "start_minute": first_event.get(
                "minute"
            ),
            "start_second": first_event.get(
                "second"
            ),
            "end_minute": last_event.get(
                "minute"
            ),
            "end_second": last_event.get(
                "second"
            ),
            "players": players,
            "event_types": event_types,
            "event_count": len(
                selected_events
            ),
            "source_event_ids": (
                source_event_ids
            ),
            "source_event_indices": (
                source_event_indices
            ),
            "text": text,
            "token_count": (
                self.count_tokens(
                    text
                )
            ),
        }