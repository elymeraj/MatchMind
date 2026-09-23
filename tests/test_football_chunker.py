"""Tests for the MatchMind football-aware chunker."""

import pytest

from src.chunking import (
    FootballChunker,
    FootballChunkerConfig,
    get_match_phase,
    get_period_name,
)


class SimpleTokenizer:
    """Minimal tokenizer used for deterministic chunking tests."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> dict[str, list[int]]:
        tokens = text.split()

        token_count = len(tokens)

        if add_special_tokens:
            token_count += 2

        return {
            "input_ids": list(
                range(token_count)
            )
        }


def make_event(
    *,
    index: int,
    text: str,
    possession_id: int = 10,
    team: str = "Argentina",
    period: int = 1,
    minute: int = 10,
) -> dict:
    """Create a minimal event document for chunking tests."""

    return {
        "event_id": f"event_{index}",
        "index": index,
        "possession_id": possession_id,
        "possession_team": team,
        "period": period,
        "minute": minute,
        "second": index,
        "play_pattern": "Regular Play",
        "player": "Test Player",
        "event_type": "Pass",
        "text": text,
    }


def test_period_names() -> None:
    """Football periods should have readable labels."""

    assert get_period_name(1) == "First Half"
    assert get_period_name(2) == "Second Half"
    assert get_period_name(3) == "First Extra Time"
    assert get_period_name(4) == "Second Extra Time"


def test_match_phases() -> None:
    """Periods should map to broader match phases."""

    assert get_match_phase(1) == "Regulation Time"
    assert get_match_phase(2) == "Regulation Time"
    assert get_match_phase(3) == "Extra Time"
    assert get_match_phase(4) == "Extra Time"


def test_invalid_configuration() -> None:
    """Invalid chunking parameters should raise errors."""

    with pytest.raises(ValueError):
        FootballChunkerConfig(
            max_tokens=0
        )

    with pytest.raises(ValueError):
        FootballChunkerConfig(
            overlap_events=-1
        )


def test_chunker_preserves_metadata() -> None:
    """Chunks should preserve useful structured metadata."""

    tokenizer = SimpleTokenizer()

    chunker = FootballChunker(
        tokenizer,
        FootballChunkerConfig(
            max_tokens=100,
            overlap_events=1,
        ),
    )

    documents = [
        make_event(
            index=1,
            text="10:00 - Argentina completes a pass.",
        ),
        make_event(
            index=2,
            text="10:02 - Argentina carries the ball.",
        ),
    ]

    chunks = chunker.build_chunks(
        documents
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk["possession_id"] == 10
    assert chunk["possession_team"] == "Argentina"
    assert chunk["period_name"] == "First Half"
    assert chunk["match_phase"] == "Regulation Time"
    assert chunk["event_count"] == 2

    assert (
        "Match phase: Regulation Time"
        not in chunk["text"]
    )


def test_chunker_respects_token_budget() -> None:
    """Every produced chunk must respect the token budget."""

    tokenizer = SimpleTokenizer()

    chunker = FootballChunker(
        tokenizer,
        FootballChunkerConfig(
            max_tokens=18,
            overlap_events=1,
        ),
    )

    documents = [
        make_event(
            index=1,
            text="Argentina completes a short pass.",
        ),
        make_event(
            index=2,
            text="Argentina carries the ball forward.",
        ),
        make_event(
            index=3,
            text="Argentina completes another forward pass.",
        ),
    ]

    chunks = chunker.build_chunks(
        documents
    )

    assert len(chunks) >= 2

    assert all(
        chunk["token_count"] <= 18
        for chunk in chunks
    )


def test_chunker_keeps_event_order() -> None:
    """Events should remain ordered by their original index."""

    tokenizer = SimpleTokenizer()

    chunker = FootballChunker(
        tokenizer,
        FootballChunkerConfig(
            max_tokens=100,
            overlap_events=1,
        ),
    )

    documents = [
        make_event(
            index=3,
            text="Third event.",
        ),
        make_event(
            index=1,
            text="First event.",
        ),
        make_event(
            index=2,
            text="Second event.",
        ),
    ]

    chunks = chunker.build_chunks(
        documents
    )

    text = chunks[0]["text"]

    assert (
        text.index("First event")
        < text.index("Second event")
        < text.index("Third event")
    )