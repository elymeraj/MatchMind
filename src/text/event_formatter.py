"""
Text representation utilities for MatchMind.

This module converts structured StatsBomb football events into
human-readable text that can later be used for semantic search,
embeddings, Retrieval-Augmented Generation (RAG), and AI agents.

Two levels of representation are provided:

1. Event-level documents
   Each football event is converted into a short textual description.

2. Possession-level documents
   Consecutive events belonging to the same StatsBomb possession are
   grouped into a richer narrative describing the sequence.

The module intentionally operates on raw StatsBomb dictionaries rather
than Pandas DataFrames because the nested event structure contains the
information required to generate natural-language descriptions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


# Some StatsBomb events are useful for the event model but add little
# semantic information to a possession narrative or duplicate another
# event already present in the sequence.
DEFAULT_IGNORED_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Ball Receipt*",
    "Player On",
    "Player Off",
}


def _get_nested_name(
    record: JSONRecord,
    key: str,
    default: str = "Unknown",
) -> str:
    """
    Extract the ``name`` field from a nested StatsBomb object.

    Parameters
    ----------
    record : JSONRecord
        StatsBomb event dictionary.

    key : str
        Name of the nested object.

    default : str, default="Unknown"
        Value returned when the requested information is unavailable.

    Returns
    -------
    str
        Human-readable name.

    Examples
    --------
    A StatsBomb event may contain:

        "team": {
            "id": 779,
            "name": "Argentina"
        }

    Calling:

        _get_nested_name(event, "team")

    returns:

        "Argentina"
    """
    value = record.get(key)

    if isinstance(value, dict):
        name = value.get("name")

        if isinstance(name, str):
            return name

    return default


def _format_match_time(
    event: JSONRecord,
) -> str:
    """
    Format a StatsBomb event time as MM:SS.

    Parameters
    ----------
    event : JSONRecord
        StatsBomb event dictionary.

    Returns
    -------
    str
        Match time such as ``79:32``.
    """
    minute = int(event.get("minute", 0))
    second = int(event.get("second", 0))

    return f"{minute}:{second:02d}"


def _format_location(
    location: Any,
) -> str | None:
    """
    Convert a StatsBomb location into readable coordinates.

    Parameters
    ----------
    location : Any
        Expected StatsBomb location in the form ``[x, y]``.

    Returns
    -------
    str | None
        Formatted coordinates or None when the location is invalid.
    """
    if (
        isinstance(location, list)
        and len(location) >= 2
    ):
        x = location[0]
        y = location[1]

        return f"({x:.1f}, {y:.1f})"

    return None


def format_event(
    event: JSONRecord,
    include_location: bool = False,
) -> str:
    """
    Convert one StatsBomb event into a human-readable sentence.

    The function uses event-specific templates for the event types that
    are particularly useful for football analysis.

    Parameters
    ----------
    event : JSONRecord
        Raw StatsBomb event.

    include_location : bool, default=False
        Whether spatial coordinates should be included in the text.

        Coordinates are useful for some analytical queries, but adding
        them to every sentence may introduce unnecessary numerical noise
        for semantic retrieval. They are therefore disabled by default.

    Returns
    -------
    str
        Natural-language representation of the event.
    """
    event_type = _get_nested_name(
        event,
        "type",
    )

    team = _get_nested_name(
        event,
        "team",
    )

    player = _get_nested_name(
        event,
        "player",
        default="Unknown player",
    )

    time = _format_match_time(event)

    prefix = (
        f"{time} - {team}: "
    )

    location_text = ""

    if include_location:
        location = _format_location(
            event.get("location")
        )

        if location is not None:
            location_text = (
                f" from location {location}"
            )

    if event_type == "Pass":
        pass_data = event.get("pass", {})

        recipient = _get_nested_name(
            pass_data,
            "recipient",
            default="a teammate",
        )

        outcome = _get_nested_name(
            pass_data,
            "outcome",
            default="Complete",
        )

        if outcome == "Complete":
            sentence = (
                f"{player} completes a pass to {recipient}"
            )
        else:
            sentence = (
                f"{player} attempts a pass to {recipient}, "
                f"but the outcome is {outcome}"
            )

    elif event_type == "Carry":
        sentence = (
            f"{player} carries the ball"
        )

        if include_location:
            carry_data = event.get("carry", {})

            end_location = _format_location(
                carry_data.get("end_location")
            )

            if end_location is not None:
                sentence += (
                    f"{location_text} to {end_location}"
                )
                location_text = ""

    elif event_type == "Shot":
        shot_data = event.get("shot", {})

        xg = shot_data.get(
            "statsbomb_xg"
        )

        outcome = _get_nested_name(
            shot_data,
            "outcome",
            default="Unknown outcome",
        )

        if isinstance(xg, (int, float)):
            sentence = (
                f"{player} takes a shot with an xG "
                f"of {xg:.3f}"
            )
        else:
            sentence = (
                f"{player} takes a shot"
            )

        if outcome == "Goal":
            sentence += " and scores"
        else:
            sentence += (
                f". The shot outcome is {outcome}"
            )

    elif event_type == "Dribble":
        dribble_data = event.get(
            "dribble",
            {}
        )

        outcome = _get_nested_name(
            dribble_data,
            "outcome",
            default="Unknown outcome",
        )

        sentence = (
            f"{player} attempts a dribble "
            f"with outcome {outcome}"
        )

    elif event_type == "Ball Recovery":
        sentence = (
            f"{player} recovers the ball"
        )

    elif event_type == "Interception":
        sentence = (
            f"{player} makes an interception"
        )

    elif event_type == "Pressure":
        sentence = (
            f"{player} applies pressure"
        )

    elif event_type == "Miscontrol":
        sentence = (
            f"{player} miscontrols the ball"
        )

    elif event_type == "Clearance":
        sentence = (
            f"{player} clears the ball"
        )

    elif event_type == "Foul Won":
        sentence = (
            f"{player} wins a foul"
        )

    elif event_type == "Foul Committed":
        sentence = (
            f"{player} commits a foul"
        )

    elif event_type == "Duel":
        sentence = (
            f"{player} is involved in a duel"
        )

    elif event_type == "Substitution":
        substitution_data = event.get(
            "substitution",
            {}
        )

        replacement = _get_nested_name(
            substitution_data,
            "replacement",
            default="Unknown player",
        )

        sentence = (
            f"{player} is replaced by {replacement}"
        )

    elif event_type == "Goal Keeper":
        goalkeeper_data = event.get(
            "goalkeeper",
            {}
        )

        action = _get_nested_name(
            goalkeeper_data,
            "type",
            default="goalkeeper action",
        )

        sentence = (
            f"{player} performs a goalkeeper action: "
            f"{action}"
        )

    else:
        sentence = (
            f"{player} performs a {event_type} event"
        )

    if location_text:
        sentence += location_text

    return prefix + sentence + "."


def build_event_documents(
    events: JSONRecords,
    include_location: bool = False,
    include_shootout: bool = False,
) -> list[JSONRecord]:
    """
    Convert StatsBomb events into event-level text documents.

    Each document contains both human-readable text and structured
    metadata.

    Parameters
    ----------
    events : JSONRecords
        Raw StatsBomb event records.

    include_location : bool, default=False
        Whether event coordinates should appear in the text.

    include_shootout : bool, default=False
        Whether period 5 penalty-shootout events should be included.

    Returns
    -------
    list[JSONRecord]
        Event-level documents containing text and metadata.
    """
    documents: list[JSONRecord] = []

    for event in events:
        period = int(
            event.get("period", 0)
        )

        if (
            not include_shootout
            and period == 5
        ):
            continue

        event_type = _get_nested_name(
            event,
            "type",
        )

        if (
            event_type
            in DEFAULT_IGNORED_EVENT_TYPES
        ):
            continue

        document = {
            "event_id": event.get("id"),
            "index": event.get("index"),
            "period": period,
            "minute": event.get("minute"),
            "second": event.get("second"),
            "event_type": event_type,
            "team": _get_nested_name(
                event,
                "team",
            ),
            "player": _get_nested_name(
                event,
                "player",
                default=None,
            ),
            "possession_id": event.get(
                "possession"
            ),
            "possession_team": _get_nested_name(
                event,
                "possession_team",
            ),
            "play_pattern": _get_nested_name(
                event,
                "play_pattern",
            ),
            "text": format_event(
                event,
                include_location=include_location,
            ),
        }

        documents.append(document)

    return documents


def build_possession_documents(
    events: JSONRecords,
    include_location: bool = False,
    include_shootout: bool = False,
) -> list[JSONRecord]:
    """
    Group StatsBomb events into possession-level text documents.

    Events sharing the same StatsBomb possession identifier are grouped
    together and converted into a coherent textual sequence.

    Possession-level documents are a first candidate representation for
    semantic retrieval because they preserve the local football context
    around an action instead of treating every event independently.

    Parameters
    ----------
    events : JSONRecords
        Raw StatsBomb event records.

    include_location : bool, default=False
        Whether spatial coordinates should be included in event text.

    include_shootout : bool, default=False
        Whether penalty-shootout possessions should be included.

    Returns
    -------
    list[JSONRecord]
        Possession documents with text and structured metadata.
    """
    possessions: dict[int, JSONRecords] = defaultdict(list)

    for event in events:
        period = int(
            event.get("period", 0)
        )

        if (
            not include_shootout
            and period == 5
        ):
            continue

        possession_id = event.get(
            "possession"
        )

        if possession_id is None:
            continue

        possessions[
            int(possession_id)
        ].append(event)

    documents: list[JSONRecord] = []

    for possession_id, possession_events in possessions.items():

        possession_events = sorted(
            possession_events,
            key=lambda event: event.get(
                "index",
                0,
            ),
        )

        first_event = possession_events[0]
        last_event = possession_events[-1]

        narrative_events = [
            event
            for event in possession_events
            if _get_nested_name(
                event,
                "type",
            )
            not in DEFAULT_IGNORED_EVENT_TYPES
        ]

        if not narrative_events:
            continue

        possession_team = _get_nested_name(
            first_event,
            "possession_team",
        )

        play_pattern = _get_nested_name(
            first_event,
            "play_pattern",
        )

        start_time = _format_match_time(
            first_event
        )

        end_time = _format_match_time(
            last_event
        )

        sentences = [
            format_event(
                event,
                include_location=include_location,
            )
            for event in narrative_events
        ]

        header = (
            f"Possession {possession_id}. "
            f"{possession_team}. "
            f"Play pattern: {play_pattern}. "
            f"From {start_time} to {end_time}."
        )

        text = (
            header
            + "\n"
            + " ".join(sentences)
        )

        players = sorted({
            _get_nested_name(
                event,
                "player",
                default="",
            )
            for event in narrative_events
            if _get_nested_name(
                event,
                "player",
                default="",
            )
        })

        event_types = sorted({
            _get_nested_name(
                event,
                "type",
            )
            for event in narrative_events
        })

        document = {
            "possession_id": possession_id,
            "period": first_event.get(
                "period"
            ),
            "possession_team": possession_team,
            "play_pattern": play_pattern,
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
            "event_count": len(
                narrative_events
            ),
            "players": players,
            "event_types": event_types,
            "text": text,
        }

        documents.append(document)

    return documents