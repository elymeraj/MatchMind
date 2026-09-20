"""Convert raw StatsBomb events into text documents for semantic search."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]

# These events add little meaning to a possession narrative or duplicate
# another action already present in the same sequence.
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
    default: Any = "Unknown",
) -> Any:
    """Read ``record[key]['name']``, returning ``default`` when unavailable."""
    value = record.get(key)
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"]
    return default


def _format_match_time(event: JSONRecord) -> str:
    """Format a StatsBomb minute and second as ``MM:SS``."""
    return f"{int(event.get('minute', 0))}:{int(event.get('second', 0)):02d}"


def _format_location(location: Any) -> str | None:
    """Format a StatsBomb ``[x, y]`` location, or return ``None`` if invalid."""
    if isinstance(location, list) and len(location) >= 2:
        return f"({location[0]:.1f}, {location[1]:.1f})"
    return None


def format_event(event: JSONRecord, include_location: bool = False) -> str:
    """Turn one StatsBomb event into a short human-readable sentence."""
    event_type = _get_nested_name(event, "type")
    team = _get_nested_name(event, "team")
    player = _get_nested_name(event, "player", default="Unknown player")
    prefix = f"{_format_match_time(event)} - {team}: "

    location_text = ""
    if include_location:
        location = _format_location(event.get("location"))
        if location is not None:
            location_text = f" from location {location}"

    if event_type == "Pass":
        pass_data = event.get("pass", {})
        recipient = _get_nested_name(pass_data, "recipient", default="a teammate")
        outcome = _get_nested_name(pass_data, "outcome", default="Complete")
        if outcome == "Complete":
            sentence = f"{player} completes a pass to {recipient}"
        else:
            sentence = (
                f"{player} attempts a pass to {recipient}, but the outcome is {outcome}"
            )
    elif event_type == "Carry":
        sentence = f"{player} carries the ball"
        if include_location:
            end_location = _format_location(event.get("carry", {}).get("end_location"))
            if end_location is not None:
                sentence += f"{location_text} to {end_location}"
                location_text = ""
    elif event_type == "Shot":
        shot_data = event.get("shot", {})
        xg = shot_data.get("statsbomb_xg")
        outcome = _get_nested_name(shot_data, "outcome", default="Unknown outcome")
        sentence = f"{player} takes a shot"
        if isinstance(xg, (int, float)):
            sentence += f" with an xG of {xg:.3f}"
        if outcome == "Goal":
            sentence += " and scores"
        else:
            sentence += f". The shot outcome is {outcome}"
    elif event_type == "Dribble":
        outcome = _get_nested_name(
            event.get("dribble", {}), "outcome", default="Unknown outcome"
        )
        sentence = f"{player} attempts a dribble with outcome {outcome}"
    elif event_type == "Ball Recovery":
        sentence = f"{player} recovers the ball"
    elif event_type == "Interception":
        sentence = f"{player} makes an interception"
    elif event_type == "Pressure":
        sentence = f"{player} applies pressure"
    elif event_type == "Miscontrol":
        sentence = f"{player} miscontrols the ball"
    elif event_type == "Clearance":
        sentence = f"{player} clears the ball"
    elif event_type == "Foul Won":
        sentence = f"{player} wins a foul"
    elif event_type == "Foul Committed":
        sentence = f"{player} commits a foul"
    elif event_type == "Duel":
        sentence = f"{player} is involved in a duel"
    elif event_type == "Substitution":
        replacement = _get_nested_name(
            event.get("substitution", {}), "replacement", default="Unknown player"
        )
        sentence = f"{player} is replaced by {replacement}"
    elif event_type == "Goal Keeper":
        action = _get_nested_name(
            event.get("goalkeeper", {}), "type", default="goalkeeper action"
        )
        sentence = f"{player} performs a goalkeeper action: {action}"
    else:
        sentence = f"{player} performs a {event_type} event"

    if location_text:
        sentence += location_text
    return prefix + sentence + "."


def build_event_documents(
    events: JSONRecords,
    include_location: bool = False,
    include_shootout: bool = False,
) -> list[JSONRecord]:
    """Build one text document per useful event, with retrieval metadata."""
    documents: list[JSONRecord] = []
    for event in events:
        period = int(event.get("period", 0))
        if not include_shootout and period == 5:
            continue

        event_type = _get_nested_name(event, "type")
        if event_type in DEFAULT_IGNORED_EVENT_TYPES:
            continue

        documents.append(
            {
                "event_id": event.get("id"),
                "index": event.get("index"),
                "period": period,
                "minute": event.get("minute"),
                "second": event.get("second"),
                "event_type": event_type,
                "team": _get_nested_name(event, "team"),
                "player": _get_nested_name(event, "player", default=None),
                "possession_id": event.get("possession"),
                "possession_team": _get_nested_name(event, "possession_team"),
                "play_pattern": _get_nested_name(event, "play_pattern"),
                "text": format_event(event, include_location=include_location),
            }
        )
    return documents


def build_possession_documents(
    events: JSONRecords,
    include_location: bool = False,
    include_shootout: bool = False,
) -> list[JSONRecord]:
    """Group events by possession and build contextual text documents."""
    possessions: dict[int, JSONRecords] = defaultdict(list)
    for event in events:
        period = int(event.get("period", 0))
        if not include_shootout and period == 5:
            continue
        possession_id = event.get("possession")
        if possession_id is not None:
            possessions[int(possession_id)].append(event)

    documents: list[JSONRecord] = []
    for possession_id, possession_events in possessions.items():
        possession_events = sorted(
            possession_events, key=lambda event: event.get("index", 0)
        )
        first_event = possession_events[0]
        last_event = possession_events[-1]
        narrative_events = [
            event
            for event in possession_events
            if _get_nested_name(event, "type") not in DEFAULT_IGNORED_EVENT_TYPES
        ]
        if not narrative_events:
            continue

        possession_team = _get_nested_name(first_event, "possession_team")
        play_pattern = _get_nested_name(first_event, "play_pattern")
        sentences = [
            format_event(event, include_location=include_location)
            for event in narrative_events
        ]
        header = (
            f"Possession {possession_id}. {possession_team}. "
            f"Play pattern: {play_pattern}. "
            f"From {_format_match_time(first_event)} "
            f"to {_format_match_time(last_event)}."
        )
        players = sorted(
            {
                player
                for event in narrative_events
                if (player := _get_nested_name(event, "player", default=""))
            }
        )
        event_types = sorted(
            {_get_nested_name(event, "type") for event in narrative_events}
        )
        documents.append(
            {
                "possession_id": possession_id,
                "period": first_event.get("period"),
                "possession_team": possession_team,
                "play_pattern": play_pattern,
                "start_minute": first_event.get("minute"),
                "start_second": first_event.get("second"),
                "end_minute": last_event.get("minute"),
                "end_second": last_event.get("second"),
                "event_count": len(narrative_events),
                "players": players,
                "event_types": event_types,
                "text": header + "\n" + " ".join(sentences),
            }
        )
    return documents
