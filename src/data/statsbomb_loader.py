"""Load MatchMind's local StatsBomb Open Data JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"

JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the custom data directory, or MatchMind's default directory."""
    if data_dir is None:
        return DEFAULT_DATA_DIR
    return Path(data_dir).expanduser().resolve()


def load_json(file_path: str | Path) -> JSONRecords:
    """Load a JSON list from ``file_path``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the path is not JSON or its top-level value is not a list.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"StatsBomb data file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, received: {path.name}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path.name}, received {type(data).__name__}."
        )
    return data


def load_events(match_id: int, data_dir: str | Path | None = None) -> JSONRecords:
    """Load the event records for a StatsBomb match."""
    return load_json(_resolve_data_dir(data_dir) / f"{match_id}_events.json")


def load_lineups(match_id: int, data_dir: str | Path | None = None) -> JSONRecords:
    """Load the team and player lineups for a StatsBomb match."""
    return load_json(_resolve_data_dir(data_dir) / f"{match_id}_lineups.json")


def load_360(match_id: int, data_dir: str | Path | None = None) -> JSONRecords:
    """Load StatsBomb 360 spatial context for a match."""
    return load_json(_resolve_data_dir(data_dir) / f"{match_id}_360.json")


def load_matches(
    file_name: str = "world_cup_2022_matches.json",
    data_dir: str | Path | None = None,
) -> JSONRecords:
    """Load match-level metadata from the selected StatsBomb directory."""
    return load_json(_resolve_data_dir(data_dir) / file_name)
