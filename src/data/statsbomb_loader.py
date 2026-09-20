"""
Utilities for loading local StatsBomb Open Data files.

This module centralizes file loading for MatchMind. It keeps data access
separate from football analytics so that the same loading functions can be
reused by notebooks, analytics modules, visualizations, and future AI tools.

Expected data directory:

    MatchMind/
    └── data/
        └── raw/
            └── statsbomb/
                ├── 3869685_events.json
                ├── 3869685_lineups.json
                ├── 3869685_360.json
                └── world_cup_2022_matches.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# MatchMind project root.
#
# This file is located at:
# MatchMind/src/data/statsbomb_loader.py
#
# parents[0] -> src/data
# parents[1] -> src
# parents[2] -> MatchMind
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default location of the local StatsBomb Open Data files.
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb"


# Type aliases used to make function signatures easier to read.
JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """
    Resolve the directory containing StatsBomb data.

    Parameters
    ----------
    data_dir : str | Path | None
        Optional custom data directory. If no directory is provided,
        MatchMind's default StatsBomb data directory is used.

    Returns
    -------
    Path
        Absolute path to the selected data directory.
    """
    if data_dir is None:
        return DEFAULT_DATA_DIR

    return Path(data_dir).expanduser().resolve()


def load_json(file_path: str | Path) -> JSONRecords:
    """
    Load a JSON file containing a list of records.

    Parameters
    ----------
    file_path : str | Path
        Path to the JSON file.

    Returns
    -------
    JSONRecords
        List of dictionaries loaded from the JSON file.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.

    ValueError
        If the file is not a JSON file or if its top-level structure
        is not a list.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"StatsBomb data file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected a JSON file, received: {path.name}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {path.name}, "
            f"received {type(data).__name__}."
        )

    return data


def load_events(
    match_id: int,
    data_dir: str | Path | None = None,
) -> JSONRecords:
    """
    Load StatsBomb event data for a match.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.

    data_dir : str | Path | None
        Optional custom StatsBomb data directory.

    Returns
    -------
    JSONRecords
        Match event records.
    """
    directory = _resolve_data_dir(data_dir)
    file_path = directory / f"{match_id}_events.json"

    return load_json(file_path)


def load_lineups(
    match_id: int,
    data_dir: str | Path | None = None,
) -> JSONRecords:
    """
    Load StatsBomb lineup data for a match.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.

    data_dir : str | Path | None
        Optional custom StatsBomb data directory.

    Returns
    -------
    JSONRecords
        Team and player lineup records.
    """
    directory = _resolve_data_dir(data_dir)
    file_path = directory / f"{match_id}_lineups.json"

    return load_json(file_path)


def load_360(
    match_id: int,
    data_dir: str | Path | None = None,
) -> JSONRecords:
    """
    Load StatsBomb 360 data for a match.

    StatsBomb 360 provides additional spatial context around supported
    events, such as visible players and freeze-frame information.

    Parameters
    ----------
    match_id : int
        StatsBomb match identifier.

    data_dir : str | Path | None
        Optional custom StatsBomb data directory.

    Returns
    -------
    JSONRecords
        StatsBomb 360 records.
    """
    directory = _resolve_data_dir(data_dir)
    file_path = directory / f"{match_id}_360.json"

    return load_json(file_path)


def load_matches(
    file_name: str = "world_cup_2022_matches.json",
    data_dir: str | Path | None = None,
) -> JSONRecords:
    """
    Load match-level metadata.

    Parameters
    ----------
    file_name : str
        Name of the local match metadata file.

    data_dir : str | Path | None
        Optional custom StatsBomb data directory.

    Returns
    -------
    JSONRecords
        Match metadata records.
    """
    directory = _resolve_data_dir(data_dir)
    file_path = directory / file_name

    return load_json(file_path)
