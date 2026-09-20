"""Deterministic football analysis helpers for normalized StatsBomb events."""

from __future__ import annotations

from typing import Any

import pandas as pd

JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]

SHOT_COLUMNS = [
    "period",
    "minute",
    "second",
    "team.name",
    "player.name",
    "location",
    "shot.statsbomb_xg",
    "shot.outcome.name",
    "shot.body_part.name",
    "shot.technique.name",
]
SHOT_RENAMES = {
    "team.name": "team",
    "player.name": "player",
    "shot.statsbomb_xg": "xg",
    "shot.outcome.name": "outcome",
    "shot.body_part.name": "body_part",
    "shot.technique.name": "technique",
}
SUBSTITUTION_COLUMNS = [
    "period",
    "minute",
    "second",
    "team.name",
    "player.name",
    "substitution.replacement.name",
]
PASS_COLUMNS = [
    "period",
    "minute",
    "second",
    "team.name",
    "player.name",
    "pass.recipient.name",
    "location",
    "pass.end_location",
    "pass.outcome.name",
]


def _matching_rows(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Return matching rows with a clean, zero-based index."""
    return frame.loc[frame[column] == value].copy().reset_index(drop=True)


def events_to_dataframe(events: JSONRecords) -> pd.DataFrame:
    """Flatten nested StatsBomb event records into a pandas DataFrame."""
    if not events:
        raise ValueError("The event list is empty.")
    return pd.json_normalize(events)


def get_match_shots(
    events_df: pd.DataFrame,
    include_shootout: bool = False,
) -> pd.DataFrame:
    """Return match shots, excluding period-five shootout attempts by default."""
    shots = events_df.loc[events_df["type.name"] == "Shot"].copy()
    if not include_shootout:
        shots = shots.loc[shots["period"] <= 4].copy()
    return shots[SHOT_COLUMNS].rename(columns=SHOT_RENAMES).reset_index(drop=True)


def get_player_shots(shots_df: pd.DataFrame, player: str) -> pd.DataFrame:
    """Return all shots taken by an exact StatsBomb player name."""
    return _matching_rows(shots_df, "player", player)


def get_team_shots(shots_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return all shots taken by an exact StatsBomb team name."""
    return _matching_rows(shots_df, "team", team)


def get_goals(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Return shots whose StatsBomb outcome is ``Goal``."""
    return _matching_rows(shots_df, "outcome", "Goal")


def get_team_xg(shots_df: pd.DataFrame) -> pd.Series:
    """Return total expected goals by team, highest first."""
    return shots_df.groupby("team")["xg"].sum().sort_values(ascending=False)


def get_player_xg(shots_df: pd.DataFrame) -> pd.Series:
    """Return total expected goals by player, highest first."""
    return shots_df.groupby("player")["xg"].sum().sort_values(ascending=False)


def get_shot_counts_by_team(shots_df: pd.DataFrame) -> pd.Series:
    """Return shot counts by team, highest first."""
    return shots_df.groupby("team").size().sort_values(ascending=False)


def get_shot_counts_by_player(shots_df: pd.DataFrame) -> pd.Series:
    """Return shot counts by player, highest first."""
    return shots_df.groupby("player").size().sort_values(ascending=False)


def get_substitutions(
    events_df: pd.DataFrame,
    team: str | None = None,
) -> pd.DataFrame:
    """Return substitutions for both teams or for one optional team."""
    substitutions = events_df.loc[
        events_df["type.name"] == "Substitution", SUBSTITUTION_COLUMNS
    ].rename(
        columns={
            "team.name": "team",
            "player.name": "player_out",
            "substitution.replacement.name": "player_in",
        }
    )
    if team is not None:
        substitutions = substitutions.loc[substitutions["team"] == team]
    return substitutions.reset_index(drop=True)


def get_passes(events_df: pd.DataFrame, team: str | None = None) -> pd.DataFrame:
    """Return passes for both teams or for one optional team."""
    passes = events_df.loc[events_df["type.name"] == "Pass", PASS_COLUMNS].rename(
        columns={
            "team.name": "team",
            "player.name": "player",
            "pass.recipient.name": "recipient",
            "pass.end_location": "end_location",
            "pass.outcome.name": "outcome",
        }
    )
    if team is not None:
        passes = passes.loc[passes["team"] == team]
    return passes.reset_index(drop=True)
