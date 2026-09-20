"""
Football analytics utilities for MatchMind.

This module contains deterministic analysis functions operating on
StatsBomb event data.

The functions in this file do not load files directly. Data loading is
handled by ``src.data.statsbomb_loader``. Keeping loading and analysis
separate makes the project easier to test, maintain, and extend.

Typical workflow
----------------
1. Load raw StatsBomb events.
2. Convert the events into a normalized Pandas DataFrame.
3. Filter or aggregate specific football event types.
4. Return structured results that can later be used by visualizations,
   RAG components, or AI agent tools.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


JSONRecord = dict[str, Any]
JSONRecords = list[JSONRecord]


def events_to_dataframe(events: JSONRecords) -> pd.DataFrame:
    """
    Convert raw StatsBomb event records into a normalized DataFrame.

    StatsBomb events contain nested dictionaries. ``pd.json_normalize``
    flattens these nested structures into columns such as:

        team.name
        player.name
        type.name
        shot.statsbomb_xg

    Parameters
    ----------
    events : JSONRecords
        Raw StatsBomb event records.

    Returns
    -------
    pd.DataFrame
        Normalized event data.

    Raises
    ------
    ValueError
        If no events are provided.
    """
    if not events:
        raise ValueError("The event list is empty.")

    return pd.json_normalize(events)


def get_match_shots(
    events_df: pd.DataFrame,
    include_shootout: bool = False,
) -> pd.DataFrame:
    """
    Extract shot events from the match.

    By default, penalty-shootout attempts are excluded because they
    should not normally be mixed with shots produced during regular
    time or extra time.

    StatsBomb periods:
        1 -> first half
        2 -> second half
        3 -> first half of extra time
        4 -> second half of extra time
        5 -> penalty shootout

    Parameters
    ----------
    events_df : pd.DataFrame
        Normalized StatsBomb event data.

    include_shootout : bool, default=False
        Whether period 5 penalty-shootout attempts should be included.

    Returns
    -------
    pd.DataFrame
        Clean shot-level table.
    """
    shots = events_df[
        events_df["type.name"] == "Shot"
    ].copy()

    if not include_shootout:
        shots = shots[
            shots["period"] <= 4
        ].copy()

    shot_columns = [
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

    shots = shots[shot_columns].rename(
        columns={
            "team.name": "team",
            "player.name": "player",
            "shot.statsbomb_xg": "xg",
            "shot.outcome.name": "outcome",
            "shot.body_part.name": "body_part",
            "shot.technique.name": "technique",
        }
    )

    return shots.reset_index(drop=True)


def get_player_shots(
    shots_df: pd.DataFrame,
    player: str,
) -> pd.DataFrame:
    """
    Return all shots taken by a specific player.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame returned by ``get_match_shots``.

    player : str
        Exact StatsBomb player name.

    Returns
    -------
    pd.DataFrame
        Shots taken by the selected player.
    """
    player_shots = shots_df[
        shots_df["player"] == player
    ].copy()

    return player_shots.reset_index(drop=True)


def get_team_shots(
    shots_df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Return all shots taken by a specific team.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame returned by ``get_match_shots``.

    team : str
        Exact StatsBomb team name.

    Returns
    -------
    pd.DataFrame
        Shots taken by the selected team.
    """
    team_shots = shots_df[
        shots_df["team"] == team
    ].copy()

    return team_shots.reset_index(drop=True)


def get_goals(
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract goals from a shot-level DataFrame.

    A StatsBomb shot is considered a goal when:

        outcome == "Goal"

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    Returns
    -------
    pd.DataFrame
        Goal events.
    """
    goals = shots_df[
        shots_df["outcome"] == "Goal"
    ].copy()

    return goals.reset_index(drop=True)


def get_team_xg(
    shots_df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate total expected goals (xG) for each team.

    The result is computed by summing the StatsBomb xG values of all
    shots contained in ``shots_df``.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    Returns
    -------
    pd.Series
        Total xG indexed by team, sorted from highest to lowest.
    """
    return (
        shots_df
        .groupby("team")["xg"]
        .sum()
        .sort_values(ascending=False)
    )


def get_player_xg(
    shots_df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate total expected goals (xG) for each player.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    Returns
    -------
    pd.Series
        Total xG indexed by player, sorted from highest to lowest.
    """
    return (
        shots_df
        .groupby("player")["xg"]
        .sum()
        .sort_values(ascending=False)
    )


def get_shot_counts_by_team(
    shots_df: pd.DataFrame,
) -> pd.Series:
    """
    Count the number of shots taken by each team.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    Returns
    -------
    pd.Series
        Shot counts indexed by team.
    """
    return (
        shots_df
        .groupby("team")
        .size()
        .sort_values(ascending=False)
    )


def get_shot_counts_by_player(
    shots_df: pd.DataFrame,
) -> pd.Series:
    """
    Count the number of shots taken by each player.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    Returns
    -------
    pd.Series
        Shot counts indexed by player.
    """
    return (
        shots_df
        .groupby("player")
        .size()
        .sort_values(ascending=False)
    )


def get_substitutions(
    events_df: pd.DataFrame,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Extract substitution events.

    In StatsBomb data:

        player.name
            -> player leaving the pitch

        substitution.replacement.name
            -> player entering the pitch

    Parameters
    ----------
    events_df : pd.DataFrame
        Normalized StatsBomb event data.

    team : str | None, default=None
        Optional team filter. If no team is provided, substitutions
        from both teams are returned.

    Returns
    -------
    pd.DataFrame
        Clean substitution table.
    """
    substitutions = events_df[
        events_df["type.name"] == "Substitution"
    ].copy()

    substitution_columns = [
        "period",
        "minute",
        "second",
        "team.name",
        "player.name",
        "substitution.replacement.name",
    ]

    substitutions = substitutions[
        substitution_columns
    ].rename(
        columns={
            "team.name": "team",
            "player.name": "player_out",
            "substitution.replacement.name": "player_in",
        }
    )

    if team is not None:
        substitutions = substitutions[
            substitutions["team"] == team
        ]

    return substitutions.reset_index(drop=True)


def get_passes(
    events_df: pd.DataFrame,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Extract pass events from normalized StatsBomb data.

    Parameters
    ----------
    events_df : pd.DataFrame
        Normalized StatsBomb event data.

    team : str | None, default=None
        Optional team filter.

    Returns
    -------
    pd.DataFrame
        Clean pass-level table.
    """
    passes = events_df[
        events_df["type.name"] == "Pass"
    ].copy()

    pass_columns = [
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

    passes = passes[
        pass_columns
    ].rename(
        columns={
            "team.name": "team",
            "player.name": "player",
            "pass.recipient.name": "recipient",
            "pass.end_location": "end_location",
            "pass.outcome.name": "outcome",
        }
    )

    if team is not None:
        passes = passes[
            passes["team"] == team
        ]

    return passes.reset_index(drop=True)