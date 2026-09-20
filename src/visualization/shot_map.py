"""
Shot-map visualization utilities for MatchMind.

This module contains reusable Matplotlib functions for visualizing
StatsBomb shot data.

The functions expect shot-level DataFrames produced by
``src.analytics.match_analysis.get_match_shots``.

Keeping visualization separate from data loading and analytics makes
the project easier to maintain and allows the same analytical results
to be displayed in notebooks, applications, or future AI agent tools.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


PITCH_LENGTH = 120
PITCH_WIDTH = 80


def add_shot_coordinates(
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract explicit x and y coordinates from StatsBomb shot locations.

    StatsBomb stores event locations as:

        [x, y]

    where the event coordinate system is approximately:

        x: 0 -> 120
        y: 0 -> 80

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame containing a ``location`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with two additional columns:

        - ``x``
        - ``y``

    Raises
    ------
    ValueError
        If the ``location`` column is missing or contains invalid
        coordinate values.
    """
    if "location" not in shots_df.columns:
        raise ValueError(
            "The shot DataFrame must contain a 'location' column."
        )

    shots = shots_df.copy()

    valid_locations = shots["location"].apply(
        lambda location: (
            isinstance(location, list)
            and len(location) >= 2
        )
    )

    if not valid_locations.all():
        raise ValueError(
            "At least one shot does not contain a valid [x, y] location."
        )

    shots["x"] = shots["location"].apply(
        lambda location: location[0]
    )

    shots["y"] = shots["location"].apply(
        lambda location: location[1]
    )

    return shots


def draw_pitch(
    ax: Axes,
) -> Axes:
    """
    Draw a simplified football pitch using StatsBomb coordinates.

    The pitch uses the same 120 x 80 coordinate space as the StatsBomb
    event locations used by MatchMind.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis on which the pitch should be drawn.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the pitch drawing.
    """
    # Pitch boundaries
    ax.plot([0, 0], [0, PITCH_WIDTH])
    ax.plot(
        [0, PITCH_LENGTH],
        [PITCH_WIDTH, PITCH_WIDTH],
    )
    ax.plot(
        [PITCH_LENGTH, PITCH_LENGTH],
        [PITCH_WIDTH, 0],
    )
    ax.plot([PITCH_LENGTH, 0], [0, 0])

    # Halfway line
    ax.plot(
        [PITCH_LENGTH / 2, PITCH_LENGTH / 2],
        [0, PITCH_WIDTH],
    )

    # Centre circle
    centre_circle = Circle(
        (60, 40),
        radius=10,
        fill=False,
    )
    ax.add_patch(centre_circle)

    # Centre spot
    ax.scatter(60, 40, s=10)

    # Left penalty area
    ax.add_patch(
        Rectangle(
            (0, 18),
            18,
            44,
            fill=False,
        )
    )

    # Right penalty area
    ax.add_patch(
        Rectangle(
            (102, 18),
            18,
            44,
            fill=False,
        )
    )

    # Left six-yard box
    ax.add_patch(
        Rectangle(
            (0, 30),
            6,
            20,
            fill=False,
        )
    )

    # Right six-yard box
    ax.add_patch(
        Rectangle(
            (114, 30),
            6,
            20,
            fill=False,
        )
    )

    # Penalty spots
    ax.scatter(
        [12, 108],
        [40, 40],
        s=10,
    )

    # Goals
    ax.plot(
        [0, 0],
        [36, 44],
        linewidth=4,
    )

    ax.plot(
        [120, 120],
        [36, 44],
        linewidth=4,
    )

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)

    ax.set_aspect("equal")
    ax.axis("off")

    return ax


def plot_shot_map(
    shots_df: pd.DataFrame,
    title: str = "Shot Map",
    attacking_half: bool = False,
    show_goals: bool = True,
    annotate_goals: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot a shot map from a MatchMind shot DataFrame.

    Marker position represents the shot location.

    Marker size represents StatsBomb xG.

    Different teams are displayed as separate plotting groups.

    Goals can optionally be highlighted using star markers.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame produced by ``get_match_shots``.

    title : str, default="Shot Map"
        Figure title.

    attacking_half : bool, default=False
        If True, display only the attacking half of the pitch.

    show_goals : bool, default=True
        If True, goal events are displayed using star markers.

    annotate_goals : bool, default=False
        If True, add scorer name and match minute next to each goal.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axis.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_columns = {
        "team",
        "player",
        "location",
        "xg",
        "outcome",
        "minute",
    }

    missing_columns = required_columns.difference(
        shots_df.columns
    )

    if missing_columns:
        raise ValueError(
            "The shot DataFrame is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    shots = add_shot_coordinates(shots_df)

    shots["is_goal"] = (
        shots["outcome"] == "Goal"
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    draw_pitch(ax)

    for team, team_shots in shots.groupby("team"):

        if show_goals:
            non_goals = team_shots[
                ~team_shots["is_goal"]
            ]

            goals = team_shots[
                team_shots["is_goal"]
            ]

            ax.scatter(
                non_goals["x"],
                non_goals["y"],
                s=non_goals["xg"] * 1000,
                alpha=0.55,
                label=f"{team} shots",
            )

            ax.scatter(
                goals["x"],
                goals["y"],
                s=goals["xg"] * 1000,
                marker="*",
                label=f"{team} goals",
            )

        else:
            ax.scatter(
                team_shots["x"],
                team_shots["y"],
                s=team_shots["xg"] * 1000,
                alpha=0.6,
                label=team,
            )

    if annotate_goals:
        goals = shots[
            shots["is_goal"]
        ]

        for _, goal in goals.iterrows():
            ax.text(
                goal["x"] - 2,
                goal["y"] - 2,
                f"{goal['player']} {goal['minute']}'",
                fontsize=8,
            )

    if attacking_half:
        ax.set_xlim(60, 122)
        ax.set_ylim(-2, 82)

    ax.legend()
    ax.set_title(
        title,
        fontsize=16,
    )

    return fig, ax


def plot_player_shots(
    shots_df: pd.DataFrame,
    player: str,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot all shots taken by a specific player.

    Parameters
    ----------
    shots_df : pd.DataFrame
        Shot-level DataFrame.

    player : str
        Exact StatsBomb player name.

    title : str | None, default=None
        Optional custom figure title.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axis.

    Raises
    ------
    ValueError
        If the player has no shots in the provided DataFrame.
    """
    player_shots = shots_df[
        shots_df["player"] == player
    ].copy()

    if player_shots.empty:
        raise ValueError(
            f"No shots found for player: {player}"
        )

    if title is None:
        title = f"{player} - Shot Map"

    return plot_shot_map(
        player_shots,
        title=title,
        attacking_half=True,
        show_goals=True,
        annotate_goals=True,
    )

