"""Reusable Matplotlib shot maps using StatsBomb's 120-by-80 pitch."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

PITCH_LENGTH = 120
PITCH_WIDTH = 80


def add_shot_coordinates(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``shots_df`` with ``x`` and ``y`` location columns.

    Raises:
        ValueError: If the location column is missing or contains invalid data.
    """
    if "location" not in shots_df.columns:
        raise ValueError("The shot DataFrame must contain a 'location' column.")

    shots = shots_df.copy()
    valid_locations = shots["location"].apply(
        lambda location: isinstance(location, list) and len(location) >= 2
    )
    if not valid_locations.all():
        raise ValueError("At least one shot does not contain a valid [x, y] location.")

    shots["x"] = shots["location"].apply(lambda location: location[0])
    shots["y"] = shots["location"].apply(lambda location: location[1])
    return shots


def draw_pitch(ax: Axes) -> Axes:
    """Draw a simple football pitch on ``ax`` in StatsBomb coordinates."""
    ax.plot([0, 0], [0, PITCH_WIDTH])
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH])
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [PITCH_WIDTH, 0])
    ax.plot([PITCH_LENGTH, 0], [0, 0])
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH])
    ax.add_patch(Circle((60, 40), radius=10, fill=False))
    ax.scatter(60, 40, s=10)
    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False))
    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False))
    ax.scatter([12, 108], [40, 40], s=10)
    ax.plot([0, 0], [36, 44], linewidth=4)
    ax.plot([120, 120], [36, 44], linewidth=4)
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
    """Plot shots by team, sizing markers by xG and highlighting goals."""
    required_columns = {"team", "player", "location", "xg", "outcome", "minute"}
    missing_columns = required_columns.difference(shots_df.columns)
    if missing_columns:
        raise ValueError(
            f"The shot DataFrame is missing required columns: {sorted(missing_columns)}"
        )

    shots = add_shot_coordinates(shots_df)
    shots["is_goal"] = shots["outcome"] == "Goal"
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    for team, team_shots in shots.groupby("team"):
        if show_goals:
            non_goals = team_shots.loc[~team_shots["is_goal"]]
            goals = team_shots.loc[team_shots["is_goal"]]
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
        for _, goal in shots.loc[shots["is_goal"]].iterrows():
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
    ax.set_title(title, fontsize=16)
    return fig, ax


def plot_player_shots(
    shots_df: pd.DataFrame,
    player: str,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot every shot by one player on the attacking half of the pitch."""
    player_shots = shots_df.loc[shots_df["player"] == player].copy()
    if player_shots.empty:
        raise ValueError(f"No shots found for player: {player}")
    return plot_shot_map(
        player_shots,
        title=title or f"{player} - Shot Map",
        attacking_half=True,
        show_goals=True,
        annotate_goals=True,
    )
