import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axes, patches

import datatools.matplotsoccer as mps
from datatools import config, utils


def plot_player_scores(scores: pd.DataFrame, roles: List[str] = ["LB", "LCB", "RCB", "RB"]):
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 5))

    scores = scores[scores["position"].isin(roles)].copy()
    scores["team"] = pd.Categorical(scores["object_id"].str[:4], categories=["home", "away"], ordered=True)
    scores["position"] = pd.Categorical(scores["position"], categories=roles, ordered=True)
    scores = scores.sort_values(["team", "position", "mins_played"], ascending=[True, True, False])

    selected_scores = scores.drop_duplicates(subset=["team", "position"], keep="first").copy()
    x = np.arange(len(selected_scores))

    action = selected_scores[["intercept_pass", "intercept_shot"]].sum(axis=1).values
    disturb = selected_scores[["disturb_pass", "disturb_shot"]].sum(axis=1).values
    deter = selected_scores["deter_pass"].values
    concede = selected_scores[["concede_pass", "concede_shot", "foul"]].sum(axis=1).values

    mins_played = selected_scores["mins_played"].values
    action = action / mins_played * 90
    disturb = disturb / mins_played * 90
    deter = deter / mins_played * 90
    concede = concede / mins_played * 90

    offset = 0.15
    width = 0.3
    ax.bar(x - offset, action, width=width, label="Intercept")
    ax.bar(x - offset, disturb, width=width, bottom=action, label="Disturb")
    ax.bar(x - offset, deter, width=width, bottom=action + disturb, label="Deter")
    ax.bar(x - offset, concede, width=width, label="Concede")
    ax.bar(x + offset, selected_scores["score"].values, width=width, label="Net Credit")

    text_y = ax.get_ylim()[1] * 1.08
    home_count = (selected_scores["team"] == "home").sum()
    away_count = (selected_scores["team"] == "away").sum()
    if home_count:
        ax.text((home_count - 1) / 2, text_y, "Home", fontweight="bold", ha="center", va="center")
    if away_count:
        ax.text(home_count + (away_count - 1) / 2, text_y, "Away", fontweight="bold", ha="center", va="center")

    ax.axhline(0, color="black", linewidth=1)
    if home_count and away_count:
        ax.axvline(home_count - 0.5, color="black", linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(selected_scores["position"].astype(str).values)
    ax.set_xlabel("Role")
    ax.set_ylabel("Credit")
    ax.grid(axis="y")
    ax.legend()


def draw_pitch_heatmaps(credits: pd.DataFrame):
    teams = ["home", "away"]
    defense_types = ["intercept", "disturb", "deter", "concede"]

    bins_x = np.linspace(0, config.FIELD_SIZE[0], 7)
    bins_y = np.linspace(0, config.FIELD_SIZE[1], 6)

    credits = credits.copy()
    credits["ix"] = pd.cut(credits["x"], bins=bins_x, right=False, labels=range(6))
    credits["iy"] = pd.cut(credits["y"], bins=bins_y, right=False, labels=range(5))
    credits = credits.dropna(subset=["ix", "iy"])

    agg = credits.groupby(["team", "defense_type", "iy", "ix"], observed=False)["player_credit"].sum().reset_index()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5), constrained_layout=True)

    for r, team in enumerate(teams):
        for c, def_type in enumerate(defense_types):
            filtered = agg[(agg["team"] == team) & (agg["defense_type"] == def_type)]
            heatmap = filtered.set_index(["iy", "ix"])["player_credit"].unstack(fill_value=0).astype(float)

            ax = axes[r, c]
            mps.field("white", config.FIELD_SIZE[0], config.FIELD_SIZE[1], fig, ax, show=False)
            im = ax.imshow(
                heatmap.values,
                origin="lower",
                cmap="RdBu",
                vmin=-0.5,
                vmax=0.5,
                extent=(0, config.FIELD_SIZE[0], 0, config.FIELD_SIZE[1]),
                aspect="equal",
                zorder=-100,
            )
            ax.set_title(f"{team.title()} - {def_type.title()}", fontdict={"size": 15})
            ax.set_axis_off()
            ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, drawedges=False)
    cbar.ax.tick_params(labelsize=14)

    plt.show()


def pairwise_matrix(
    credits: pd.DataFrame,
    team: str,
    action_types: list = None,
    defense_types: list = None,
) -> pd.DataFrame:
    action_types = action_types or ["pass", "dribble", "shot"]
    defense_types = defense_types or ["intercept", "disturb", "deter", "concede"]

    team_credits = credits[credits["defender"].str[:4] == team].copy()
    team_credits["attacker"] = np.where(
        team_credits["option"].str.endswith("goal"),
        team_credits["possessor"],
        team_credits["option"],
    )
    defenders = sorted(team_credits["defender"].unique(), key=utils.player_sort_key)
    attackers = sorted(team_credits["attacker"].unique(), key=utils.player_sort_key)

    filtered = team_credits[
        (team_credits["action_type"].isin(action_types)) & (team_credits["defense_type"].isin(defense_types))
    ].copy()

    mat = filtered.infer_objects(copy=False).pivot_table("player_credit", "defender", "attacker", "sum", fill_value=0.0)
    return mat.reindex(index=defenders, columns=attackers, fill_value=0.0).copy()


def draw_matrix_heatmap(matrix: pd.DataFrame, ax: axes.Axes, title: str = None):
    sns.heatmap(matrix, ax=ax, cmap="RdBu", vmin=-0.2, vmax=0.2)
    ax.set_xlabel("Attacking player")
    ax.set_ylabel("Defending player")
    if title is not None:
        ax.set_title(title, fontdict={"size": 14})


def draw_pairwise_matrix_heatmaps(credits: pd.DataFrame, team: str = None) -> List[pd.DataFrame]:
    if team is not None and team not in ["home", "away"]:
        raise ValueError("team must be None, 'home', or 'away'")

    teams = [team] if team is not None else ["home", "away"]
    action_types = ["pass", "dribble", "shot"]
    gain_types = ["intercept", "disturb", "deter"]
    lose_types = ["concede"]

    fig, axes = plt.subplots(len(teams), 2, figsize=(12, 5 * len(teams)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    gain_mats = []
    loss_mats = []

    for i, home_away in enumerate(teams):
        opponent = "away" if home_away == "home" else "home"
        team_gain = pairwise_matrix(credits, home_away, action_types, gain_types)
        team_loss = pairwise_matrix(credits, home_away, action_types, lose_types)
        draw_matrix_heatmap(team_gain, axes[i, 0], f"{home_away.title()} credit gains against {opponent.title()}")
        draw_matrix_heatmap(team_loss, axes[i, 1], f"{home_away.title()} penalties against {opponent.title()}")
        gain_mats.append(team_gain)
        loss_mats.append(team_loss)

    plt.show()

    return gain_mats, loss_mats


def draw_player_focused_credits(matrix: pd.DataFrame, tracking: pd.DataFrame, focused_defender: str, min_credit=0.01):
    focused_credits = matrix.loc[focused_defender].abs()
    focused_attackers = focused_credits[focused_credits > min_credit].index.tolist()

    xy_cols = [c for c in tracking.columns if re.fullmatch(r"(home|away)_\d+_(x|y)", c)]
    mean_xy = tracking[xy_cols].mean().rename("value").reset_index()
    mean_xy["object_id"] = mean_xy["index"].str[:-2]
    mean_xy["axis_type"] = mean_xy["index"].str[-1]
    mean_xy = mean_xy.pivot_table("value", "object_id", "axis_type").drop("away_4")

    color_dict = {"home": "tab:red", "away": "tab:blue"}

    fig, ax = plt.subplots(figsize=(9, 6))
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    mps.field("green", config.FIELD_SIZE[0], config.FIELD_SIZE[1], fig, ax, show=False)

    attackers = [p for p in matrix.columns if p in mean_xy.index]
    defenders = [p for p in matrix.index if p in mean_xy.index]

    max_edge_val = matrix.loc[defenders, attackers].abs().values.max()

    edge_players = set()
    color = "black"

    for a in attackers:
        for d in defenders:
            value = round(matrix.at[d, a], 3)

            if abs(value) >= min_credit:
                edge_players.update([a, d])
                x1, y1 = mean_xy.loc[a, ["x", "y"]]
                x2, y2 = mean_xy.loc[d, ["x", "y"]]

                if d == focused_defender:
                    arrow_ec, arrow_z, alpha = color, 12, 1
                    text_xy = (x1, y1 - 5) if a in ["home_8"] else (x1, y1 + 4)
                    # text_xy = (x1 - 4, y1 - 5) if a == "home_3" else (x1 + 9, y1)
                    ax.annotate(
                        value,
                        xy=text_xy,
                        ha="center",
                        va="center",
                        color="k",
                        fontsize=14,
                        fontweight="bold",
                        zorder=30,
                    )

                    arrow_lw = (6 * abs(value)) / max_edge_val
                    arrow = patches.FancyArrowPatch(
                        (x1, y1),
                        (x2, y2),
                        arrowstyle="simple",
                        mutation_scale=20,
                        linewidth=arrow_lw,
                        color=arrow_ec,
                        alpha=alpha,
                        zorder=arrow_z,
                    )
                    ax.add_patch(arrow)
                # else:
                #     arrow_ec, arrow_z, alpha = "gray", 11, 0.7

    max_node_val = focused_credits.max()

    for p in mean_xy.index:
        player_num = p.split("_")[-1]
        x, y = mean_xy.loc[p, ["x", "y"]].values

        node_z = 20 if p in attackers else 10

        if p in focused_attackers:
            node_ec = color
            node_lw = (focused_credits.get(p, 0) / max_node_val) * 5 + 2
        else:
            node_ec = None
            node_lw = 0

        ax.scatter(
            x,
            y,
            s=500,
            facecolors=color_dict[p[:4]],
            edgecolors=node_ec,
            linewidths=node_lw,
            zorder=node_z,
        )
        ax.annotate(
            player_num,
            xy=(x, y),
            ha="center",
            va="center",
            color="w",
            fontsize=15,
            fontweight="bold",
            zorder=node_z + 1,
        )

    plt.title(f"Penalty imposed on {focused_defender} from each attacker")
    plt.show()
