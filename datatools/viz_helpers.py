import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import axes, patches

import datatools.matplotsoccer as mps
from datatools import config, utils
from datatools.defcon import DEFCON


def plot_agg_scores(scores: pd.DataFrame, roles: List[str] = ["LB", "LCB", "RCB", "RB"], save_path: str = None):
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

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")


def plot_credit_timeline(
    defcon: DEFCON,
    threshold: float = 1e-5,
    fps: int = 25,
    minute_interval: int = 5,
    half_length_min: int = 45,
    zmin: float = -0.03,
    zmax: float = 0.03,
    colorscale: str = "RdBu",
    width: int = 1000,
    row_height: int = 28,
    min_height: int = 700,
) -> go.Figure:
    credits = defcon.credits[defcon.credits["player_credit"].abs() > threshold]
    pivot = credits.pivot_table("player_credit", "index", "defender")
    sorted_players = sorted(pivot.columns, key=utils.player_sort_key)

    action_frames = defcon.match.actions.loc[pivot.index, ["period_id", "frame_id", "synced_ts"]]
    actions_with_epv = pd.concat([action_frames, defcon.epv.loc[pivot.index]], axis=1)

    tracking = defcon.match.tracking
    period_ids = actions_with_epv["period_id"].unique()
    xticks, xlabels = [], []

    for period_id in period_ids:
        period_frames = tracking[tracking["period_id"] == period_id].index

        for minutes in np.arange(0, 100, minute_interval):
            if period_id < period_ids.max() and minutes >= half_length_min:
                continue

            target_frame = period_frames[0] + minutes * 60 * fps
            if target_frame > period_frames.max():
                continue

            target_xtick = np.searchsorted(action_frames["frame_id"], target_frame) - 0.5
            xticks.append(target_xtick)
            xlabels.append(f"{(period_id - 1) * half_length_min + minutes}'")

    n_players = len(sorted_players)
    ts_arr = np.tile(actions_with_epv["synced_ts"].astype(str).values, (n_players, 1))
    index_arr = np.tile(pivot.index.values, (n_players, 1))
    type_arr = np.tile(actions_with_epv["spadl_type"].astype(str).values, (n_players, 1))
    player_arr = np.tile(actions_with_epv["object_id"].astype(str).values, (n_players, 1))
    receiver_arr = np.tile(actions_with_epv["receiver_id"].astype(str).values, (n_players, 1))
    customdata = np.stack([ts_arr, index_arr, type_arr, player_arr, receiver_arr], axis=-1)

    hm = go.Heatmap(
        z=pivot[sorted_players].T.values,
        x=np.arange(pivot.shape[0]),
        y=sorted_players,
        zmin=zmin,
        zmax=zmax,
        colorscale=colorscale,
        reversescale=False,
        customdata=customdata,
        hovertemplate=(
            "Timestamp: %{customdata[0]}<br>"
            "Action index: %{customdata[1]}<br>"
            "Action type: %{customdata[2]}<br>"
            "Action player: %{customdata[3]}<br>"
            "Receiver: %{customdata[4]}<br>"
            "Defender: %{y}<br>"
            "Credit: %{z:.6f}<extra></extra>"
        ),
        hoverlabel=dict(font=dict(size=18)),
        showscale=True,
        colorbar=dict(
            thickness=25,
            len=1,
            outlinewidth=0,
            tickfont=dict(size=20),
            ticklen=8,
            ticks="outside",
        ),
    )
    fig = go.Figure(data=hm)

    fig.update_xaxes(
        title="Time (min)",
        title_font=dict(size=20),
        tickmode="array",
        tickvals=xticks,
        ticktext=xlabels,
        tickfont=dict(size=18),
        showgrid=True,
        gridcolor="gray",
    )
    fig.update_yaxes(
        autorange="reversed",
        title="Defender",
        title_font=dict(size=20),
        tickfont=dict(size=18),
        type="category",
        categoryorder="array",
        categoryarray=sorted_players,
        tickmode="linear",
        dtick=1,
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        width=width,
        height=max(min_height, row_height * n_players + 120),
        template="plotly_white",
        margin=dict(l=80, r=60, t=40, b=60),
    )

    return fig


def draw_pitch_heatmaps(defcon: DEFCON, threshold: float = 1e-5, save_path: str = None):
    credits = defcon.credits[defcon.credits["player_credit"].abs() > threshold].copy()
    credits["frame_id"] = defcon.match.actions.loc[credits["index"], "frame_id"].values
    credits["team"] = credits["defender"].str[:4]
    credits["defense_type"] = credits["defense_type"].apply(lambda x: x.split("_")[0]).replace("foul", "concede")
    tracking = defcon.match.tracking
    credits["x"] = credits.apply(lambda r: tracking.at[r["frame_id"], r["defender"] + "_x"], axis=1)
    credits["y"] = credits.apply(lambda r: tracking.at[r["frame_id"], r["defender"] + "_y"], axis=1)

    teams = ["home", "away"]
    defense_types = ["intercept", "disturb", "deter", "concede"]

    bins_x = np.linspace(0, config.FIELD_SIZE[0], 7)
    bins_y = np.linspace(0, config.FIELD_SIZE[1], 6)

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
            ax.set_title(f"{team.title()}-{def_type.title()}", fontdict={"size": 18})
            ax.set_axis_off()
            ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, drawedges=False)
    cbar.ax.tick_params(labelsize=18)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

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

    action_mask = team_credits["action_type"].isin(action_types)
    defense_mask = team_credits["defense_type"].isin(defense_types)
    filtered: pd.DataFrame = team_credits[action_mask & defense_mask].copy()

    mat = filtered.infer_objects(copy=False).pivot_table("player_credit", "defender", "attacker", "sum", fill_value=0.0)
    return mat.reindex(index=defenders, columns=attackers, fill_value=0.0).copy()


def draw_matrix_heatmap(matrix: pd.DataFrame, ax: axes.Axes, title: str = None, cbar: bool = True):
    sns.heatmap(matrix, ax=ax, cmap="RdBu", vmin=-0.2, vmax=0.2, cbar=cbar, xticklabels=True, yticklabels=True)
    ax.set_xlabel("Attacking player", fontsize=17)
    ax.set_ylabel("Defending player", fontsize=17)
    ax.tick_params(axis="both", labelsize=14)
    if cbar and ax.collections and ax.collections[-1].colorbar is not None:
        ax.collections[-1].colorbar.ax.tick_params(labelsize=14)
    if title is not None:
        ax.set_title(title, fontdict={"size": 18})


def draw_pairwise_matrix_heatmaps(
    defcon: DEFCON,
    team: str = None,
    threshold: float = 1e-5,
    exclude_keepers: bool = True,
    period_id: int = None,
    save_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if team is not None and team not in ["home", "away"]:
        raise ValueError("team must be None, 'home', or 'away'")

    credits = defcon.credits[defcon.credits["player_credit"].abs() > threshold].copy()
    credits["possessor"] = defcon.match.actions.loc[credits["index"], "object_id"].values
    credits["action_type"] = defcon.match.actions.loc[credits["index"], "action_type"].values
    credits["defense_type"] = credits["defense_type"].apply(lambda x: x.split("_")[0]).replace("foul", "concede")

    if exclude_keepers:
        credits = credits[~credits["defender"].isin(defcon.match.keepers)]
    if period_id is not None:
        period_actions = defcon.match.actions[defcon.match.actions["period_id"] == period_id]
        credits = credits[credits["index"].isin(period_actions.index)]

    teams = [team] if team is not None else ["home", "away"]
    action_types = ["pass", "dribble", "shot"]
    gain_types = ["intercept", "disturb", "deter"]
    lose_types = ["concede"]

    n_rows = len(teams)
    fig = plt.figure(figsize=(12, 5 * n_rows))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.02], wspace=0.08)
    inner_gs_data = outer_gs[0].subgridspec(n_rows, 2, wspace=0.4, hspace=0.6)
    axes = np.array([[fig.add_subplot(inner_gs_data[r, c]) for c in range(2)] for r in range(n_rows)])
    inner_gs_cbar = outer_gs[1].subgridspec(3, 1, height_ratios=[0.075, 0.85, 0.075])
    cax = fig.add_subplot(inner_gs_cbar[1])

    gain_mats = []
    loss_mats = []

    for i, home_away in enumerate(teams):
        opponent = "away" if home_away == "home" else "home"
        team_gain = pairwise_matrix(credits, home_away, action_types, gain_types)
        team_loss = pairwise_matrix(credits, home_away, action_types, lose_types)
        title_gain = f"{home_away.title()} credit gains against {opponent.title()}"
        title_loss = f"{home_away.title()} penalties against {opponent.title()}"
        draw_matrix_heatmap(team_gain, axes[i, 0], title=title_gain, cbar=False)
        draw_matrix_heatmap(team_loss, axes[i, 1], title=title_loss, cbar=False)
        gain_mats.append(team_gain)
        loss_mats.append(team_loss)

    mappable = axes[0, 0].collections[0]
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False)
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()

    return gain_mats, loss_mats


def _find_label_position(anchor, arrow_samples, radius_h=9.0, radius_v=4.0, threshold=2.5):
    """
    Place the label above `anchor` by default; if a black arrow passes through
    that area, fall back to bottom/right/left whichever is clearest of arrows.
    """
    ax_x, ax_y = anchor
    top = (ax_x, ax_y + radius_v)
    fallback = [
        (ax_x, ax_y - radius_v),  # bottom
        (ax_x + radius_h, ax_y),  # right
        (ax_x - radius_h, ax_y),  # left
    ]

    def min_dist(point):
        if not arrow_samples:
            return np.inf
        return min(np.hypot(point[0] - sx, point[1] - sy) for sx, sy in arrow_samples)

    if min_dist(top) >= threshold:
        return top
    return max(fallback, key=min_dist)


def draw_player_focused_credits(
    matrix: pd.DataFrame,
    tracking: pd.DataFrame,
    focused_defender: str,
    min_credit=0.01,
    xlim: tuple = None,
    save_path: str = None,
):
    focused_credits = matrix.loc[focused_defender].abs()
    focused_attackers = focused_credits[focused_credits > min_credit].index.tolist()

    xy_cols = [c for c in tracking.columns if re.fullmatch(r"(home|away)_\d+_(x|y)", c)]
    mean_xy = tracking[xy_cols].mean().rename("value").reset_index()
    mean_xy["object_id"] = mean_xy["index"].str[:-2]
    mean_xy["axis_type"] = mean_xy["index"].str[-1]
    mean_xy = mean_xy.pivot_table("value", "object_id", "axis_type").drop("away_4")

    color_dict = {"home": "tab:red", "away": "tab:blue"}

    base_figsize = (9, 6)
    pitch_margin_x = 6
    if xlim is not None:
        full_x_range = config.FIELD_SIZE[0] + 2 * pitch_margin_x
        fig_width = base_figsize[0] * (xlim[1] - xlim[0]) / full_x_range
        figsize = (fig_width, base_figsize[1])
    else:
        figsize = base_figsize

    fig, ax = plt.subplots(figsize=figsize)
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 14})
    mps.field("green", config.FIELD_SIZE[0], config.FIELD_SIZE[1], fig, ax, show=False)

    attackers = [p for p in matrix.columns if p in mean_xy.index]
    defenders = [p for p in matrix.index if p in mean_xy.index]

    max_edge_val = matrix.loc[defenders, attackers].abs().values.max()
    max_node_val = focused_credits.max()

    arrow_samples = []
    fd_x, fd_y = mean_xy.loc[focused_defender, ["x", "y"]]
    for a in attackers:
        if abs(matrix.at[focused_defender, a]) >= min_credit:
            ax_x, ay_y = mean_xy.loc[a, ["x", "y"]]
            arrow_samples.extend(
                (ax_x + t * (fd_x - ax_x), ay_y + t * (fd_y - ay_y)) for t in np.linspace(0.05, 0.95, 10)
            )

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
                    text_xy = _find_label_position((x1, y1), arrow_samples)
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

    if xlim is not None:
        ax.set_xlim(xlim)

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight")

    plt.show()
