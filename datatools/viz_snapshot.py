import os
import sys
from typing import Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes

import datatools.matplotsoccer as mps
from datatools import config


class SnapshotVisualizer:
    def __init__(
        self,
        snapshot: pd.DataFrame,
        ball_xy: pd.DataFrame = None,
        player_sizes: pd.Series = None,
        player_colors: pd.Series = None,
        player_annots: pd.Series = None,
        player_marks: pd.DataFrame | Dict[str, List[str]] = None,
        edges: np.ndarray = None,
        arrows: List[Tuple[str, str]] = None,
        heatmap: np.ndarray = None,
        show_velocities=True,
        unnormalize=False,
    ):
        self.snapshot = snapshot.copy()
        self.sizes = player_sizes
        self.colors = player_colors
        self.annots = player_annots
        self.marks = player_marks
        self.ball_xy = ball_xy
        self.edges = edges
        self.arrows = arrows
        self.heatmap = heatmap
        self.show_velocities = show_velocities

        if unnormalize:
            x_cols = [c for c in self.snapshot.columns if c.endswith("_x")]
            y_cols = [c for c in self.snapshot.columns if c.endswith("_y")]
            self.snapshot[x_cols] = (self.snapshot[x_cols] + 1) * config.FIELD_SIZE[0] / 2
            self.snapshot[y_cols] = (self.snapshot[y_cols] + 1) * config.FIELD_SIZE[1] / 2

    def plot_team_players(
        self,
        ax: axes.Axes,
        xy: pd.DataFrame,
        sizes: np.ndarray,
        colors: np.ndarray,
        anonymize=True,
        annotate=None,
    ):
        n_features = len(np.unique([c.split("_")[-1] for c in xy.columns]))  # ["x", "y", "vx", "vy"] for each player
        x = xy[xy.columns[0::n_features]].values[-1]
        y = xy[xy.columns[1::n_features]].values[-1]

        players = [c[:-2] for c in xy.columns[0::n_features]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))

        if self.marks is None:
            ax.scatter(x, y, s=sizes, c=colors, zorder=2)

        else:
            linewidth = 3
            player_zorder = 2

            if isinstance(self.marks, pd.DataFrame):
                self.marks = self.marks[self.marks["object_id"].str.startswith(players[0][0])]
                edgecolors = np.where(np.isin(players, self.marks["object_id"]), "gold", "none")

                for i in self.marks.index:
                    p = self.marks.at[i, "object_id"]
                    start_frame = self.marks.at[i, "start_frame_id"]
                    end_frame = self.marks.at[i, "end_frame_id"]
                    player_x = xy.loc[start_frame:end_frame, f"{p}_x"].values
                    player_y = xy.loc[start_frame:end_frame, f"{p}_y"].values
                    ax.plot(player_x, player_y, c="gold", linewidth=5, zorder=2)

                    if annotate is not None:
                        text = self.marks.at[i, annotate]

                        dx = player_x[-1] - player_x[-4]
                        dy = player_y[-1] - player_y[-4]
                        norm = np.sqrt(dx**2 + dy**2)
                        text_xy = [player_x[-1] + dx / norm * 3, player_y[-1] + dy / norm * 3]

                        assert isinstance(colors, str)
                        text_color = colors.replace("tab:", "dark")

                        ax.annotate(text, xy=text_xy, ha="center", va="center", color=text_color, fontsize=24, zorder=3)

            else:
                # highlights = [p for p in self.highlights if p[0] == players[0][0]]
                edgecolors = np.array(["none"] * len(players), dtype=object)

                for color, players_to_highlight in self.marks.items():
                    teammates_to_highlight = [p for p in players_to_highlight if p[:4] == players[0][:4]]
                    edgecolors[np.isin(players, teammates_to_highlight)] = color

                    for p in teammates_to_highlight:
                        # If any player is highlighted in the team, place all teammates above their opponents
                        player_zorder = 3
                        player_x = xy[f"{p}_x"].values
                        player_y = xy[f"{p}_y"].values
                        ax.plot(player_x, player_y, c=color, linewidth=5, zorder=player_zorder)

            ax.scatter(x, y, s=sizes, c=colors, edgecolors=edgecolors, linewidths=linewidth, zorder=player_zorder)

        if self.show_velocities:
            assert n_features >= 4
            vx = xy[xy.columns[2::n_features]].values[-1]
            vy = xy[xy.columns[3::n_features]].values[-1]
            plt.quiver(
                x,
                y,
                vx * 1.5,
                vy * 1.5,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=colors,
                width=0.003,  # 0.006
                headwidth=5,
            )

        for i, p in enumerate(players):
            player_xy = xy[[f"{p}_x", f"{p}_y"]].values
            player_num = player_dict[p] if anonymize else int(p.split("_")[1])
            player_color = colors if isinstance(colors, str) else colors[i]

            ax.plot(player_xy[:, 0], player_xy[:, 1], c=player_color, ls="--", zorder=0)
            ax.annotate(
                player_num,
                xy=player_xy[-1],
                ha="center",
                va="center",
                color="w",
                fontsize=18,
                fontweight="bold",
                zorder=3,
            )

        return ax

    def plot(
        self,
        rotate_pitch=False,
        half=None,
        focus_xy: Tuple[float, float] = None,
        switch_teams=False,
        anonymize=False,
        annotate=None,
        smin=400,
        smax=2000,
        cmin=0,
        cmax=0.05,
        annot_type=None,
        hm_cmap="jet",
    ):
        if focus_xy is None:
            figsize = (13.5, 9.0) if half is None else (10.4, 14.4)  # (9, 6)
        else:
            figsize = (4, 5)

        if self.colors is not None:
            figsize = (figsize[0] * 1.2, figsize[1])

        fig, ax = plt.subplots(figsize=figsize)
        mps.field("green", config.FIELD_SIZE[0], config.FIELD_SIZE[1], fig, ax, show=False)

        if half == "left":
            ax.set_xlim(-4, config.FIELD_SIZE[0] / 2)
        elif half == "right":
            ax.set_xlim(config.FIELD_SIZE[0] / 2, config.FIELD_SIZE[0] + 4)

        if focus_xy is not None:
            ax.set_xlim(focus_xy[0] - 20, focus_xy[0] + 20)
            ax.set_ylim(focus_xy[1] - 20, focus_xy[1] + 20)

        snapshot = self.snapshot.dropna(axis=1, how="all")
        xy_cols = [c for c in snapshot.columns if c.split("_")[-1] in ["x", "y", "vx", "vy"]]
        xy_cols = [c for c in xy_cols if c.split("_")[1] != "goal"]

        if rotate_pitch:
            snapshot[xy_cols[0::4]] = config.FIELD_SIZE[0] - snapshot[xy_cols[0::4]]
            snapshot[xy_cols[1::4]] = config.FIELD_SIZE[1] - snapshot[xy_cols[1::4]]

        home_xy = snapshot[[c for c in xy_cols if c.startswith("home")]]
        away_xy = snapshot[[c for c in xy_cols if c.startswith("away")]]

        home_players = [c[:-2] for c in xy_cols[0::4] if c.startswith("home")]
        away_players = [c[:-2] for c in xy_cols[0::4] if c.startswith("away")]

        if self.sizes is not None and home_players[0] in self.sizes.index:
            home_sizes = self.sizes[home_players].values * (smax - smin) + smin
        else:
            home_sizes = 600

        if self.sizes is not None and away_players[0] in self.sizes.index:
            away_sizes = self.sizes[away_players].values * (smax - smin) + smin
        else:
            away_sizes = 600

        home_colors = "tab:blue" if switch_teams else "tab:red"
        away_colors = "tab:red" if switch_teams else "tab:blue"
        annot_args = {"ha": "center", "va": "center", "color": "k", "fontsize": 16, "fontweight": "bold", "zorder": 7}

        if self.annots is not None:
            if "scoring" in annot_type or "conceding" in annot_type or "epv" in annot_type:
                annots = self.annots[self.annots != 0.0]
            elif "select" in annot_type or "posterior" in annot_type:
                annots = self.annots[self.annots > 0.05]
            elif "credit" in annot_type:
                annots = self.annots[(self.annots > 0.001) | (self.annots < -0.001)]
            else:
                annots = self.annots[self.annots >= 0.005]

            for p, value in annots.items():
                if "credit" in annot_type and value > 0:
                    text = f"+{value:.3f}"
                elif "credit" in annot_type and value < 0:
                    text = f"-{-value:.3f}"
                else:
                    text = f"{value:.3f}"

                text_xy = snapshot[[f"{p}_x", f"{p}_y"]].values[-1]
                if not p.endswith("_goal"):
                    text_xy[1] += 2.5
                elif p == "home_goal":
                    text_xy[0] += 3
                elif p == "away_goal":
                    text_xy[0] -= 3
                ax.annotate(text, xy=text_xy, **annot_args)

        if self.colors is not None:
            if home_players[0] in self.colors.index:  # Use [0.7, 1] of jet as the colormap
                cmap = plt.get_cmap("jet_r") if switch_teams else plt.get_cmap("jet")
                cbound = 0.7
                scores = (self.colors[home_players].values - cmin) * (1 - cbound) / (cmax - cmin) + cbound
                scores = np.clip(scores, cbound, 1)
                cmap_segment = mpl.colors.LinearSegmentedColormap.from_list("cmap", cmap(np.linspace(cbound, 1, 256)))
                home_colors = cmap(scores)

            elif away_players[0] in self.colors.index:  # Use [0.6, 1] of jet_r as the colormap
                cmap = plt.get_cmap("jet") if switch_teams else plt.get_cmap("jet_r")
                cbound = 0.65
                scores = (self.colors[away_players].values - cmin) * (1 - cbound) / (cmax - cmin) + cbound
                scores = np.clip(scores, cbound, 1)
                cmap_segment = mpl.colors.LinearSegmentedColormap.from_list("cmap", cmap(np.linspace(cbound, 1, 256)))

                # cmap = plt.get_cmap("jet")
                # scores = (self.colors[away_players].values - cmin) * (0.4) / (cmax - cmin)
                # scores = np.clip(scores, 0, 0.4)
                # cmap_segment = mpl.colors.LinearSegmentedColormap.from_list("cmap", cmap(np.linspace(0, 0.4, 256)))
                away_colors = cmap(scores)

            norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_segment)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Defender responsibility", fontsize=25, rotation=-90, labelpad=25)
            cbar.ax.tick_params(labelsize=20)

        if self.edges is not None:
            for src, dst in self.edges:
                edge_x = snapshot[[f"{src}_x", f"{dst}_x"]].values[-1]
                edge_y = snapshot[[f"{src}_y", f"{dst}_y"]].values[-1]
                if src[:4] == "home" and dst[:4] == "home":
                    edgecolor = "tab:blue" if switch_teams else "tab:red"
                elif src[:4] == "away" and dst[:4] == "away":
                    edgecolor = "tab:red" if switch_teams else "tab:blue"
                else:
                    edgecolor = "dimgray"
                plt.plot(edge_x, edge_y, color=edgecolor, linewidth=2, alpha=0.5, zorder=-100)

        if self.arrows is not None:
            for src, dst in self.arrows:
                if isinstance(src, str):
                    src_x = snapshot[f"{src}_x"].values[-1]
                    src_y = snapshot[f"{src}_y"].values[-1]
                else:
                    src_x, src_y = src

                if isinstance(dst, str):
                    dst_x = snapshot[f"{dst}_x"].values[-1]
                    dst_y = snapshot[f"{dst}_y"].values[-1]
                else:
                    dst_x, dst_y = dst

                ax.arrow(
                    src_x,
                    src_y,
                    dst_x - src_x,
                    dst_y - src_y,
                    color="k",
                    width=0.5,
                    length_includes_head=True,
                    zorder=4,
                )

        if self.heatmap is not None:
            hm_extent = (0, config.FIELD_SIZE[0], 0, config.FIELD_SIZE[1])
            ax.imshow(self.heatmap, cmap=hm_cmap, vmin=cmin, vmax=cmax, extent=hm_extent, alpha=0.7, zorder=0)

        if len(home_xy.columns) > 0:
            self.plot_team_players(ax, home_xy, home_sizes, home_colors, anonymize, annotate)

        if len(away_xy.columns) > 0:
            self.plot_team_players(ax, away_xy, away_sizes, away_colors, anonymize, annotate)

        if self.ball_xy is not None:
            ball_x = self.ball_xy["ball_x"].values
            ball_y = self.ball_xy["ball_y"].values
            ax.scatter(ball_x[-1], ball_y[-1], s=200, c="w", edgecolors="k", marker="o", zorder=5)
            ax.plot(ball_x, ball_y, "k", zorder=3)

        elif "ball_x" in snapshot.columns:
            ball_x = snapshot["ball_x"].values
            ball_y = snapshot["ball_y"].values
            ax.scatter(ball_x[-1], ball_y[-1], s=200, c="w", edgecolors="k", marker="o", zorder=5)
            ax.plot(ball_x[-30:], ball_y[-30:], "k", zorder=3)

        plt.show()
