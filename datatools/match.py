import os
import re
import sys
from abc import ABC
from typing import Dict

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch

import datatools.preprocess as proc
from datatools import config, utils
from datatools.viz_snapshot import SnapshotVisualizer


class Match(ABC):
    def __init__(
        self,
        events: pd.DataFrame,
        tracking: pd.DataFrame,
        lineup: pd.DataFrame,
        action_type: str = "all",
        fps: int = 25,
        include_keepers: bool = True,
        include_goals: bool = False,
    ):
        self.events = events.copy()
        self.tracking = tracking.copy()
        self.lineup = lineup.copy()

        self.action_type = action_type
        self.fps = fps

        object_id_map = self.events.set_index("player_id")["object_id"].drop_duplicates()
        self.lineup["object_id"] = self.lineup["player_id"].map(object_id_map)
        self.lineup = self.lineup[config.LINEUP_HEADER].sort_values(["contestant_name", "shirt_number"])
        self.lineup = self.lineup.dropna(subset=["object_id"])
        self.keepers = self.lineup.loc[lineup["advanced_position"] == "goal_keeper", "object_id"].dropna().values

        if "episode_id" not in self.tracking.columns:
            self.tracking = proc.label_frames_and_episodes(self.tracking, self.fps).set_index("frame_id")

        if "phase_id" not in self.tracking.columns:
            self.phases = proc.summarize_phases(self.tracking, self.keepers)
            self.tracking["phase_id"] = 0

            for i in self.phases.index:
                start_frame = self.phases.at[i, "start_frame"]
                end_frame = self.phases.at[i, "end_frame"]
                self.tracking.loc[start_frame:end_frame, "phase_id"] = i

        if "start_x" not in self.events.columns:
            self.events["start_x"] = self.events.apply(
                lambda e: tracking.at[e["frame_id"], f"{e['object_id']}_x"] if not pd.isna(e["frame_id"]) else np.nan,
                axis=1,
            )
            self.events["start_y"] = self.events.apply(
                lambda e: tracking.at[e["frame_id"], f"{e['object_id']}_y"] if not pd.isna(e["frame_id"]) else np.nan,
                axis=1,
            )
            self.events["start_z"] = self.events["frame_id"].apply(
                lambda t: self.tracking.at[t, "ball_z"] if not pd.isna(t) else np.nan
            )

        if "end_x" not in self.events.columns:
            self.events["end_x"] = self.events["receive_frame_id"].apply(
                lambda t: tracking.at[t, "ball_x"] if not pd.isna(t) else np.nan
            )
            self.events["end_y"] = self.events["receive_frame_id"].apply(
                lambda t: tracking.at[t, "ball_y"] if not pd.isna(t) else np.nan
            )

        if action_type == "predefined":
            self.actions = events.copy()

        elif action_type == "pass":
            self.actions = self.filter_passes()

        elif action_type == "dribble":
            self.actions = self.filter_dribbles()

        elif action_type == "shot":
            self.actions = self.filter_shots()

        elif action_type == "shot_augment":
            self.actions = self.filter_shots(augment_size=30)

        elif action_type == "pass_dribble":
            passes = self.filter_passes()
            dribbles = self.filter_dribbles()
            self.actions = pd.concat([passes, dribbles]).sort_index()

        elif action_type == "pass_shot":
            passes = self.filter_passes()
            shots = self.filter_shots()
            self.actions = pd.concat([passes, shots]).sort_index()

        elif action_type == "failure":
            include_goals = True
            failed_passes = self.filter_passes(failure_only=True)
            blocked_shots = self.filter_shots(block_only=True)
            self.actions = pd.concat([failed_passes, blocked_shots]).sort_index()

        elif action_type.startswith("all"):
            include_goals = True
            passes = self.filter_passes()
            dribbles = self.filter_dribbles()
            shots = self.filter_shots()
            actions = pd.concat([passes, dribbles, shots]).astype({"frame_id": int}).sort_index()
            self.actions = actions.dropna(subset=["next_type"]).copy()

        self.include_keepers = include_keepers
        self.include_goals = include_goals
        self.max_players = 20 + int(include_keepers) * 2 + int(include_goals) * 2

        if self.include_goals:
            features = ["x", "y", "vx", "vy", "speed", "accel"]
            self.tracking[[f"{team}_goal_{x}" for team in ["home", "away"] for x in features]] = 0.0
            self.tracking["home_goal_x"] = config.FIELD_SIZE[0]
            self.tracking["home_goal_y"] = config.FIELD_SIZE[1] / 2
            self.tracking["away_goal_y"] = config.FIELD_SIZE[1] / 2

        self.graph_features_0 = None
        self.graph_features_1 = None
        self.tabular_features_0 = None
        self.tabular_features_1 = None
        self.labels = None

    # To make the home team always play from left to right (not needed for the current dataset)
    def rotate_pitch_per_phase(self):
        for phase in self.tracking["phase"].unique():
            phase_tracking = self.tracking[self.tracking["phase"] == phase].dropna(axis=1, how="all")

            x_cols = [c for c in phase_tracking.columns if c.endswith("_x")]
            y_cols = [c for c in phase_tracking.columns if c.endswith("_y")]
            vx_cols = [c for c in phase_tracking.columns if c.endswith("_vx")]
            vy_cols = [c for c in phase_tracking.columns if c.endswith("_vy")]

            if not proc.is_home_on_left(phase_tracking, halfline_x=config.FIELD_SIZE[0] / 2):
                self.tracking.loc[phase_tracking.index, x_cols] = config.FIELD_SIZE[0] - phase_tracking[x_cols]
                self.tracking.loc[phase_tracking.index, y_cols] = config.FIELD_SIZE[1] - phase_tracking[y_cols]
                self.tracking.loc[phase_tracking.index, vx_cols] = -phase_tracking[vx_cols]
                self.tracking.loc[phase_tracking.index, vy_cols] = -phase_tracking[vy_cols]

        self.events["x"] = self.tracking.loc[self.events["frame_id"], "ball_x"].values
        self.events["y"] = self.tracking.loc[self.events["frame_id"], "ball_y"].values

    def filter_passes(self, failure_only=False) -> pd.DataFrame:
        passes: pd.DataFrame = self.events[
            self.events["spadl_type"].isin(config.PASS)
            & self.events[["frame_id", "receive_frame_id"]].notna().all(axis=1)
            & (
                (self.events["receiver_id"] == self.events["next_player_id"])
                | (self.events["receiver_id"] == "out")
                | (self.events["next_type"].isin(["foul", "freekick_short"]))
            )
        ].copy()
        passes["action_type"] = "pass"
        passes["success"] = False
        passes["blocked"] = False  # To be updated
        passes["anomaly"] = False
        passes["woodwork"] = False

        pass_team = passes["object_id"].str[:4]
        receive_team = passes["receiver_id"].str[:4]
        passes.loc[(pass_team == receive_team) & ~passes["offside"], "success"] = True

        if failure_only:
            return passes[~passes["success"]].astype({"frame_id": int}).copy()
        else:
            return passes.astype({"frame_id": int})

    def filter_dribbles(self) -> pd.DataFrame:
        dribble_mask = self.events["spadl_type"].isin(["take_on", "dispossessed"])
        dribbles = self.events[dribble_mask].dropna(subset=["frame_id"]).copy()

        last_event_idxs = self.events.reset_index().groupby("period_id")["index"].max().values
        dribbles = dribbles[~dribbles.index.isin(last_event_idxs)].copy()

        dribbles["action_type"] = "dribble"
        dribbles.loc[dribbles["spadl_type"] == "dispossessed", "success"] = False
        dribbles["blocked"] = False
        dribbles["anomaly"] = False
        dribbles["woodwork"] = False
        dribbles["receiver_id"] = dribbles["next_player_id"]
        dribbles["receive_frame_id"] = self.events.loc[dribbles.index + 1, "frame_id"].values

        return dribbles.astype({"frame_id": int})

    def filter_shots(self, augment_size=0, random_state=42, block_only=False) -> pd.DataFrame:
        shots = self.events[self.events["spadl_type"].isin(config.SHOT)].dropna(subset="frame_id").copy()
        shots["action_type"] = "shot"
        shots["blocked"] = shots["next_type"] == "shot_block"
        shots["anomaly"] = False
        shots["woodwork"] = False

        for i in shots.index:
            anomaly_i, woodwork_i = utils.is_shot_anomaly(self.tracking, shots, i)
            shots.at[i, "anomaly"] = anomaly_i
            shots.at[i, "woodwork"] = woodwork_i

        if augment_size > 0:
            passes = self.events[self.events["spadl_type"].isin(["pass", "cross"])].dropna(subset="frame_id").copy()
            passes["action_type"] = "pass"

            passes["dist_x"] = passes["start_x"]
            passes["dist_y"] = passes["start_y"] - config.FIELD_SIZE[1] / 2
            home_passes = passes[passes["object_id"].str.startswith("home")]
            passes.loc[home_passes.index, "dist_x"] = config.FIELD_SIZE[0] - home_passes["start_x"]
            passes["goal_dist"] = passes[["dist_x", "dist_y"]].apply(np.linalg.norm, axis=1).round(2)

            far_passes = passes[passes["goal_dist"] > 30].drop(["dist_x", "dist_y", "goal_dist"], axis=1).copy()
            sampled_passes = far_passes.sample(augment_size, random_state=random_state)
            sampled_passes["success"] = False
            sampled_passes["blocked"] = True

            shots = pd.concat([shots, sampled_passes]).sort_index()

            # shot_features = XGModel.calc_shot_features(shots)
            # shots["xg_unblocked"] = self.xg_model.pred(shot_features)
            # shots["blocker_id"] = shots.apply(utils.find_nearest_blocker, axis=1, args=(self.tracking, self.keepers))

            # augmented_shots = (shots["xg_unblocked"] > 0.05) & (shots["blocker_id"].notna()) & (shots["start_z"] < 1)
            # shots = shots[(shots["action_type"] == "shot") | augmented_shots].copy()
            # shots["blocked"] = (shots["next_type"] == "shot_block") | (~shots["action_type"].isin(config.SHOT))

        if block_only:
            return shots[shots["blocked"]].astype({"frame_id": int}).copy()
        else:
            return shots.astype({"frame_id": int})

    def label_post_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        # To accurately estimate expected returns after actions
        actions = actions.copy()

        actions["end_player_id"] = actions["receiver_id"]
        actions["end_frame_id"] = actions["receive_frame_id"]
        actions["end_type"] = actions["next_type"]

        for period in actions["period_id"].unique():
            period_events = self.events[self.events["period_id"] == period]
            period_actions = actions[actions["period_id"] == period]

            for i in period_actions.index[:-1]:
                event_index = actions.at[i, "index"] if "index" in actions.columns else i
                event_player = actions.at[i, "object_id"]
                next_player = actions.at[i, "next_player_id"]
                next_type = actions.at[i, "next_type"]

                if next_type in config.SET_PIECE_OOP:
                    actions.at[i, "end_player_id"] = next_player
                    actions.at[i, "end_frame_id"] = self.events.at[event_index + 1, "frame_id"]
                    actions.at[i, "end_type"] = self.events.at[event_index + 1, "spadl_type"]

                elif next_type in config.DEFENSIVE_TOUCH:
                    valid_events = period_events[~period_events["spadl_type"].isin(config.DEFENSIVE_TOUCH)]
                    next_actions = valid_events.loc[event_index + 2 :, "frame_id"].dropna()

                    if len(next_actions) > 0:
                        next_action_index = next_actions.index[0]
                        actions.at[i, "end_player_id"] = self.events.at[next_action_index, "object_id"]
                        actions.at[i, "end_frame_id"] = self.events.at[next_action_index, "frame_id"]
                        actions.at[i, "end_type"] = self.events.at[next_action_index, "spadl_type"]

                elif next_type == "foul":
                    mask = period_events["spadl_type"].isin(config.SET_PIECE_FOUL)
                    next_set_pieces = period_events[mask].loc[event_index + 1 :]

                    if len(next_set_pieces) > 0:
                        next_sp_index = next_set_pieces.index[0]
                        next_sp_frame = next_set_pieces.at[next_sp_index, "frame_id"]

                        if not pd.isna(next_sp_frame) and next_sp_index < event_index + 5:
                            actions.at[i, "end_player_id"] = self.events.at[next_sp_index, "object_id"]
                            actions.at[i, "end_frame_id"] = self.events.at[next_sp_index, "frame_id"]
                            actions.at[i, "end_type"] = self.events.at[next_sp_index, "spadl_type"]

                            if actions.at[i, "spadl_type"] == "take_on":
                                if event_player[:4] == self.events.at[next_sp_index, "object_id"][:4]:
                                    actions.at[i, "success"] = True
                                    actions.at[i, "receiver_id"] = event_player
                                else:
                                    actions.at[i, "success"] = False
                                    foul = self.events.loc[event_index:next_sp_index].copy()
                                    foul_won = foul.loc[(foul["spadl_type"] == "foul") & foul["success"]]
                                    if not foul_won.empty:
                                        actions.at[i, "receiver_id"] = foul_won["object_id"].iloc[0]

            out_mask = (actions["period_id"] == period) & (actions["end_player_id"] == "out")
            actions.loc[out_mask, ["end_player_id", "end_frame_id", "end_type"]] = np.nan

        return actions

    def construct_labels(
        self,
        discount_xg=True,
        lookahead_len: int = 10,
        gamma: float = 0.9,
    ) -> torch.Tensor:
        self.actions = utils.label_intended_receivers(self.actions, self.tracking, self.action_type)

        self.events = utils.label_returns(self.events, lookahead_len)
        self.actions["scores"] = self.events.loc[self.actions.index, "scores"]
        self.actions["concedes"] = self.events.loc[self.actions.index, "concedes"]
        self.actions["scores_xg"] = self.events.loc[self.actions.index, "scores_xg"]
        self.actions["concedes_xg"] = self.events.loc[self.actions.index, "concedes_xg"]

        if discount_xg:
            self.events = utils.label_discounted_returns(self.events, gamma)
            self.actions["scores_xg_disc"] = self.events.loc[self.actions.index, "scores_xg_disc"]
            self.actions["concedes_xg_disc"] = self.events.loc[self.actions.index, "concedes_xg_disc"]

        labels_list = []

        for i in self.actions.index:
            frame = self.actions.at[i, "frame_id"]
            snapshot = self.tracking.loc[frame].dropna()
            if pd.isna(frame):
                continue

            if self.include_goals:
                home_players = [c[:-2] for c in snapshot.index if re.match(r"home_.*_x", c)]
                away_players = [c[:-2] for c in snapshot.index if re.match(r"away_.*_x", c)]
            else:
                home_players = [c[:-2] for c in snapshot.index if re.match(r"home_\d+_x", c)]
                away_players = [c[:-2] for c in snapshot.index if re.match(r"away_\d+_x", c)]

            intent_id: str = self.actions.at[i, "intent_id"]
            if pd.isna(intent_id) or self.action_type in ["predefined", "shot", "shot_augment"]:
                intent_index = -1
            elif intent_id.startswith("home"):
                intent_index = home_players.index(intent_id)
            else:  # if intent.startswith("away"):
                intent_index = away_players.index(intent_id)

            receiver_id: str = self.actions.at[i, "receiver_id"]
            receive_frame: float = self.actions.at[i, "receive_frame_id"]
            duration = 0.0 if pd.isna(receive_frame) else round((receive_frame - frame) / self.fps, 2)

            start_x = start_y = end_x = end_y = intent_x = intent_y = np.nan

            if self.actions.at[i, "action_type"] == "pass" and not self.action_type.startswith("shot"):
                # For shot feature generation, selected passes are regarded as potentially failed shots
                try:
                    if receiver_id == "out":
                        receiver_index = -1
                    elif self.actions.at[i, "object_id"].startswith("home"):
                        receiver_index = (home_players + away_players).index(receiver_id)
                    elif self.actions.at[i, "object_id"].startswith("away"):
                        receiver_index = (away_players + home_players).index(receiver_id)

                except ValueError:
                    continue

                start_x = self.actions.at[i, "start_x"]
                start_y = self.actions.at[i, "start_y"]
                end_x = self.actions.at[i, "end_x"]
                end_y = self.actions.at[i, "end_y"]
                if not pd.isna(intent_id):
                    intent_x = self.tracking.at[receive_frame, f"{intent_id}_x"]
                    intent_y = self.tracking.at[receive_frame, f"{intent_id}_y"]

                # Make the attacking team plays from left to right
                if self.actions.at[i, "object_id"].startswith("away"):
                    start_x = config.FIELD_SIZE[0] - start_x
                    start_y = config.FIELD_SIZE[1] - start_y
                    end_x = config.FIELD_SIZE[0] - end_x
                    end_y = config.FIELD_SIZE[1] - end_y
                    intent_x = config.FIELD_SIZE[0] - intent_x
                    intent_y = config.FIELD_SIZE[1] - intent_y

                is_real = 1

            else:
                duration = 0.0

                if self.actions.at[i, "action_type"] == "shot" or self.action_type.startswith("shot"):
                    is_real = int(self.actions.at[i, "action_type"] == "shot")

                    if pd.isna(receiver_id):
                        continue
                    if receiver_id == "out" or receiver_id.endswith("goal"):
                        receiver_index = -1
                    elif self.actions.at[i, "object_id"].startswith("home"):
                        receiver_index = (home_players + away_players).index(receiver_id)
                    elif self.actions.at[i, "object_id"].startswith("away"):
                        receiver_index = (away_players + home_players).index(receiver_id)

                else:
                    is_real = 1
                    receiver_index = -1

            labels_list.append(
                [
                    i,
                    int(self.actions.at[i, "action_type"] == "pass"),
                    int(self.actions.at[i, "action_type"] == "dribble"),
                    int(self.actions.at[i, "action_type"] == "shot"),
                    len(home_players) + len(away_players),
                    intent_index,
                    receiver_index,
                    duration,
                    start_x if not pd.isna(start_x) else -config.FIELD_SIZE[0],
                    start_y if not pd.isna(start_y) else -config.FIELD_SIZE[1],
                    end_x if not pd.isna(end_x) else -config.FIELD_SIZE[0],
                    end_y if not pd.isna(end_y) else -config.FIELD_SIZE[1],
                    intent_x if not pd.isna(intent_x) else -config.FIELD_SIZE[0],
                    intent_y if not pd.isna(intent_y) else -config.FIELD_SIZE[1],
                    is_real,  # Whether this label is for a real or an augmented event
                    int(self.actions.at[i, "blocked"]),
                    int(self.actions.at[i, "success"]),
                    self.actions.at[i, "scores"],
                    self.actions.at[i, "scores_xg_disc"] if discount_xg else self.actions.at[i, "scores_xg"],
                    self.actions.at[i, "concedes"],
                    self.actions.at[i, "concedes_xg_disc"] if discount_xg else self.actions.at[i, "concedes_xg"],
                ]
            )

        return torch.tensor(labels_list, dtype=torch.float32)

    def plot_snapshot(
        self,
        event_index: int,
        heatmap: Dict[int, np.ndarray] = None,
        edges: np.ndarray = None,
        team_to_drop: str = None,  # home or away
        cmap="jet",
        cmin=0,
        cmax=0.05,
    ):
        event_type = self.actions.at[event_index, "spadl_type"]
        player_id = self.actions.at[event_index, "object_id"]
        intent_id = self.actions.at[event_index, "intent_id"]
        receiver_id = self.actions.at[event_index, "receiver_id"]
        print(f"{event_type} by {player_id}: intended to {intent_id}, received by {receiver_id}")

        event_frame = self.events.at[event_index, "frame_id"]
        frame_data = self.tracking.loc[event_frame:].iloc[:1].dropna(axis=1).copy()

        if team_to_drop is not None:
            assert team_to_drop in ["home", "away"]
            filtered_cols = [c for c in frame_data.columns if not c.startswith(team_to_drop)]
            frame_data = frame_data[filtered_cols].copy()

        end_x = self.actions.at[event_index, "end_x"]
        end_y = self.actions.at[event_index, "end_y"]
        arrows = [(player_id, (end_x, end_y))]

        if heatmap is not None and player_id[:4] == "away":
            heatmap = heatmap[::-1, ::-1]
            cmap = f"{cmap}_r"

        viz = SnapshotVisualizer(frame_data, edges=edges, arrows=arrows, heatmap=heatmap)
        viz.plot(hm_cmap=cmap, cmin=cmin, cmax=cmax)
