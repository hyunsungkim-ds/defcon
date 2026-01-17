import os
import sys
from collections import defaultdict
from typing import Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from datatools import config
from datatools.event_xg import EventXGModel
from datatools.match import Match
from datatools.utils import abbr_position, find_active_players, sparsify_edges
from datatools.viz_snapshot import SnapshotVisualizer
from inference import inference_boost, inference_gnn, inference_gnn_posterior
from models.utils import load_model


class DEFCON:
    def __init__(
        self,
        match: Match,
        shot_success: str = "unblocked",  # choices: [goal, on_target, unblocked]
        event_xg_model: EventXGModel = None,
        select_model_id: str = None,
        pass_success_model_id: str = None,
        shot_block_model_id: str = None,
        shot_xg_model_id: str = None,
        score_model_id: str = None,
        concede_model_id: str = None,
        posterior_model_id: str = None,
        device="cuda",
    ):
        self.match = match
        self.actions = self.match.actions[self.match.actions["end_frame_id"].notna()].copy()
        self.shot_success = shot_success
        self.device = device

        self.event_xg_model = event_xg_model
        self.select_model = load_model(select_model_id, device)
        self.pass_success_model = load_model(pass_success_model_id, device)
        self.shot_block_model = load_model(shot_block_model_id, device)
        self.shot_xg_model = load_model(shot_xg_model_id, device)
        self.score_model = load_model(score_model_id, device)
        self.concede_model = load_model(concede_model_id, device)
        self.posterior_model = load_model(posterior_model_id, device)

        self.select_probs_0: pd.DataFrame = None
        self.success_probs_0: pd.DataFrame = None
        self.s_score_probs_0: pd.DataFrame = None
        self.f_score_probs_0: pd.DataFrame = None
        self.s_concede_probs_0: pd.DataFrame = None
        self.f_concede_probs_0: pd.DataFrame = None

        self.select_probs_1: pd.DataFrame = None
        self.success_probs_1: pd.DataFrame = None
        self.s_score_probs_1: pd.DataFrame = None
        self.f_score_probs_1: pd.DataFrame = None
        self.s_concede_probs_1: pd.DataFrame = None
        self.f_concede_probs_1: pd.DataFrame = None

        self.posteriors: pd.DataFrame = None
        self.shot_out_prob: float = 2380 / 6950  # Average probability of a shot going out of play

        self.option_values_0: pd.DataFrame = None
        self.option_values_1: pd.DataFrame = None
        self.epv: pd.DataFrame = None

        self.credits = None
        self.player_scores = None

    def estimate_components(self, use_vendor_xg: bool = False):
        indices = self.actions.index

        self.select_probs_0, _ = inference_gnn(self.match, self.select_model, self.device, False, indices)
        self.select_probs_1, _ = inference_gnn(self.match, self.select_model, self.device, True, indices)
        self.success_probs_0, _ = inference_gnn(self.match, self.pass_success_model, self.device, False, indices)
        self.success_probs_1, _ = inference_gnn(self.match, self.pass_success_model, self.device, True, indices)

        score_probs_0 = inference_gnn(self.match, self.score_model, self.device, False, indices)
        score_probs_1 = inference_gnn(self.match, self.score_model, self.device, True, indices)
        concede_probs_0 = inference_gnn(self.match, self.concede_model, self.device, False, indices)
        concede_probs_1 = inference_gnn(self.match, self.concede_model, self.device, True, indices)

        self.f_score_probs_0, self.s_score_probs_0 = score_probs_0
        self.f_score_probs_1, self.s_score_probs_1 = score_probs_1
        self.f_concede_probs_0, self.s_concede_probs_0 = concede_probs_0
        self.f_concede_probs_1, self.s_concede_probs_1 = concede_probs_1

        self.posteriors = inference_gnn_posterior(self.match, self.posterior_model, self.device, indices)
        self.estimate_shot_components(use_vendor_xg)

        self.select_probs_0, self.success_probs_0, self.s_score_probs_0 = self.adjust_set_piece_probs(
            self.select_probs_0,
            self.success_probs_0,
            self.s_score_probs_0,
            post_action=False,
        )
        self.select_probs_1, self.success_probs_1, self.s_score_probs_1 = self.adjust_set_piece_probs(
            self.select_probs_1,
            self.success_probs_1,
            self.s_score_probs_1,
            post_action=True,
        )

    def estimate_shot_components(self, use_vendor_xg: bool = False):
        if self.shot_success == "unblocked" and self.shot_block_model is not None:
            block_probs_0, xg_unblocked_0 = self.estimate_shot_block_probs(self.actions, False)
            block_probs_1, xg_unblocked_1 = self.estimate_shot_block_probs(self.actions, True)

            self.success_probs_0 = DEFCON.combine_pass_shot_success_probs(self.success_probs_0, 1 - block_probs_0)
            self.success_probs_1 = DEFCON.combine_pass_shot_success_probs(self.success_probs_1, 1 - block_probs_1)

            if use_vendor_xg and "expected_goal" in self.actions.columns:
                # Use vendor-provided xG values for real shots
                shot_mask = self.actions["expected_goal"].notna()
                shot_xg_0 = self.actions.loc[shot_mask, "expected_goal"].astype(float)
                xg_unblocked_0.loc[shot_mask] = np.minimum(shot_xg_0 / (1 - block_probs_0[shot_mask]), 1)

                pre_shot_mask = self.actions["expected_goal"].shift(-1).notna()
                shot_xg_1 = self.actions["expected_goal"].shift(-1).loc[pre_shot_mask].astype(float)
                xg_unblocked_1.loc[pre_shot_mask] = np.minimum(shot_xg_1 / (1 - block_probs_1[pre_shot_mask]), 1)

            corners = self.actions[self.actions["spadl_type"].str.startswith("corner")]
            xg_unblocked_0.loc[corners.index] = 0.0
            self.s_score_probs_0, self.f_score_probs_0, self.s_concede_probs_0, self.f_concede_probs_0 = (
                DEFCON.combine_pass_shot_goal_probs(
                    self.s_score_probs_0,
                    self.f_score_probs_0,
                    self.s_concede_probs_0,
                    self.f_concede_probs_0,
                    xg_unblocked_0,
                )
            )
            self.s_score_probs_1, self.f_score_probs_1, self.s_concede_probs_1, self.f_concede_probs_1 = (
                DEFCON.combine_pass_shot_goal_probs(
                    self.s_score_probs_1,
                    self.f_score_probs_1,
                    self.s_concede_probs_1,
                    self.f_concede_probs_1,
                    xg_unblocked_1,
                )
            )

        if self.shot_success in ["goal", "on_target"] and self.shot_xg_model is not None:
            xg_0 = inference_boost(self.match, self.shot_xg_model, False, event_indices=self.actions.index)
            xg_1 = inference_boost(self.match, self.shot_xg_model, True, event_indices=self.actions.index)

            if use_vendor_xg and "expected_goal" in self.actions.columns:
                # Use vendor-provided xG values for real shots
                shot_mask = self.actions["expected_goal"].notna()
                xg_0.loc[shot_mask] = self.actions.loc[shot_mask, "expected_goal"].astype(float)

                pre_shot_mask = self.actions["expected_goal"].shift(-1).notna() & (self.actions["end_type"] == "shot")
                xg_1.loc[pre_shot_mask] = self.actions["expected_goal"].shift(-1).loc[pre_shot_mask].astype(float)

            self.success_probs_0 = DEFCON.combine_pass_shot_success_probs(self.success_probs_0, xg_0)
            self.success_probs_1 = DEFCON.combine_pass_shot_success_probs(self.success_probs_1, xg_1)

            self.s_score_probs_0, self.f_score_probs_0, self.s_concede_probs_0, self.f_concede_probs_0 = (
                DEFCON.combine_pass_shot_goal_probs(
                    self.s_score_probs_0,
                    self.f_score_probs_0,
                    self.s_concede_probs_0,
                    self.f_concede_probs_0,
                )
            )
            self.s_score_probs_1, self.f_score_probs_1, self.s_concede_probs_1, self.f_concede_probs_1 = (
                DEFCON.combine_pass_shot_goal_probs(
                    self.s_score_probs_1,
                    self.f_score_probs_1,
                    self.s_concede_probs_1,
                    self.f_concede_probs_1,
                )
            )

            if self.shot_success == "on_target":  # Define shot success as being on target instead of scoring a goal
                self.success_probs_0, self.s_score_probs_0 = self.replace_xg_with_xgot(
                    self.success_probs_0, self.s_score_probs_0
                )
                self.success_probs_1, self.s_score_probs_1 = self.replace_xg_with_xgot(
                    self.success_probs_1, self.s_score_probs_1
                )

    def estimate_shot_block_probs(self, actions: pd.DataFrame, post_event: bool = False) -> Tuple[pd.Series, pd.Series]:
        assert self.shot_block_model is not None and self.event_xg_model is not None

        if post_event:
            actions = actions.loc[actions["end_frame_id"].notna(), ["end_frame_id", "end_player_id", "end_type"]].copy()
            actions.columns = ["frame_id", "object_id", "spadl_type"]
            actions["start_x"] = actions.apply(lambda x: self.match.tracking.at[x["frame_id"], "ball_x"], axis=1)
            actions["start_y"] = actions.apply(lambda x: self.match.tracking.at[x["frame_id"], "ball_y"], axis=1)
            actions["start_z"] = actions.apply(lambda x: self.match.tracking.at[x["frame_id"], "ball_z"], axis=1)

        shot_features = self.event_xg_model.calc_shot_features(actions)
        xg_unblocked = self.event_xg_model.pred(shot_features)  # xG if the shot were unblocked

        chance_indices = actions[xg_unblocked > 0.01].index
        block_probs = inference_gnn(self.match, self.shot_block_model, self.device, post_event, chance_indices)[0]

        return block_probs, xg_unblocked

    def replace_xg_with_xgot(
        self,
        success_probs: pd.DataFrame,
        score_probs: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        success_probs = success_probs.copy()
        score_probs = score_probs.copy()

        targets = ["home_goal", "away_goal"]
        keepers = np.unique(self.match.phases["active_keepers"].sum())
        shot_mask = self.posteriors["option"].isin(targets) & self.posteriors["defender"].isin(keepers)
        shot_resp = self.posteriors[shot_mask].copy()

        for i in success_probs.index:
            target = f"{self.actions.at[i, 'object_id'][:4]}_goal"
            xg = success_probs.at[i, target]

            frame = self.actions.at[i, "frame_id"]
            phase = self.match.tracking.at[frame, "phase_id"]
            keeper = [p for p in self.match.phases.at[phase, "active_keepers"] if p[:4] != target[:4]][0]

            mask_i = (shot_resp["index"] == i) & (shot_resp["option"] == target) & (shot_resp["defender"] == keeper)
            keeper_resp = shot_resp[mask_i]["posterior"].iloc[0]

            sot_prob = xg + (1 - xg) * (1 - self.shot_out_prob) * keeper_resp  # Probability of the shot on target
            xgot = xg / sot_prob  # xG on target

            success_probs.at[i, target] = sot_prob
            score_probs.at[i, target] = xgot

        return success_probs, score_probs

    @staticmethod
    def combine_pass_shot_success_probs(
        pass_success_probs: pd.DataFrame,
        shot_success_probs: pd.Series,
    ) -> pd.DataFrame:
        success_probs = pass_success_probs.copy()
        success_probs["home_goal"] = np.nan
        success_probs["away_goal"] = np.nan

        for i in success_probs.index:
            team = success_probs.loc[i].dropna().index[0][:4]
            if i in shot_success_probs.index:
                success_probs.at[i, f"{team}_goal"] = shot_success_probs.at[i]
            else:
                success_probs.at[i, f"{team}_goal"] = 0.0

        return success_probs

    @staticmethod
    def combine_pass_shot_goal_probs(
        s_score_probs: pd.DataFrame,
        f_score_probs: pd.DataFrame,
        s_concede_probs: pd.DataFrame,
        f_concede_probs: pd.DataFrame,
        xg_unblocked: pd.Series = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        s_score_probs = s_score_probs.copy()
        f_score_probs = f_score_probs.copy()
        s_concede_probs = s_concede_probs.copy()
        f_concede_probs = f_concede_probs.copy()

        for goal in ["home_goal", "away_goal"]:
            s_score_probs[goal] = np.nan
            f_score_probs[goal] = np.nan
            s_concede_probs[goal] = np.nan
            f_concede_probs[goal] = np.nan

        for i in s_score_probs.index:
            team = s_score_probs.loc[i].dropna().index[0][:4]
            s_score_probs.at[i, f"{team}_goal"] = 1.0 if xg_unblocked is None else xg_unblocked.at[i]
            f_score_probs.at[i, f"{team}_goal"] = 0.0
            s_concede_probs.at[i, f"{team}_goal"] = 0.0
            f_concede_probs.at[i, f"{team}_goal"] = 0.0

        return s_score_probs, f_score_probs, s_concede_probs, f_concede_probs

    def adjust_set_piece_probs(
        self,
        select_probs: pd.DataFrame,
        success_probs: pd.DataFrame,
        s_score_probs: pd.DataFrame,
        post_action: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        select_probs = select_probs.copy()
        success_probs = success_probs.copy()
        s_score_probs = s_score_probs.copy()

        player_col = "end_player_id" if post_action else "object_id"
        type_col = "end_type" if post_action else "spadl_type"
        set_piece_mask = self.actions[type_col].isin(config.SET_PIECE)

        for i in self.actions[set_piece_mask].index:
            player_id = self.actions.at[i, player_col]
            shot_target = f"{player_id[:4]}_goal"

            if self.actions.at[i, type_col] == "shot_penalty":
                select_probs.loc[i] *= 0.0
                select_probs.at[i, shot_target] = 1.0

                if self.shot_success == "goal":
                    success_probs.at[i, shot_target] = 0.7884
                    s_score_probs.at[i, shot_target] = 1.0
                else:
                    success_probs.at[i, shot_target] = 1.0
                    s_score_probs.at[i, shot_target] = 0.7884

            else:
                select_probs.at[i, player_id] = 0.0
                success_probs.at[i, player_id] = 0.0

                if self.actions.at[i, type_col] == "throw_in":
                    select_probs.at[i, shot_target] = 0.0
                    success_probs.at[i, shot_target] = 0.0

                select_probs.loc[i] /= select_probs.loc[i].astype(float).sum()

        return select_probs, success_probs, s_score_probs

    def estimate_epv(self) -> pd.DataFrame:
        s_values_0 = self.s_score_probs_0 - self.s_concede_probs_0
        f_values_0 = self.f_score_probs_0 - self.f_concede_probs_0
        self.option_values_0 = self.success_probs_0 * s_values_0 + (1 - self.success_probs_0) * f_values_0
        epv_0 = (self.select_probs_0 * self.option_values_0).sum(axis=1)

        s_values_1 = self.s_score_probs_1 - self.s_concede_probs_1
        f_values_1 = self.f_score_probs_1 - self.f_concede_probs_1
        self.option_values_1 = self.success_probs_1 * s_values_1 + (1 - self.success_probs_1) * f_values_1
        epv_1 = (self.select_probs_1 * self.option_values_1).sum(axis=1)

        corner_mask = self.actions["spadl_type"].str.startswith("corner")
        pre_corner_mask = self.actions["end_type"].str.startswith("corner")
        epv_0.loc[corner_mask] = 0.02
        epv_1.loc[pre_corner_mask] = 0.02
        epv_1.loc[self.actions["offside"]] = epv_0.loc[self.actions["offside"]]

        start_teams = self.actions["object_id"].str[:4]
        end_teams = self.actions["end_player_id"].str[:4]
        epv_1 *= np.where(start_teams == end_teams, 1, -1)

        epv_1.loc[(self.actions["action_type"] == "shot") & (self.actions["success"])] = 1.0
        epv_1.loc[self.actions["spadl_type"] == "own_goal"] = -1.0

        epv = self.actions[["object_id", "action_type", "spadl_type", "receiver_id", "next_type"]].copy()
        epv["start"] = epv_0.values
        epv["end"] = epv_1.values
        epv["diff"] = epv["end"] - epv["start"]

        dispossess = (epv["receiver_id"].str[:4] != epv["object_id"].str[:4]) & (epv["receiver_id"].str[:4] != "out")
        offense_bad_touch = (self.actions["next_type"] == "bad_touch") & (end_teams != start_teams)
        defense_bad_touch = (self.actions["next_type"] == "bad_touch") & (end_teams == start_teams)

        epv["defense_type"] = None
        epv.loc[~self.actions["success"] & dispossess & ~defense_bad_touch, "defense_type"] = "intercept"
        epv.loc[~self.actions["success"] & dispossess & defense_bad_touch, "defense_type"] = "concede"
        epv.loc[~self.actions["success"] & ~dispossess, "defense_type"] = "disturb"
        epv.loc[self.actions["success"] & (epv["diff"] < 0) & ~offense_bad_touch, "defense_type"] = "deter"
        epv.loc[self.actions["success"] & (epv["diff"] >= 0), "defense_type"] = "concede"

        epv.loc[self.actions["action_type"] == "shot", "defense_type"] += "_shot"
        epv.loc[self.actions["action_type"] != "shot", "defense_type"] += "_pass"

        epv.loc[(self.actions["next_type"] == "foul") & (end_teams == start_teams), "defense_type"] = "foul"

        return epv

    def assign_pass_credits(self, i: int) -> pd.DataFrame:
        target = self.actions.at[i, "intent_id"]
        receiver = self.actions.at[i, "receiver_id"]

        epv = self.epv.at[i, "start"]
        s_values = self.s_score_probs_0.loc[i] - self.s_concede_probs_0.loc[i]
        f_values = self.f_score_probs_0.loc[i] - self.f_concede_probs_0.loc[i]

        deter_values = (1 - self.success_probs_0.loc[i]) * (s_values - f_values)
        deter_values = deter_values.where(s_values >= epv, 0.0)
        denom = float(deter_values.sum())

        credits_i = self.posteriors[self.posteriors["index"] == i].copy()
        credits_i["defense_type"] = None
        credits_i["team_credit"] = -float(self.epv.at[i, "diff"])
        credits_i["option_resp"] = 0.0
        credits_i["weight"] = 0.0

        defense_type = self.epv.at[i, "defense_type"]
        if defense_type == "intercept_pass":
            credits_i.loc[credits_i["option"] == target, "option_resp"] = 1.0

            blocker_mask = (credits_i["option"] == target) & (credits_i["defender"] == receiver)
            defenders_mask = (credits_i["option"] == target) & (credits_i["defender"] != receiver)
            credits_i.loc[blocker_mask, "defense_type"] = "intercept_pass"
            credits_i.loc[defenders_mask, "defense_type"] = "disturb_pass"

            if pd.isna(target):
                credits_i.loc[blocker_mask, "weight"] = 1.0
            else:
                success_prob_ij = self.success_probs_0.at[i, target]
                posteriors_ij = credits_i.loc[credits_i["option"] == target, "posterior"].values
                credits_i.loc[credits_i["option"] == target, "weight"] = (1 - success_prob_ij) * posteriors_ij
                credits_i.loc[blocker_mask, "weight"] += success_prob_ij

        elif defense_type in ["disturb_pass", "concede_pass"]:
            target_mask = credits_i["option"] == target
            credits_i.loc[target_mask, "defense_type"] = defense_type
            credits_i.loc[target_mask, "option_resp"] = 1.0
            credits_i.loc[target_mask, "weight"] = credits_i.loc[target_mask, "posterior"].values

        elif defense_type == "deter_pass" and denom > 0:
            credits_i["defense_type"] = "deter_pass"
            credits_i["option_resp"] = credits_i["option"].map(deter_values).fillna(0.0).to_numpy() / denom
            credits_i["weight"] = credits_i["option_resp"] * credits_i["posterior"]
            credits_i.loc[credits_i["option_resp"] > 0, "defense_type"] = "deter_pass"

        credits_i["player_credit"] = credits_i["team_credit"] * credits_i["weight"]
        return credits_i[credits_i["defense_type"].notna()].copy()

    def assign_shot_credits(self, i: int) -> pd.DataFrame:
        target = self.actions.at[i, "intent_id"]
        receiver = self.actions.at[i, "receiver_id"]

        epv_0 = float(self.epv.at[i, "start"])
        epv_1 = 1.0 if self.actions.at[i, "success"] else float(self.epv.at[i, "end"])

        shot_success_prob = float(self.success_probs_0.at[i, target])  # sot_prob or (1 - block_prob)
        xg_on_success = float(self.s_score_probs_0.at[i, target])  # xgot or xg_unblocked

        credits_i = self.posteriors[
            (self.posteriors["index"] == i)
            & (self.posteriors["option"] == target)
            & (self.posteriors["defender"] != "out")
        ].copy()

        frame = self.actions.at[i, "frame_id"]
        phase = self.match.tracking.at[frame, "phase_id"]
        keeper = [p for p in self.match.phases.at[phase, "active_keepers"] if p[:4] != target[:4]][0]
        keeper_mask = credits_i["defender"] == keeper

        credits_i["team_credit"] = 0.0
        credits_i["player_credit"] = 0.0

        if self.actions.at[i, "success"]:  # Conceding a goal
            credits_i["defense_type"] = "concede_shot"
            credits_i["team_credit"] = epv_0 - 1.0
            credits_i["player_credit"] = credits_i["team_credit"] * credits_i["posterior"]

        elif self.actions.at[i, "next_type"] == "keeper_save":  # Saved shot
            credits_i["defense_type"] = "concede_shot"
            credits_i["team_credit"] = epv_0 - epv_1

            credits_i.loc[keeper_mask, "posterior"] = 0.0
            credits_i["posterior"] /= credits_i["posterior"].sum()
            credits_i.loc[~keeper_mask, "player_credit"] = (epv_0 - xg_on_success) * credits_i["posterior"]
            credits_i.loc[keeper_mask, "player_credit"] = xg_on_success - epv_1

            if self.actions.at[i, "next_type"] == "keeper_save":
                credits_i.loc[keeper_mask, "defense_type"] = "intercept_shot"

        elif self.actions.at[i, "next_type"] == "shot_block":  # Blocked shot
            credits_i["defense_type"] = "disturb_shot"
            credits_i.loc[keeper_mask, "posterior"] = 0.0
            credits_i["posterior"] /= credits_i["posterior"].sum()
            credits_i["weight"] = (1 - shot_success_prob) * credits_i["posterior"]

            blocker_mask = credits_i["defender"] == receiver
            credits_i.loc[blocker_mask, "defense_type"] = "intercept_shot"
            credits_i.loc[blocker_mask, "weight"] += shot_success_prob

            credits_i["team_credit"] = epv_0 - epv_1
            credits_i["player_credit"] = credits_i["team_credit"] * credits_i["weight"]

        elif self.actions.at[i, "woodwork"]:  # Shot off the woodwork
            credits_i["defense_type"] = "concede_shot"
            credits_i["team_credit"] = min(epv_0 - 0.5, 0)
            credits_i["player_credit"] = credits_i["team_credit"] * credits_i["posterior"]

        elif receiver[:4] in [target[:4], "out"]:  # Unblocked shot
            if self.shot_success == "on_target":
                xg = xg_on_success * shot_success_prob
                unblocked_xg = xg / min(shot_success_prob + (1 - xg) * self.shot_out_prob, 1)
            elif self.shot_success == "unblocked":
                unblocked_xg = xg_on_success
            else:
                unblocked_xg = epv_0

            credits_i["defense_type"] = "concede_shot"
            credits_i["team_credit"] = epv_0 - unblocked_xg

            credits_i.loc[keeper_mask, "posterior"] = 0.0
            credits_i["posterior"] /= credits_i["posterior"].sum()
            credits_i.loc[~keeper_mask, "player_credit"] = credits_i["team_credit"] * credits_i["posterior"]

        return credits_i

    def assign_foul_credits(self, i: int) -> pd.DataFrame:
        player_id = self.actions.at[i, "object_id"]
        action_type = self.actions.at[i, "action_type"]

        next_events = self.match.events.loc[i + 1 : i + 2].copy()
        oppo_foul_mask = (next_events["spadl_type"] == "foul") & (next_events["object_id"].str[:4] != player_id[:4])
        oppo_foul_i = next_events[oppo_foul_mask]

        if len(oppo_foul_i) == 1:
            oppo_foul_i = oppo_foul_i.iloc[0]
            credits_i = {
                "index": i,
                "option": self.actions.at[i, "intent_id"],
                "defender": oppo_foul_i["object_id"],
                "posterior": 1.0,
                "team_credit": -self.epv.at[i, "diff"],
                "weight": 1.0,
                "player_credit": -self.epv.at[i, "diff"],
            }
            if oppo_foul_i["success"]:
                credits_i["defense_type"] = "disturb_shot" if action_type == "shot" else "disturb_pass"
            else:
                credits_i["defense_type"] = "foul"
            return pd.Series(credits_i).to_frame().T
        elif action_type == "shot":
            return self.assign_shot_credits(i)
        else:
            return self.assign_pass_credits(i)

    def assign_credits(self) -> pd.DataFrame:
        credits = []
        anomaly_mask = self.actions["anomaly"].fillna(False)

        for i in tqdm(self.epv.index, "defensive_credit"):
            if anomaly_mask.at[i] or self.actions.at[i, "spadl_type"] == "own_goal":
                continue
            elif self.actions.at[i, "next_type"] == "foul":
                credits_i = self.assign_foul_credits(i)
            elif self.actions.at[i, "action_type"] == "shot":
                credits_i = self.assign_shot_credits(i)
            else:
                credits_i = self.assign_pass_credits(i)
            credits.append(credits_i)

        return pd.concat(credits, ignore_index=True)

    def compute_playing_times(self) -> pd.DataFrame:
        tracking = self.match.tracking
        seconds = dict()

        for p in self.match.lineup["object_id"]:
            alive_tracking = tracking[tracking[f"{p}_x"].notna() & (tracking["ball_state"] == "alive")]
            player_seconds = alive_tracking.groupby("ball_owning_home_away")["timestamp"].count() / self.match.fps
            if p[:4] == "home":
                seconds[p] = [player_seconds["home"], player_seconds["away"]]
            else:
                seconds[p] = [player_seconds["away"], player_seconds["home"]]

        seconds = pd.DataFrame(seconds, index=["attack_time", "defend_time"]).T
        seconds.index.name = "object_id"

        return seconds

    def evaluate_players(self) -> pd.DataFrame:
        if self.success_probs_0 is None:
            self.estimate_components()

        self.epv = self.estimate_epv()
        self.credits = self.assign_credits()

        player_scores = self.credits.pivot_table("player_credit", "defender", "defense_type", "sum")
        player_scores["score"] = player_scores.sum(axis=1)
        player_scores = player_scores.reset_index().rename(columns={"defender": "player_id"})

        lineup = self.match.lineup[["object_id", "advanced_position", "match_name", "mins_played"]].copy()
        lineup.columns = ["player_id", "position", "player_name", "mins_played"]
        lineup["position"] = lineup["position"].apply(abbr_position)
        player_scores = pd.merge(lineup, player_scores).set_index("player_id").infer_objects(copy=False).fillna(0)

        playing_times = self.compute_playing_times()
        player_scores = pd.merge(player_scores.copy(), playing_times, left_index=True, right_index=True)
        player_scores = player_scores.reset_index().rename(columns={"index": "object_id"})

        lineup = self.match.lineup[config.LINEUP_HEADER[:6]].copy()
        lineup.columns = ["match_id", "match_date", "team_name", "player_id", "object_id", "uniform_number"]

        return pd.merge(lineup, player_scores)

    def find_snapshot_values(self, event_index: int, post_event=False) -> pd.DataFrame:
        spadl_type = self.actions.at[event_index, "spadl_type"]
        possessor = self.actions.at[event_index, "object_id"]
        receiver = self.actions.at[event_index, "receiver_id"]

        if self.epv is not None:
            epv_0 = round(self.epv.at[event_index, "start"], 4)
            epv_1 = round(self.epv.at[event_index, "end"], 4)
            print(f"{spadl_type} from {possessor} ({epv_0:.4f}) to {receiver} ({epv_1:.4f})")
        else:
            print(f"{spadl_type} from {possessor} to {receiver}")

        if post_event:
            frame = int(self.actions.at[event_index, "end_frame_id"])
            team = self.actions.at[event_index, "end_player_id"][:4]
            players = find_active_players(self.match.tracking, frame, team, include_goals=True)
            values = pd.DataFrame(index=players[0])

            values["select"] = self.select_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()
            values["success"] = self.success_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()

            values["s_score"] = self.s_score_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()
            values["s_concede"] = self.s_concede_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()
            values["s_value"] = values["s_score"] - values["s_concede"]

            values["f_score"] = self.f_score_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()
            values["f_concede"] = self.f_concede_probs_1.loc[event_index, players[0]].dropna().astype(float).copy()
            values["f_value"] = values["f_score"] - values["f_concede"]

            values["option_value"] = values["success"] * values["s_value"] + (1 - values["success"]) * values["f_value"]
            values["advantage"] = values["s_value"] - values["f_value"]

        else:
            frame = int(self.match.events.at[event_index, "frame_id"])
            team = self.match.events.at[event_index, "object_id"][:4]
            players = find_active_players(self.match.tracking, frame, team, include_goals=True)
            values = pd.DataFrame(index=players[0])

            values["select"] = self.select_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()
            values["success"] = self.success_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()

            values["s_score"] = self.s_score_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()
            values["s_concede"] = self.s_concede_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()
            values["s_value"] = values["s_score"] - values["s_concede"]

            values["f_score"] = self.f_score_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()
            values["f_concede"] = self.f_concede_probs_0.loc[event_index, players[0]].dropna().astype(float).copy()
            values["f_value"] = values["f_score"] - values["f_concede"]

            values["option_value"] = values["success"] * values["s_value"] + (1 - values["success"]) * values["f_value"]
            values["advantage"] = values["s_value"] - values["f_value"]

        return values

    def visualize_snapshot(
        self,
        event_index,
        post_event=False,
        hypo_target=None,
        size=None,
        color=None,
        annot=None,
        show_edges=False,
    ) -> pd.DataFrame:
        if post_event:
            frame = self.actions.at[event_index, "end_frame_id"]
            spadl_type = self.actions.at[event_index, "end_type"]
            possessor = self.actions.at[event_index, "end_player_id"]
            receiver, ball_xy = None, None

        else:
            frame = self.match.events.at[event_index, "frame_id"]
            spadl_type = self.match.events.at[event_index, "spadl_type"]
            possessor = self.match.events.at[event_index, "object_id"]
            receiver = possessor if spadl_type == "take_on" else self.match.events.at[event_index, "receiver_id"]
            receive_frame = self.match.events.at[event_index, "receive_frame_id"]

            if not pd.isna(receive_frame):
                ball_xy = self.match.tracking.loc[frame:receive_frame, ["ball_x", "ball_y"]]
            else:
                ball_xy = None

        frame_data = self.match.tracking.loc[frame:frame].dropna(axis=1).copy()
        players = find_active_players(self.match.tracking, frame, possessor[:4], include_goals=True)

        if event_index in self.actions.index:
            target = self.actions.at[event_index, "intent_id"] if not post_event else None
            values = self.find_snapshot_values(event_index, post_event)
        else:
            size, color, annot, target, values = None, None, None, None, None
            print(f"{spadl_type} by {possessor}")

        if hypo_target is None:
            next_player_id = self.match.events.at[event_index, "next_player_id"]
            next_type = self.match.events.at[event_index, "next_type"]
            node_marks = dict()

            # if not pd.isna(target) and not target.endswith("_goal"):
            #     node_marks["black"] = [target]

            if not pd.isna(receiver):
                end_x = self.match.events.at[event_index, "end_x"]
                end_y = self.match.events.at[event_index, "end_y"]
                arrows = [(possessor, (end_x, end_y))]
                if next_type in config.DEFENSIVE_TOUCH:
                    node_marks["gold"] = [next_player_id]
                elif next_type not in config.SET_PIECE_OOP:
                    node_marks["gold"] = [receiver]
            else:
                arrows = []

        else:
            hypo_target = f"{possessor[:4]}_goal" if hypo_target == -1 else f"{possessor[:4]}_{hypo_target}"
            node_marks = dict() if hypo_target.endswith("_goal") else {"black": [hypo_target]}
            arrows = [(target, hypo_target)] if spadl_type == "tackle" else [(possessor, hypo_target)]

        if "posterior" in [size, color, annot]:
            target = self.actions.at[event_index, "intent_id"] if hypo_target is None else hypo_target
            if "posterior" in [size, color, annot]:
                mask = (
                    (self.posteriors["index"] == event_index)
                    & (self.posteriors["option"] == target)
                    & (self.posteriors["defender"] != "out")
                )
                posteriors_i = self.posteriors[mask].set_index("defender")["posterior"].astype(float)

        if "player_credit" in [size, color, annot]:
            assert self.credits is not None
            mask = (self.credits["index"] == event_index) & (self.credits["defender"] != "out")
            credits_i = self.credits[mask].set_index("defender")["player_credit"]

        data_args = {"snapshot": frame_data, "ball_xy": ball_xy, "player_marks": node_marks}
        # data_args = {"snapshot": frame_data, "arrows": arrows}
        for k, col_name in {"player_sizes": size, "player_colors": color, "player_annots": annot}.items():
            if col_name is not None:
                if col_name == "posterior":
                    data_args[k] = posteriors_i
                elif col_name == "player_credit":
                    data_args[k] = credits_i
                else:
                    data_args[k] = values[col_name]

        if show_edges:
            data_index = torch.argwhere(self.match.labels[:, 0] == event_index).item()
            graph: Data = self.match.graph_features_0[data_index]
            graph = sparsify_edges(graph, "delaunay")
            edge_index = graph.edge_index.cpu().detach().numpy()

            src = [(players[0] + players[1])[i] for i in edge_index[0]]
            dst = [(players[0] + players[1])[i] for i in edge_index[1]]
            data_args["edges"] = np.array([src, dst]).T

        style_args = pd.DataFrame(
            {
                "select": [400, 2400, 0, 0.5],
                "success": [0, 2000, 0.3, 1],
                "posterior": [400, 2000, 0, 0.5],
                "team_credit": [500, 20500, -0.01, 0.01],
                "player_credit": [500, 20500, -0.05, 0.05],
            },
            index=["min_size", "max_size", "min_color", "max_color"],
        ).T
        min_sizes = defaultdict(lambda: 500, style_args["min_size"].to_dict())
        max_sizes = defaultdict(lambda: 20500, style_args["max_size"].to_dict())
        min_colors = defaultdict(lambda: 0.0, style_args["min_color"].to_dict())
        max_colors = defaultdict(lambda: 0.05, style_args["max_color"].to_dict())

        viz = SnapshotVisualizer(**data_args)
        viz.plot(
            smin=min_sizes[size],
            smax=max_sizes[size],
            cmin=min_colors[color],
            cmax=max_colors[color],
            annot_type=annot,
        )

        return values.round(4) if isinstance(values, pd.DataFrame) else None

    def load_components(self, result_dir="data/ajax/defcon_components"):
        match_id = self.match.lineup["stats_perform_match_id"].iloc[0]

        self.select_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/select_probs_0.parquet")
        self.success_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/success_probs_0.parquet")
        self.s_score_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/s_score_probs_0.parquet")
        self.f_score_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/f_score_probs_0.parquet")
        self.s_concede_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/s_concede_probs_0.parquet")
        self.f_concede_probs_0 = pd.read_parquet(f"{result_dir}/{match_id}/f_concede_probs_0.parquet")

        self.select_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/select_probs_1.parquet")
        self.success_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/success_probs_1.parquet")
        self.s_score_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/s_score_probs_1.parquet")
        self.f_score_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/f_score_probs_1.parquet")
        self.s_concede_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/s_concede_probs_1.parquet")
        self.f_concede_probs_1 = pd.read_parquet(f"{result_dir}/{match_id}/f_concede_probs_1.parquet")

        self.posteriors = pd.read_parquet(f"{result_dir}/{match_id}/posteriors.parquet")

    def save_components(self, result_dir="data/ajax/defcon_components"):
        match_id = self.match.lineup["stats_perform_match_id"].iloc[0]
        os.makedirs(f"{result_dir}/{match_id}", exist_ok=True)

        self.select_probs_0.to_parquet(f"{result_dir}/{match_id}/select_probs_0.parquet")
        self.success_probs_0.to_parquet(f"{result_dir}/{match_id}/success_probs_0.parquet")
        self.s_score_probs_0.to_parquet(f"{result_dir}/{match_id}/s_score_probs_0.parquet")
        self.f_score_probs_0.to_parquet(f"{result_dir}/{match_id}/f_score_probs_0.parquet")
        self.s_concede_probs_0.to_parquet(f"{result_dir}/{match_id}/s_concede_probs_0.parquet")
        self.f_concede_probs_0.to_parquet(f"{result_dir}/{match_id}/f_concede_probs_0.parquet")

        self.select_probs_1.to_parquet(f"{result_dir}/{match_id}/select_probs_1.parquet")
        self.success_probs_1.to_parquet(f"{result_dir}/{match_id}/success_probs_1.parquet")
        self.s_score_probs_1.to_parquet(f"{result_dir}/{match_id}/s_score_probs_1.parquet")
        self.f_score_probs_1.to_parquet(f"{result_dir}/{match_id}/f_score_probs_1.parquet")
        self.s_concede_probs_1.to_parquet(f"{result_dir}/{match_id}/s_concede_probs_1.parquet")
        self.f_concede_probs_1.to_parquet(f"{result_dir}/{match_id}/f_concede_probs_1.parquet")

        self.posteriors.to_parquet(f"{result_dir}/{match_id}/posteriors.parquet")
