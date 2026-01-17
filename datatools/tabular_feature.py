import argparse
import os
import sys
from datetime import datetime
from typing import List

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import datatools.preprocess as proc
from datatools import config, utils
from datatools.event_xg import EventXGModel
from datatools.match import Match


def calculate_event_features(
    match: Match,
    frame: float,
    possessor: str,
    target: str = None,
    is_one_touch: bool = False,
    is_set_piece: bool = False,
    event_xg: float = 0.0,
    eps=1e-6,
) -> np.ndarray:
    snapshot = match.tracking.loc[frame].dropna(how="all").copy()

    active_keepers = match.phases.at[snapshot["phase_id"], "active_keepers"]
    if not match.include_keepers:
        keeper_cols = [c for c in snapshot.columns if "_".join(c.split("_")[:2]) in active_keepers]
        snapshot = snapshot.drop(keeper_cols).copy()

    poss_x = snapshot[f"{possessor}_x"]
    poss_y = snapshot[f"{possessor}_y"]
    poss_vx = snapshot[f"{possessor}_vx"]
    poss_vy = snapshot[f"{possessor}_vy"]
    poss_speed = snapshot[f"{possessor}_speed"]
    poss_accel = snapshot[f"{possessor}_speed"]

    home_cols = [c for c in snapshot.dropna().index if c.startswith("home")]
    away_cols = [c for c in snapshot.dropna().index if c.startswith("away")]
    opponent_cols = away_cols if possessor.startswith("home") else home_cols
    oppo_x = snapshot[opponent_cols[0::6]].values.flatten().astype(float)
    oppo_y = snapshot[opponent_cols[1::6]].values.flatten().astype(float)

    oppo_keeper = [p for p in active_keepers if p[:4] != possessor[:4] and f"{p}_x" in opponent_cols[0::6]][0]
    keeper_x = snapshot[f"{oppo_keeper}_x"]
    keeper_y = snapshot[f"{oppo_keeper}_y"]

    if possessor[:4] == "away":
        poss_x = config.FIELD_SIZE[0] - poss_x
        poss_y = config.FIELD_SIZE[1] - poss_y
        poss_vx = -poss_vx
        poss_vy = -poss_vy
        oppo_x = config.FIELD_SIZE[0] - oppo_x
        oppo_y = config.FIELD_SIZE[1] - oppo_y
        keeper_x = config.FIELD_SIZE[0] - keeper_x
        keeper_y = config.FIELD_SIZE[1] - keeper_y

    goal_x = config.FIELD_SIZE[0]
    goal_y = config.FIELD_SIZE[1] / 2
    p_goal_dx, p_goal_dy, p_goal_dist = utils.calc_dist(poss_x, poss_y, goal_x, goal_y)
    _, _, oppo_goal_dists = utils.calc_dist(oppo_x, oppo_y, goal_x, goal_y)
    _, _, p_oppo_dists = utils.calc_dist(oppo_x, oppo_y, poss_x, poss_y)

    p_nearby_opponents = np.sum(p_oppo_dists <= 3)
    p_nearest_opponent = np.min(p_oppo_dists)
    p_closer_opponents_to_goal = sum(oppo_goal_dists < p_goal_dist)

    poss_oppo_x = np.concatenate(([poss_x], oppo_x))
    poss_oppo_y = np.concatenate(([poss_y], oppo_y))
    is_teammate = np.concatenate(([1], np.zeros_like(oppo_x)))
    p_blockers = utils.count_potential_blockers(goal_x, goal_y, poss_oppo_x, poss_oppo_y, is_teammate)[0]

    ball_z = snapshot["ball_z"] if "ball_z" in snapshot.index else 0.0
    keeper_angle = utils.calc_angle(poss_x, poss_y, keeper_x, keeper_y, goal_x, goal_y)

    features = [
        float(is_one_touch),
        float(is_set_piece),
        poss_x,
        poss_y,
        poss_vx,
        poss_vy,
        poss_speed,
        poss_accel,
        p_goal_dist,
        p_goal_dx / (p_goal_dist + eps),
        p_goal_dy / (p_goal_dist + eps),
        p_nearby_opponents,
        p_nearest_opponent,
        p_closer_opponents_to_goal,
        p_blockers,
        ball_z,
        keeper_x,
        keeper_y,
        np.cos(keeper_angle),
        np.sin(keeper_angle),
        event_xg,
    ]

    if not pd.isna(target):
        target_x = snapshot[f"{target}_x"]
        target_y = snapshot[f"{target}_y"]
        target_vx = snapshot[f"{target}_vx"]
        target_vy = snapshot[f"{target}_vy"]
        target_speed = snapshot[f"{target}_speed"]
        target_accel = snapshot[f"{target}_speed"]

        if target[:4] == "away":
            target_x = config.FIELD_SIZE[0] - target_x
            target_y = config.FIELD_SIZE[1] - target_y
            target_vx = -target_vx
            target_vy = -target_vy

        t_goal_dx, t_goal_dy, t_goal_dist = utils.calc_dist(target_x, target_y, goal_x, goal_y)
        _, _, t_oppo_dists = utils.calc_dist(oppo_x, oppo_y, target_x, target_y)
        t_nearby_opponents = np.sum(t_oppo_dists <= 3)
        t_nearest_opponent = np.min(t_oppo_dists)
        t_closer_opponents_to_goal = sum(oppo_goal_dists < t_goal_dist)

        pass_dx, pass_dy, pass_dist = utils.calc_dist(poss_x, poss_y, target_x, target_y)
        poss_vangle = utils.calc_angle(target_vx, target_vy, poss_vx, poss_vy, eps=eps)

        target_oppo_x = np.concatenate(([target_x], oppo_x))
        target_oppo_y = np.concatenate(([target_y], oppo_y))
        is_teammate = np.concatenate(([1], np.zeros_like(oppo_x)))

        t_blockers = utils.count_potential_blockers(
            goal_x,
            goal_y,
            target_oppo_x,
            target_oppo_y,
            is_teammate,
        )[0]
        pass_interceptors = utils.count_potential_interceptors(
            poss_x,
            poss_y,
            target_oppo_x,
            target_oppo_y,
            is_teammate,
            corridor_width=10.0,
        )[0]
        pass_nearest_opponent = utils.find_nearest_opponent_to_pass(
            poss_x,
            poss_y,
            target_oppo_x,
            target_oppo_y,
            is_teammate,
        )[0]

        features += [
            target_x,
            target_y,
            target_vx,
            target_vy,
            target_speed,
            target_accel,
            t_goal_dist,
            t_goal_dx / (t_goal_dist + eps),
            t_goal_dy / (t_goal_dist + eps),
            t_nearby_opponents,
            t_nearest_opponent,
            t_closer_opponents_to_goal,
            t_blockers,
            pass_dist,
            pass_dx / (pass_dist + eps),
            pass_dy / (pass_dist + eps),
            pass_interceptors,
            pass_nearest_opponent,
            np.cos(poss_vangle),
            np.sin(poss_vangle),
        ]

    else:
        features += [-100.0] * 20  # Reserve placeholder values when intent target is missing.

    return torch.tensor(features, dtype=torch.float32)


def construct_tabular_features(
    match: Match,
    action_type: str = "shot",
    event_xg_model: EventXGModel = None,
    post_action=False,
    verbose=True,
) -> torch.Tensor:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    if post_action:
        match.actions = match.label_post_actions(match.actions)

    one_touch_mask = match.events["frame_id"] < match.events["receive_frame_id"].shift(1) + 3
    incoming_mask = match.events["spadl_type"].isin(config.INCOMING)

    feature_tensors: List[torch.Tensor] = []

    for period in match.events["period_id"].unique():
        period_actions: pd.DataFrame = match.actions[match.actions["period_id"] == period]
        action_indices = np.intersect1d(period_actions.index, match.labels[:, 0].long().numpy())
        iterator = tqdm(action_indices, desc=f"Period {period}") if verbose else action_indices

        if event_xg_model is not None:
            event_xg_features = event_xg_model.calc_shot_features(match.events.loc[action_indices])
            event_xg_values = event_xg_model.pred(event_xg_features)
        else:
            event_xg_values = pd.Series(0, index=action_indices)

        for i in iterator:
            if post_action:
                frame = period_actions.at[i, "end_frame_id"]
                possessor = period_actions.at[i, "end_player_id"]
                one_touch = True
                set_piece = period_actions.at[i, "next_type"] in config.SET_PIECE

            else:
                frame = period_actions.at[i, "frame_id"]
                possessor = period_actions.at[i, "object_id"]
                one_touch = one_touch_mask.at[i] or incoming_mask.at[i]
                set_piece = period_actions.at[i, "spadl_type"] in config.SET_PIECE

            if not pd.isna(frame):
                target = period_actions.at[i, "intent_id"] if "pass" in action_type else None
                args = [match, frame, possessor, target, one_touch, set_piece, event_xg_values.loc[i]]
                event_features = calculate_event_features(*args)
                feature_tensors.append(event_features)
            else:
                feature_tensors.append(torch.full([41], -100.0))

    return torch.stack(feature_tensors)  # [B, x]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, choices=["pass", "pass_dribble", "shot"])
    parser.add_argument("--split", type=str, required=False, default="train", choices=["train", "test"])
    parser.add_argument("--return_type", type=str, required=False, default="disc_0.9", help="way of defining future xG")
    parser.add_argument("--augment", action="store_true", default=False, help="augment failed shots by far passes")
    args, _ = parser.parse_known_args()

    feature_dir = f"data/ajax/features/{args.action_type}_tabular"
    label_dir = f"data/ajax/features/{args.action_type}_labels"
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    event_files = np.sort(os.listdir("data/ajax/event_synced"))
    match_ids = np.sort([f.split(".")[0] for f in event_files if f.endswith(".csv")])

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet")
    lineups["game_id"] = lineups["stats_perform_match_id"]
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    match_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]

    if args.split == "train":
        match_ids = match_dates[(match_dates.index.isin(match_ids)) & (match_dates < datetime(2024, 8, 1))].index
    else:  # split == "test"
        match_ids = match_dates[(match_dates.index.isin(match_ids)) & (match_dates >= datetime(2024, 8, 1))].index

    n_matches = len(match_ids)
    action_type = f"{args.action_type}_augment" if args.augment else args.action_type

    event_xg_model = EventXGModel(unblocked=True)  # To use event-data-based xG as a feature
    event_xg_model.train(verbose=False)

    if args.action_type == "shot":
        features = []
        labels = []

    for i, match_id in enumerate(match_ids):
        events = pd.read_csv(f"data/ajax/event_synced/{match_id}.csv", header=0, parse_dates=["utc_timestamp"])
        tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id]

        match = Match(events, tracking, match_lineup, action_type, include_goals=False)
        match_date = match_dates[match_id].date()
        match_name = " vs ".join(match_lineup["contestant_name"].unique())
        print(f"\n[{i+1}/{n_matches}] {match_id}: {match_name} on {match_date}")

        if args.action_type == "shot":
            match.labels = match.construct_labels()
            match.tabular_features_0 = construct_tabular_features(match, verbose=False)
            features.append(match.tabular_features_0)

            action_indices = match.labels[:, 0].numpy().astype(int)
            assert match.tabular_features_0.shape[0] == match.labels.shape[0]
            assert np.all(np.sort(action_indices) == action_indices)

            match.actions["match_id"] = match_id
            labels.append(match.actions.loc[action_indices])

        else:
            if args.return_type.startswith("disc"):
                gamma = float(args.return_type.split("_")[-1])
                match.labels = match.construct_labels(discount_xg=True, gamma=gamma)
            elif args.return_type.startswith("next"):
                lookahead_len = int(args.return_type.split("_")[-1])
                match.labels = match.construct_labels(discount_xg=False, lookahead_len=lookahead_len)

            action_indices = match.labels[:, 0].numpy().astype(int)
            assert match.tabular_features_0.shape[0] == match.labels.shape[0]
            assert np.all(np.sort(action_indices) == action_indices)

            match.tabular_features_0 = construct_tabular_features(match, event_xg_model=event_xg_model, verbose=True)
            torch.save(match.tabular_features_0, f"{feature_dir}/{match_id}.pt")
            torch.save(match.labels, f"{label_dir}/{match_id}.pt")
            print(f"Successfully saved for {match.labels.shape[0]} events.")

    if args.action_type == "shot":
        features = torch.cat(features)
        torch.save(features, f"{feature_dir}/{args.split}.pt")

        labels = pd.concat(labels)
        labels.to_parquet(f"{label_dir}/{args.split}.parquet")
