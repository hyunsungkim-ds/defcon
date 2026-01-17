import argparse
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

import datatools.preprocess as proc
from datatools import config, utils
from datatools.match import Match


def calculate_event_features(
    match: Match,
    snapshot: pd.DataFrame,
    possessor: str,
    extend=False,
    sequential=False,
    eps=1e-6,
) -> np.ndarray:
    seq_len = len(snapshot) if sequential else 1

    phase_id = snapshot["phase_id"].iloc[0]
    active_players = match.phases.at[phase_id, "active_players"]
    active_keepers = match.phases.at[phase_id, "active_keepers"]

    if not match.include_keepers:
        keeper_cols = [c for c in snapshot.columns if "_".join(c.split("_")[:2]) in active_keepers]
        snapshot = snapshot.drop(keeper_cols, axis=1).copy()

    home_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("home")]
    away_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("away")]
    if not match.include_goals:
        home_cols = [c for c in home_cols if not c.startswith("home_goal")]
        away_cols = [c for c in away_cols if not c.startswith("away_goal")]

    player_cols = home_cols + away_cols if possessor.startswith("home") else away_cols + home_cols
    players = [c[:-2] for c in player_cols if c.endswith("_x")]
    assert set(players) - {"home_goal", "away_goal"} == set(active_players)

    is_teammate = np.tile([int(p[:4] == possessor[:4]) for p in players], (seq_len, 1))
    is_keeper = np.tile([int(p in active_keepers) for p in players], (seq_len, 1))
    is_goal = np.tile([int("goal" in p) for p in players], (seq_len, 1))

    if not sequential:
        snapshot = snapshot[-1:].copy()

    player_x = snapshot[player_cols[0::6]].values
    player_y = snapshot[player_cols[1::6]].values
    player_vx = snapshot[player_cols[2::6]].values
    player_vy = snapshot[player_cols[3::6]].values
    player_speeds = snapshot[player_cols[4::6]].values
    player_accels = snapshot[player_cols[5::6]].values

    if possessor.endswith("out"):
        poss_x = snapshot["ball_x"].values
        poss_y = snapshot["ball_y"].values
    poss_x = snapshot[f"{possessor}_x"].values[:, np.newaxis]
    poss_y = snapshot[f"{possessor}_y"].values[:, np.newaxis]
    poss_vx = snapshot[f"{possessor}_vx"].values[:, np.newaxis]
    poss_vy = snapshot[f"{possessor}_vy"].values[:, np.newaxis]

    # Make the attacking team plays from left to right
    if possessor[:4] == "away":
        player_x = config.FIELD_SIZE[0] - player_x
        player_y = config.FIELD_SIZE[1] - player_y
        player_vx = -player_vx
        player_vy = -player_vy

        poss_x = config.FIELD_SIZE[0] - poss_x
        poss_y = config.FIELD_SIZE[1] - poss_y
        poss_vx = -poss_vx
        poss_vy = -poss_vy

    goal_x = config.FIELD_SIZE[0]
    goal_y = config.FIELD_SIZE[1] / 2
    goal_dx, goal_dy, goal_dists = utils.calc_dist(player_x, player_y, goal_x, goal_y)

    if "ball_z" in snapshot.columns:
        ball_z = np.ones((seq_len, len(players))) * snapshot["ball_z"].iloc[-1]
    else:
        ball_z = np.zeros((seq_len, len(players)))

    is_possessor = np.tile((np.array(players) == possessor).astype(int), (seq_len, 1))
    poss_dx, poss_dy, poss_dists = utils.calc_dist(player_x, player_y, poss_x, poss_y)
    poss_vangles = utils.calc_angle(player_vx, player_vy, poss_vx, poss_vy, eps=eps)

    event_features = [
        # Binary features
        is_teammate,
        is_keeper,
        is_goal,
        # Possessor-independent features
        player_x,
        player_y,
        player_vx,
        player_vy,
        player_speeds,
        player_accels,
        goal_dists,
        goal_dx / (goal_dists + eps),  # Cosine between each player-goal line and the x-axis
        goal_dy / (goal_dists + eps),  # Sine between each player-goal line and the x-axis
        ball_z,
        # Possessor features
        is_possessor,
        poss_dists,
        poss_dx / (poss_dists + eps),  # Cosine between each player-possessor line and the x-axis
        poss_dy / (poss_dists + eps),  # Sine between each player-possessor line and the x-axis
        np.cos(poss_vangles),  # Cosine between each player's velocity and the possessor's velocity
        np.sin(poss_vangles),  # Sine between each player's velocity and the possessor's velocity
    ]

    if extend:
        player_xy = np.stack([player_x[-1], player_y[-1]], axis=-1)
        dist_mat = cdist(player_xy, player_xy)

        opponent_mask = is_teammate[-1] != is_teammate[-1][:, np.newaxis]
        neighbor_mask = (dist_mat < 3.0) & opponent_mask
        nearby_opponents_to_target: np.ndarray = neighbor_mask.sum(axis=1)

        opponent_dists = np.where(opponent_mask, dist_mat, np.inf)
        nearest_opponent_to_target: np.ndarray = opponent_dists.min(axis=1)

        closer_mask = (goal_dists[-1][np.newaxis, :] < goal_dists[-1][:, np.newaxis]) & opponent_mask
        closer_opponents_to_goal: np.ndarray = closer_mask.sum(axis=1)

        args = [poss_x[-1, 0], poss_y[-1, 0], player_x[-1], player_y[-1], is_teammate[-1]]
        nearest_opponent_to_pass = utils.find_nearest_opponent_to_pass(*args)
        potential_interceptors = utils.count_potential_interceptors(*args)
        potential_blockers = utils.count_potential_blockers(goal_x, goal_y, *args[2:])

        event_features.extend(
            [
                nearby_opponents_to_target[np.newaxis, :],
                nearest_opponent_to_target[np.newaxis, :],
                closer_opponents_to_goal[np.newaxis, :],
                nearest_opponent_to_pass[np.newaxis, :],
                potential_interceptors[np.newaxis, :],
                potential_blockers[np.newaxis, :],
            ]
        )

    return np.stack(event_features, axis=-1)  # [T, N, x]


def construct_graph_features(match: Match, extend=True, post_action=False, verbose=True) -> List[Data]:
    if "ball_accel" not in match.tracking.columns:
        match.tracking = proc.calc_physical_features(match.tracking, match.fps)

    if post_action:
        match.actions = match.label_post_actions(match.actions)

    feature_tensors: List[torch.Tensor] = []

    for period in match.events["period_id"].unique():
        period_actions: pd.DataFrame = match.actions[match.actions["period_id"] == period]
        period_tracking: pd.DataFrame = match.tracking[match.tracking["period_id"] == period]
        action_indices = np.intersect1d(period_actions.index, match.labels[:, 0].long().numpy())
        iterator = tqdm(action_indices, desc=f"Period {period}") if verbose else action_indices

        for i in iterator:
            if post_action:
                frame = period_actions.at[i, "end_frame_id"]
                possessor = period_actions.at[i, "end_player_id"]
            else:
                frame = period_actions.at[i, "frame_id"]
                possessor = period_actions.at[i, "object_id"]

            if not pd.isna(frame) and possessor.split("_")[0] in ["home", "away"]:
                snapshot = period_tracking.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
                event_features = calculate_event_features(match, snapshot, possessor, extend)
                event_features = torch.tensor(event_features[0], dtype=torch.float32)

                missing_players = match.max_players - event_features.shape[0]
                padding_features = -torch.ones((missing_players, event_features.shape[-1]))

                event_features = torch.cat([event_features, padding_features], 0)
                feature_tensors.append(event_features)

            else:
                padding_features = -torch.ones((match.max_players, event_features.shape[-1]))
                feature_tensors.append(padding_features)

    node_attr = torch.stack(feature_tensors, axis=0)  # [B, N, x]
    distances = torch.cdist(node_attr[..., 3:5], node_attr[..., 3:5], p=2)  # [B, N, N]
    teammates = (node_attr[..., 0].unsqueeze(-1) == node_attr[..., 0].unsqueeze(-2)).float()  # [B, N, N]

    feature_graphs: List[Data] = []

    for i in range(node_attr.shape[0]):
        if node_attr[i, 0, 0] == -1:
            feature_graphs.append(None)

        else:
            node_mask = node_attr[i, :, 0] != -1
            node_attr_i = node_attr[i][node_mask]

            distances_i = distances[i][node_mask][:, node_mask]
            teammates_i = teammates[i][node_mask][:, node_mask]
            edge_index, _ = dense_to_sparse(torch.ones_like(distances_i))

            distances_i = distances_i[edge_index[0], edge_index[1]]
            teammates_i = teammates_i[edge_index[0], edge_index[1]]
            edge_attr_i = torch.stack([distances_i, teammates_i], dim=-1)  # [N * N, 2]

            graph = Data(x=node_attr_i, edge_index=edge_index.clone(), edge_attr=edge_attr_i)
            feature_graphs.append(graph)

    return feature_graphs


def augment_blocked_actions(match: Match, max_block_dist=5, max_block_angle=15) -> Tuple[List[Data], torch.Tensor]:
    augmented_features = []
    augmented_labels = []

    action_indices = match.labels[:, 0].numpy().astype(int)
    tqdm_desc = "Augmenting features and labels"

    for i, action_index in enumerate(tqdm(action_indices, desc=tqdm_desc)):
        augmented_features.append(match.graph_features_0[i])
        augmented_labels.append(match.labels[i])

        if match.actions.at[action_index, "spadl_type"] in config.SET_PIECE:
            continue

        frame = match.actions.at[action_index, "frame_id"]
        possessor = match.actions.at[action_index, "object_id"]
        real_intent = match.actions.at[action_index, "intent_id"]
        snapshot: pd.Series = match.tracking.loc[frame]

        teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_.*_x", c)]
        team_x = snapshot[[f"{p}_x" for p in teammates]].values
        team_y = snapshot[[f"{p}_y" for p in teammates]].values

        team_dist_x = (team_x - snapshot[f"{possessor}_x"]).astype(float)
        team_dist_y = (team_y - snapshot[f"{possessor}_y"]).astype(float)
        team_dists = np.sqrt(team_dist_x**2 + team_dist_y**2)

        oppo_team = "away" if possessor[:4] == "home" else "home"
        opponents = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{oppo_team}_.*_x", c)]
        oppo_x = snapshot[[f"{p}_x" for p in opponents]].values
        oppo_y = snapshot[[f"{p}_y" for p in opponents]].values

        oppo_dist_x = (oppo_x - snapshot[f"{possessor}_x"]).astype(float)
        oppo_dist_y = (oppo_y - snapshot[f"{possessor}_y"]).astype(float)
        oppo_dists = np.sqrt(oppo_dist_x**2 + oppo_dist_y**2)
        blockers = np.array(opponents)[np.where(oppo_dists < max_block_dist)[0]][:3]

        if match.include_goals:
            goal_index = teammates.index(f"{possessor[:4]}_goal")

        for blocker in blockers:
            poss_x = snapshot[f"{possessor}_x"]
            poss_y = snapshot[f"{possessor}_y"]
            block_x = snapshot[f"{blocker}_x"]
            block_y = snapshot[f"{blocker}_y"]

            team_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, team_x, team_y)
            oppo_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, oppo_x, oppo_y)
            blocked_teammates = np.where(team_angles < max_block_angle / 180 * np.pi)[0].tolist()
            blocked_opponents = np.where(oppo_angles < max_block_angle / 180 * np.pi)[0].tolist()
            blocked_opponents = [k for k in blocked_opponents if k != opponents.index(blocker)]

            if not blocked_opponents:
                continue

            close_teammates = np.where(team_dists < oppo_dists[blocked_opponents].max() - 10)[0].tolist()
            if match.include_goals and team_dists[goal_index] < 40:
                close_teammates.append(goal_index)  # Assume that the shot was prevented if goal distance was < 30
            blocked_intent_indices = list(set(blocked_teammates) & set(close_teammates))
            blocked_intents = [p for p in np.array(teammates)[blocked_intent_indices] if p != real_intent]

            for blocked_intent in blocked_intents:
                augmented_labels_i = match.labels[i].clone()

                if blocked_intent.endswith("goal"):  # An augmented shot that would be blocked
                    augmented_labels_i[1] = 0
                    augmented_labels_i[3] = 1
                else:  # An augmented pass that would be blocked
                    augmented_labels_i[1] = 1
                    augmented_labels_i[3] = 0

                augmented_labels_i[5] = teammates.index(blocked_intent)
                augmented_labels_i[6] = (teammates + opponents).index(blocker)
                augmented_labels_i[-7] = 0  # Indicating that this is an augmented event and not a real one
                augmented_labels_i[-6] = 1  # Indicating that this is a blocked event
                augmented_labels_i[-5] = 0  # Indicating that this is a failed event

                augmented_features.append(match.graph_features_0[i].clone())
                augmented_labels.append(augmented_labels_i)

    augmented_labels = torch.stack(augmented_labels, dim=0)
    return augmented_features, augmented_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_type", type=str, required=False, default="all", choices=["all", "shot_augment"])
    parser.add_argument("--split", type=str, required=False, default="train", choices=["train", "test"])
    parser.add_argument("--return_type", type=str, required=False, default="disc_0.9", help="way of defining future xG")
    parser.add_argument("--post_action", action="store_true", default=False, help="construct post-action features")
    parser.add_argument("--augment_blocks", action="store_true", default=False)
    args, _ = parser.parse_known_args()

    if args.action_type.startswith("shot"):
        args.action_type = "shot_augment"
        feature_dir = "data/ajax/features/augmented_shot_graphs"
        label_dir = "data/ajax/features/augmented_shot_labels"
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

    else:  # args.actions_type == "all"
        feature_dir = "data/ajax/features/action_graphs"
        label_dir = f"data/ajax/features/action_labels_{args.return_type}"
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if args.post_action:
            post_feature_dir = "data/ajax/features/post_action_graphs"
            os.makedirs(post_feature_dir, exist_ok=True)

    if args.augment_blocks:
        augmented_feature_dir = "data/ajax/features/augmented_graphs"
        augmented_label_dir = "data/ajax/features/augmented_labels"
        os.makedirs(augmented_feature_dir, exist_ok=True)
        os.makedirs(augmented_label_dir, exist_ok=True)

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

    for i, match_id in enumerate(match_ids):
        events = pd.read_csv(f"data/ajax/event_synced/{match_id}.csv", header=0, parse_dates=["utc_timestamp"])
        tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id]

        match = Match(events, tracking, match_lineup, args.action_type, include_goals=True)
        match_date = match_dates[match_id].date()
        match_name = " vs ".join(match_lineup["contestant_name"].unique())
        print(f"\n[{i+1}/{n_matches}] {match_id}: {match_name} on {match_date}")

        if args.return_type.startswith("disc"):
            gamma = float(args.return_type.split("_")[-1])
            match.labels = match.construct_labels(discount_xg=True, gamma=gamma)
        if args.return_type.startswith("next"):
            lookahead_len = int(args.return_type.split("_")[-1])
            match.labels = match.construct_labels(discount_xg=False, lookahead_len=lookahead_len)

        action_indices = match.labels[:, 0].numpy().astype(int)
        assert np.all(np.sort(action_indices) == action_indices)
        torch.save(match.labels, f"{label_dir}/{match_id}.pt")

        print("Constructing graph features for actions...")
        match.graph_features_0 = construct_graph_features(match, extend=True, post_action=False)
        torch.save(match.graph_features_0, f"{feature_dir}/{match_id}.pt")

        if args.post_action:
            print("Constructing graph features for post-actions...")
            match.graph_features_1 = construct_graph_features(match, extend=True, post_action=True)
            torch.save(match.graph_features_1, f"{post_feature_dir}/{match_id}.pt")

        print(f"Successfully saved for {match.labels.shape[0]} events.")

        if args.augment_blocks:
            augmented_graph_features, augmented_labels = augment_blocked_actions(match)
            torch.save(augmented_graph_features, f"{augmented_feature_dir}/{match_id}.pt")
            torch.save(augmented_labels, f"{augmented_label_dir}/{match_id}.pt")
            print(f"Successfully saved for {augmented_labels.shape[0]} augmented events.")
