import re
from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rdp import rdp
from scipy.spatial import Delaunay
from shapely import vectorized
from shapely.geometry import Point, Polygon
from torch_geometric.data import Batch, Data

from datatools import config


def calc_dist(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray):
    dist_x = (x - ref_x).astype(float) if isinstance(x, np.ndarray) else x - ref_x
    dist_y = (y - ref_y).astype(float) if isinstance(y, np.ndarray) else y - ref_y
    return dist_x, dist_y, np.sqrt(dist_x**2 + dist_y**2)


def calc_angle(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray = None,
    cy: np.ndarray = None,
    eps: float = 1e-6,
) -> np.ndarray:
    if cx is None or cy is None:
        # Calculate angles between the vectors a and b
        a_len = np.sqrt(ax**2 + ay**2) + eps
        b_len = np.sqrt(bx**2 + by**2) + eps
        cos = np.clip((ax * bx + ay * by) / (a_len * b_len), -1, 1)

    else:
        # Calculate angles between the lines AB and AC
        ab_x = (bx - ax).astype(float)
        ab_y = (by - ay).astype(float)
        ab_len = np.sqrt(ab_x**2 + ab_y**2) + eps

        ac_x = (cx - ax).astype(float)
        ac_y = (cy - ay).astype(float)
        ac_len = np.sqrt(ac_x**2 + ac_y**2) + eps

        cos = np.clip((ab_x * ac_x + ab_y * ac_y) / (ab_len * ac_len), -1, 1)

    return np.arccos(cos)


# Identify whether the shot trajectory is erroneous and the shot hits a post
def is_shot_anomaly(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    event_index: int,
    min_segment_len: float = 2.0,
    min_sim: float = 0.7,
    max_frames: int = 25,
) -> Tuple[bool, bool]:
    frame = events.at[event_index, "frame_id"]
    receive_frame = events.at[event_index, "receive_frame_id"]
    ball_xy = tracking.loc[frame:receive_frame, ["ball_x", "ball_y"]]
    simplified = np.array(rdp(ball_xy, epsilon=0.5))

    dirs = np.diff(simplified, axis=0)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    valid_mask = np.array((norms > min_segment_len).flatten().tolist() + [True])
    simplified = simplified[valid_mask]

    dirs = np.diff(simplified, axis=0)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_norm = dirs / (norms + 1e-8)

    woodwork = False

    for i in range(len(dirs_norm) - 1):
        sim = np.dot(dirs_norm[i], dirs_norm[i + 1])

        if sim < min_sim:
            goalpost_dist = np.linalg.norm(config.GOAL_XY - simplified[[i + 1]], axis=1)

            if goalpost_dist.min() < 1:
                woodwork = True
            elif receive_frame - frame > max_frames:
                return True, woodwork

    return False, woodwork


# To make the attacking team always play from right to left (not needed for the current dataset)
def rotate_events_for_xg(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    for i in events["period_id"].unique():
        shots = events[(events["period_id"] == i) & events["spadl_type"].isin(config.SHOT)]

        if not shots.empty:
            home_shot_x = shots.loc[shots["object_id"].str.startswith("home"), "start_x"].mean()
            if home_shot_x > config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_shot_x = shots.loc[shots["object_id"].str.startswith("away"), "start_y"].mean()
            if away_shot_x > config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

        else:
            gk_events = events[(events["period_id"] == i) & (events["advanced_position"] == "goal_keeper")]

            home_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("home"), "start_x"].mean()
            if home_gk_x < config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("away"), "start_x"].mean()
            if away_gk_x < config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

    return events


def label_intended_receivers(
    actions: pd.DataFrame,
    tracking: pd.DataFrame,
    action_type="shot",
    max_angle=45,
) -> pd.DataFrame:
    def downscale_closer_candidates(dists: np.ndarray) -> np.ndarray:
        out = np.empty_like(dists)
        out[dists < -10] = 0.5
        out[(dists >= -10) & (dists < 0)] = dists[(dists >= -10) & (dists < 0)] / 20 + 1
        out[dists >= 0] = 1.0
        return out

    actions = actions.copy()
    actions["intent_id"] = pd.Series(index=actions.index, dtype="object")

    for i in actions.index:
        event_frame = actions.at[i, "frame_id"]
        possessor = actions.at[i, "object_id"]
        snapshot: pd.Series = tracking.loc[event_frame]

        receive_frame = actions.at[i, "receive_frame_id"]
        receiver = actions.at[i, "receiver_id"]

        if actions.at[i, "action_type"] not in ["pass", "shot"]:  # Mostly for clearances and dribbles
            actions.at[i, "intent_id"] = possessor

        elif actions.at[i, "action_type"] == "shot" or action_type == "shot":
            actions.at[i, "intent_id"] = f"{possessor[:4]}_goal"

            if actions.at[i, "action_type"] == "shot":  # For real shots
                if actions.at[i, "success"]:
                    end_x = actions.at[i, "end_x"]
                    if possessor[:4] == "home" and end_x < 10:
                        actions.at[i, "spadl_type"] = "own_goal"
                        actions.at[i, "intent_id"] = possessor
                        actions.at[i, "receiver_id"] = "away_goal"
                    elif possessor[:4] == "away" and end_x > config.FIELD_SIZE[0] - 10:
                        actions.at[i, "spadl_type"] = "own_goal"
                        actions.at[i, "intent_id"] = possessor
                        actions.at[i, "receiver_id"] = "home_goal"
                    else:
                        actions.at[i, "receiver_id"] = f"{possessor[:4]}_goal"

                elif actions.at[i, "next_type"] in config.SET_PIECE_OOP:
                    actions.at[i, "receiver_id"] = "out"

                else:
                    # elif actions.at[i, "next_type"] in config.INCOMING + ["clearance"]:
                    actions.at[i, "receiver_id"] = actions.at[i, "next_player_id"]

            else:  # For passes that would be blocked if they were shots
                actions.at[i, "receiver_id"] = actions.at[i, "blocker_id"]

        elif actions.at[i, "success"]:  # For successful passes
            actions.at[i, "intent_id"] = actions.at[i, "receiver_id"]

        elif not pd.isna(receiver) and not pd.isna(receive_frame):  # For failed passes
            receive_snapshot: pd.Series = tracking.loc[int(receive_frame)]

            teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_\d+_x", c)]
            teammates.remove(possessor)

            start_x = snapshot[f"{possessor}_x"]
            start_y = snapshot[f"{possessor}_y"]
            end_x = receive_snapshot[f"{receiver}_x"] if receiver != "out" else receive_snapshot["ball_x"]
            end_y = receive_snapshot[f"{receiver}_y"] if receiver != "out" else receive_snapshot["ball_y"]
            player_x = receive_snapshot[[f"{p}_x" for p in teammates]].values
            player_y = receive_snapshot[[f"{p}_y" for p in teammates]].values

            pass_dist = calc_dist(start_x, start_y, end_x, end_y)[-1]
            origin_dist_diffs = calc_dist(player_x, player_y, start_x, start_y)[-1] - pass_dist
            weights = downscale_closer_candidates(origin_dist_diffs)
            dest_dists = np.clip(calc_dist(player_x, player_y, end_x, end_y)[-1], 1, None)
            angles = np.clip(calc_angle(start_x, start_y, end_x, end_y, player_x, player_y), 0.01, None)

            max_radian = max_angle / 180 * np.pi
            if np.min(angles) < max_radian:
                scores = weights * (np.min(dest_dists) / dest_dists) * (np.min(angles) / angles)
                scores = np.where(angles < max_radian, scores, 0)
                actions.at[i, "intent_id"] = teammates[np.argmax(scores)]

    return actions


def label_returns(events: pd.DataFrame, lookahead_len: int = 10) -> pd.DataFrame:
    events = events.copy()

    events["team"] = events["object_id"].str[:4]
    events["goal"] = events["expected_goal"].notna() & events["success"]
    events["scores"] = 0.0
    events["concedes"] = 0.0
    events["scores_xg"] = 0.0
    events["concedes_xg"] = 0.0

    for period in events["period_id"].unique():
        period_events = events[events["period_id"] == period]
        labels = period_events[["team", "goal", "expected_goal"]].copy()

        for i in range(lookahead_len):
            shifted_teams = labels["team"].shift(-i)
            shifted_goals = labels[["goal", "expected_goal"]].shift(-i).fillna(0)
            # shifted_returns = labels.shift(-i).fillna(0).infer_objects(copy=False)
            labels[f"sg+{i}"] = shifted_goals["goal"] * (shifted_teams == labels["team"]).astype(int)
            labels[f"cg+{i}"] = shifted_goals["goal"] * (shifted_teams != labels["team"]).astype(int)
            labels[f"sxg+{i}"] = shifted_goals["expected_goal"] * (shifted_teams == labels["team"]).astype(int)
            labels[f"cxg+{i}"] = shifted_goals["expected_goal"] * (shifted_teams != labels["team"]).astype(int)

        scoring_cols = [c for c in labels.columns if c.startswith("sg+")]
        scoring_xg_cols = [c for c in labels.columns if c.startswith("sxg+")]
        conceding_cols = [c for c in labels.columns if c.startswith("cg+")]
        conceding_xg_cols = [c for c in labels.columns if c.startswith("cxg+")]

        events.loc[labels.index, "scores"] = labels[scoring_cols].sum(axis=1).clip(0, 1).astype(int)
        events.loc[labels.index, "scores_xg"] = 1 - (1 - labels[scoring_xg_cols]).prod(axis=1)
        events.loc[labels.index, "concedes"] = labels[conceding_cols].sum(axis=1).clip(0, 1).astype(int)
        events.loc[labels.index, "concedes_xg"] = 1 - (1 - labels[conceding_xg_cols]).prod(axis=1)

    return events


def label_discounted_returns(events: pd.DataFrame, gamma: float = 0.95) -> pd.DataFrame:
    events = events.copy()

    expected_goals = events["expected_goal"].copy().fillna(0)
    events["goal"] = (expected_goals > 0) & events["success"]
    events["scores_xg_disc"] = 0.0
    events["concedes_xg_disc"] = 0.0
    n_events = len(events)

    for i in events.index:
        period_i = events.at[i, "period_id"]
        team_i = events.at[i, "object_id"][:4]

        prob_not_scoring = 1.0
        prob_not_conceding = 1.0

        for j in range(i, n_events):
            if events.at[j, "object_id"][:4] == team_i:  # future shot by a teammate
                prob_not_scoring *= 1 - gamma ** (j - i) * expected_goals.at[j]
            else:  # future shot by an opponent
                prob_not_conceding *= 1 - gamma ** (j - i) * expected_goals.at[j]

            if events.at[j, "goal"]:
                break
            if j + 1 < n_events and events.at[j + 1, "period_id"] != period_i:
                break
            if j + 1 < n_events and events.at[j + 1, "spadl_type"] == "goalkick":
                break

        events.at[i, "scores_xg_disc"] = 1 - prob_not_scoring
        events.at[i, "concedes_xg_disc"] = 1 - prob_not_conceding

    return events


def count_potential_interceptors(
    poss_x: np.ndarray,
    poss_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    is_teammate: np.ndarray,
    corridor_width: float = 10.0,
) -> np.ndarray:
    opponent_x = player_x[is_teammate == 0]
    opponent_y = player_y[is_teammate == 0]
    potential_interceptors = np.zeros(len(player_x))

    for i in np.nonzero(is_teammate == 1)[0]:
        target_x = player_x[i]
        target_y = player_y[i]
        pass_dx = target_x - poss_x
        pass_dy = target_y - poss_y
        pass_len = np.hypot(pass_dx, pass_dy)

        if pass_len < 1e-6:  # Skip the possessor
            continue

        buffer_x = (corridor_width / 2) * (-pass_dy / pass_len)
        buffer_y = (corridor_width / 2) * (pass_dx / pass_len)

        p1 = (poss_x - buffer_x, poss_y - buffer_y)
        p2 = (poss_x + buffer_x, poss_y + buffer_y)
        p3 = (target_x + buffer_x, target_y + buffer_y)
        p4 = (target_x - buffer_x, target_y - buffer_y)

        corridor = Polygon([p1, p2, p3, p4])
        inside_mask = vectorized.contains(corridor, opponent_x, opponent_y)
        potential_interceptors[i] = np.count_nonzero(inside_mask)

    return potential_interceptors


def find_nearest_opponent_to_pass(
    poss_x: np.ndarray,
    poss_y: np.ndarray,
    player_x: np.ndarray,
    player_y: np.ndarray,
    is_teammate: np.ndarray,
) -> np.ndarray:
    teammate_x = player_x[is_teammate == 1][np.newaxis, :]  # [1, teammates]
    teammate_y = player_y[is_teammate == 1][np.newaxis, :]  # [1, teammates]
    opponent_x = player_x[is_teammate == 0][:, np.newaxis]  # [opponents, 1]
    opponent_y = player_y[is_teammate == 0][:, np.newaxis]  # [opponents, 1]

    pass_x = teammate_x - poss_x  # [1, teammates]
    pass_y = teammate_y - poss_y  # [1, teammates]
    pass_len = np.hypot(pass_x, pass_y) + 1e-6  # [1, teammates]

    oppo_rel_x = opponent_x - poss_x
    oppo_rel_y = opponent_y - poss_y
    proj_coeffs = (oppo_rel_x * pass_x + oppo_rel_y * pass_y) / (pass_len**2)  # [opponents, teammates]
    proj_coeffs = np.clip(proj_coeffs, 0, 1)

    oppo_proj_x = poss_x + proj_coeffs * pass_x  # [opponents, teammates]
    oppo_proj_y = poss_y + proj_coeffs * pass_y  # [opponents, teammates]
    dists_to_pass = np.hypot(opponent_x - oppo_proj_x, opponent_y - oppo_proj_y)  # [opponents, teammates]
    min_dists = np.min(dists_to_pass, axis=0)  # [teammates,]

    return np.concatenate([min_dists, np.zeros(len(opponent_x))])  # [players,]


def count_potential_blockers(
    goal_x: float,
    goal_y: float,
    player_x: np.ndarray,
    player_y: np.ndarray,
    teammate_mask: np.ndarray,
    buffer: float = 1.0,
) -> np.ndarray:
    goal_left = (goal_x, goal_y - config.GOAL_SIZE / 2 + buffer)
    goal_right = (goal_x, goal_y + config.GOAL_SIZE / 2 - buffer)

    player_x = np.asarray(player_x, dtype=float)
    player_y = np.asarray(player_y, dtype=float)
    teammate_mask = np.asarray(teammate_mask).astype(bool)  # [players,]
    pairwise_opponent_mask = teammate_mask != teammate_mask[:, np.newaxis]  # [players, players]

    player_xy = np.stack([player_x, player_y], axis=-1)
    potential_blockers = np.zeros(len(player_xy))

    for i in np.nonzero(teammate_mask)[0]:
        tri = Polygon([goal_left, goal_right, tuple(player_xy[i])])
        if not tri.is_valid or tri.area < 1e-3:
            continue

        search_region = tri.buffer(buffer)
        opponent_idxs = np.where(pairwise_opponent_mask[i])[0]
        count = 0

        for idx in opponent_idxs:
            opponent_point = Point(player_x[idx], player_y[idx])
            if search_region.contains(opponent_point):
                count += 1

        potential_blockers[i] = count

    return potential_blockers


def find_nearest_blocker(event: pd.Series, tracking: pd.DataFrame, keepers: np.ndarray) -> int:
    snapshot: pd.Series = tracking.loc[event["frame_id"]]
    event_xy = event[["start_x", "start_y"]].values.tolist()

    goal_x = config.FIELD_SIZE[0] if event["object_id"].startswith("home") else 0
    goal_xy_lower = [goal_x, config.FIELD_SIZE[1] / 2 - 4]
    goal_xy_upper = [goal_x, config.FIELD_SIZE[1] / 2 + 4]
    goal_side_vertices = np.array([event_xy, goal_xy_lower, goal_xy_upper])
    goal_side = Polygon(goal_side_vertices).buffer(1)  # .intersection(Point(event_xy).buffer(10))

    oppo_team = "away" if event["object_id"].startswith("home") else "home"
    oppo_x_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_x") and c[:-2] not in keepers]
    oppo_y_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_y") and c[:-2] not in keepers]
    player_xy = np.stack([snapshot[oppo_x_cols].values, snapshot[oppo_y_cols].values]).T
    player_xy = pd.DataFrame(player_xy, index=[c[:-2] for c in oppo_x_cols], columns=["x", "y"])

    can_block = player_xy.apply(lambda p: goal_side.contains(Point(p["x"], p["y"])), axis=1)
    potential_blockers = player_xy.loc[can_block[can_block].index]

    if potential_blockers.empty:
        return np.nan
    else:
        potential_blockers["dist_x"] = potential_blockers["x"] - event_xy[0]
        potential_blockers["dist_y"] = potential_blockers["y"] - event_xy[1]
        blocker_dists = potential_blockers[["dist_x", "dist_y"]].apply(np.linalg.norm, axis=1)
        return blocker_dists.idxmin()


def drop_nodes(graph: Data, labels: torch.Tensor, node_mask: torch.BoolTensor) -> Tuple[Data, torch.Tensor]:
    node_mask_indices = torch.where(node_mask)[0]
    index_map = -torch.ones((graph.num_nodes,)).long()
    index_map[node_mask_indices] = torch.arange(len(node_mask_indices))
    index_map = torch.cat([index_map, torch.tensor([-1], dtype=torch.long)])  # To map -1 to -1

    node_attr = graph.x[node_mask]
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    edge_index = index_map[graph.edge_index[:, edge_mask]]
    edge_attr = graph.edge_attr[edge_mask]
    masked_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    masked_labels = labels.clone()
    masked_labels[4] = node_mask.long().sum()  # number of players
    masked_labels[5] = index_map[masked_labels[5].long()]  # intent index
    masked_labels[6] = index_map[masked_labels[6].long()]  # receiver index

    return masked_graph, masked_labels


def drop_opponent_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 0] == 1
    return drop_nodes(graph, labels, node_mask)


def drop_goal_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 2] == 0
    return drop_nodes(graph, labels, node_mask)


def drop_non_blocker_nodes(
    graph: Data, labels: torch.Tensor, poss_flag_index=13, buffer_x=5
) -> Tuple[Data, torch.Tensor]:
    poss_or_oppo = (graph.x[:, poss_flag_index] == 1) | (graph.x[:, 0] == 0)
    poss_x = graph.x[graph.x[:, poss_flag_index] == 1, 3].item()
    node_mask = poss_or_oppo & (graph.x[:, 3] > poss_x - buffer_x)
    return drop_nodes(graph, labels, node_mask)


def sparsify_edges(graph: Data, how="distance", possessor_index: int = None, max_dist=10) -> Data:
    if how == "distance":
        edge_index = graph.edge_index
        if possessor_index is not None:
            passer_edges = (edge_index[0] == possessor_index) | (edge_index[1] == possessor_index)
        close_edges = graph.edge_attr[:, 0] <= max_dist

        graph.edge_index = edge_index[:, passer_edges | close_edges]
        graph.edge_attr = graph.edge_attr[passer_edges | close_edges]

    elif how == "delaunay":
        # xy = graph.x[:, 1:3] if graph.x.shape[1] < 18 else graph.x[:, 3:5]
        xy = graph.x[:, 3:5]
        tri_pts = Delaunay(xy.cpu().detach().numpy()).simplices
        tri_edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        tri_edges = np.unique(tri_edges, axis=0).tolist()

        for [i, j] in tri_edges:
            if [j, i] not in tri_edges:
                tri_edges.append([j, i])

        complete_edges = graph.edge_index.cpu().detach().numpy().T
        complete_edge_dict = {tuple(e): i for i, e in enumerate(complete_edges)}
        tri_edge_index = np.sort([complete_edge_dict[tuple(e)] for e in tri_edges]).tolist()

        graph.edge_index = graph.edge_index[:, tri_edge_index]
        graph.edge_attr = graph.edge_attr[tri_edge_index]

    return graph


def filter_features_and_labels(
    features: List[Data],
    labels: torch.Tensor,
    args: Dict[str, Any],
    event_indices: np.ndarray = None,
) -> Tuple[List[Data], torch.Tensor]:
    filtered_features = []
    filtered_labels = []

    for i in range(len(labels)):
        if event_indices is not None and labels[i, 0].item() not in event_indices:
            continue

        graph: Data = features[i]
        graph_labels: torch.Tensor = labels[i]

        if graph is None:
            # filtered_features.append(graph)
            continue
        else:
            graph = graph.clone()

        try:
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
        except RuntimeError:
            continue

        if args["xy_only"]:
            graph.x[7:12] = 0
            graph.x[13:19] = 0

        if not args["possessor_aware"]:
            assert not args["extend_features"]
            graph.x[:, 13:] = 0

        if not args["poss_vel_aware"]:
            if args["possessor_aware"]:
                graph.x[graph.x[:, 13] == 1, 5:9] = 0
            graph.x[:, 17:19] = 0

        if not args["keeper_aware"]:
            graph.x[:, 1] = 0

        if not args["ball_z_aware"]:
            graph.x[:, 12] = 0

        if not args["extend_features"]:
            graph.x[:, 19:] = 0

        if not config.TASK_CONFIG.at[args["task"], "include_goals"]:
            graph, graph_labels = drop_goal_nodes(graph, graph_labels)

        if args["task"].endswith("oppo_agn"):
            graph, graph_labels = drop_opponent_nodes(graph, graph_labels)

        if "filter_blockers" in args and args["filter_blockers"]:
            assert args["possessor_aware"]
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
            graph, graph_labels = drop_non_blocker_nodes(graph, graph_labels)

        if args["sparsify"] == "distance":
            assert args["possessor_aware"]
            possessor_index = torch.nonzero(graph.x[:, 13] == 1).item()
            graph = sparsify_edges(graph, "distance", possessor_index, args["max_edge_dist"])
        elif args["sparsify"] == "delaunay" and graph.x.shape[0] > 3:
            graph = sparsify_edges(graph, "delaunay")

        filtered_features.append(graph)
        filtered_labels.append(graph_labels)

    return filtered_features, torch.stack(filtered_labels, axis=0)


def find_active_players(tracking: pd.DataFrame, frame: int = None, team: str = None, include_goals=False) -> dict:
    if pd.isna(frame):
        snapshot = tracking.dropna(how="all", axis=1).copy()
    else:
        snapshot = tracking.loc[frame:frame].dropna(how="all", axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    if not pd.isna(frame):
        team = team or tracking.at[frame, "ball_owning_home_away"]
    else:
        team = team or "home"

    if team == "home":
        players = [home_players, away_players]
    else:
        players = [away_players, home_players]

    return players


def player_sort_key(s: str):
    if s == "home_goal":
        return (2, 0)
    elif s == "away_goal":
        return (2, 1)
    else:
        team, num = s.split("_", 1)
        return (0 if team == "home" else 1, int(num))


def abbr_position(position: str) -> str:
    if position == "striker":
        return "CF"
    else:
        tokens = ["back" if t == "defender" else t for t in position.split("_")]
        return "".join(t[0].upper() for t in tokens)


def insert_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    if "index" not in actions:
        actions = actions.copy().reset_index()

    dribble_actions = []

    for i in range(len(actions) - 1):
        cur_action = actions.iloc[i]
        next_action = actions.iloc[i + 1]

        if (
            (cur_action["period_id"] == next_action["period_id"])
            and not cur_action["offside"]
            and (next_action["spadl_type"] not in config.SET_PIECE_OOP)
            and (next_action["frame_id"] - cur_action["receive_frame_id"] >= 5)
            and (next_action["object_id"] == cur_action["receiver_id"])
        ):
            cur_action_dur = 0.04 * (cur_action["receive_frame_id"] - cur_action["frame_id"])
            action = next_action.copy().to_dict()
            action["index"] = -1
            action["frame_id"] = max(cur_action["receive_frame_id"], cur_action["frame_id"] + 1)
            action["seconds"] = round(cur_action["seconds"] + cur_action_dur, 2)
            action["spadl_type"] = "dribble"
            action["success"] = True
            action["offside"] = False
            action["expected_goal"] = np.nan
            action["next_player_id"] = next_action["object_id"]
            action["next_type"] = next_action["spadl_type"]
            action["receiver_id"] = next_action["object_id"]
            action["receive_frame_id"] = next_action["frame_id"]
            action["start_x"] = cur_action["end_x"]
            action["start_y"] = cur_action["end_y"]
            action["start_z"] = 0
            action["end_x"] = next_action["start_x"]
            action["end_y"] = next_action["start_y"]
            action["action_type"] = "dribble"
            action["blocked"] = False
            action["intent_id"] = next_action["object_id"]
            dribble_actions.append(action)

    return pd.concat([actions, pd.DataFrame(dribble_actions)]).sort_values("frame_id", ignore_index=True)
