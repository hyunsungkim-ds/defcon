import argparse
import json
import os
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from datatools.config import FIELD_SIZE
from models.gnn import GNN


def num_trainable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_arg_keys: List[str], args_dict: dict, parser: argparse.ArgumentParser):
    if parser is None:
        return args_dict

    for key in model_arg_keys:
        if key.startswith("n_") or key.endswith("_dim"):
            parser.add_argument("--" + key, type=int, required=True)
        elif key == "dropout":
            parser.add_argument("--" + key, type=float, default=0)
        else:
            parser.add_argument("--" + key, action="store_true", default=False)
    model_args, _ = parser.parse_known_args()

    for key in model_arg_keys:
        args_dict[key] = getattr(model_args, key)

    return args_dict


def get_args_str(keys, args_dict: dict) -> str:
    ret = ""
    for key in keys:
        if key in args_dict:
            ret += " {} {} |".format(key, args_dict[key])
    return ret[1:-2]


def get_losses_str(losses: dict) -> str:
    ret = ""
    for key, value in losses.items():
        if key == "count":
            continue
        ret += " {}: {:.4f} |".format(key, np.mean(value))
    # if len(losses) > 1:
    #     ret += " total_loss: {:.4f} |".format(sum(losses.values()))
    return ret[:-2]


def printlog(line: str, trial_path: str) -> None:
    print(line)
    with open(trial_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def l1_regularizer(model, lambda_l1=0.1):
    l1_loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            l1_loss += lambda_l1 * model_param_value.abs().sum()
    return l1_loss


def encode_onehot(labels, classes=None):
    if classes:
        classes = [x for x in range(classes)]
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_splits(
    lineup_path="data/ajax/lineup/line_up.parquet",
    feature_dir: str = "data/ajax/features/action_graphs",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from datetime import datetime

    np.random.seed(100)
    CUTOFF_DATE = datetime(2024, 8, 1)

    lineups = pd.read_parquet(lineup_path).sort_values("game_date", ignore_index=True)
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    lineups["game_id"] = lineups["stats_perform_match_id"]
    match_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]

    match_ids = [f.split(".")[0] for f in np.sort(os.listdir(feature_dir)) if f.endswith(".pt")]
    test_match_ids = match_dates[(match_dates.index.isin(match_ids)) & (match_dates >= CUTOFF_DATE)].index

    match_ids = match_dates[(match_dates.index.isin(match_ids)) & (match_dates < CUTOFF_DATE)].index
    train_match_ids = np.sort(np.random.choice(match_ids, 200, replace=False))
    valid_match_ids = np.sort([id for id in match_ids if id not in train_match_ids])

    return train_match_ids, valid_match_ids, test_match_ids


def load_model(model_id="pass_intent/01", device="cuda") -> GNN:
    if model_id is None:
        return None

    else:
        model_path = f"saved/{model_id}"
        with open(f"{model_path}/args.json", "r") as f:
            args = json.load(f)

        if args["model"] in ["gcn", "gin", "gat"]:  # GNN models
            model = GNN(args).to(device)
            weights_path = f"{model_path}/best_weights.pt"
            state_dict = torch.load(weights_path, weights_only=False, map_location=lambda storage, _: storage)

            # Backward compatibility for older checkpoints saved with encoder.gat_layers.*
            if any(key.startswith("encoder.gat_layers") for key in state_dict.keys()):
                remapped_state = OrderedDict()
                for key, value in state_dict.items():
                    if key.startswith("encoder.gat_layers"):
                        new_key = key.replace("encoder.gat_layers", "encoder.gnn_layers", 1)
                    else:
                        new_key = key
                    remapped_state[new_key] = value
                state_dict = remapped_state
            model.load_state_dict(state_dict)

        elif args["model"] in ["xgboost", "catboost"]:  # Gradient boosting models
            with open(f"{model_path}/best_params.json", "r") as f:
                params = json.load(f)
            model = XGBClassifier(**params) if args["model"] == "xgboost" else CatBoostClassifier(**params)
            model.load_model(f"{model_path}/best_model.json")

        return model


def estimate_propensity(dataset, model_id="pass_intent/00", device="cuda", min_clip=0.01) -> torch.Tensor:
    model = load_model(model_id, device)
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, pin_memory=True)
    likelihoods = []

    for batch_graphs, batch_labels, _ in tqdm(loader):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            batch_graphs.x = batch_graphs.x[:, : model.args["node_in_dim"]]
            out: torch.Tensor = model(batch_graphs)
            for graph_index in range(batch_graphs.num_graphs):
                logits = out[(batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)]
                probs = nn.Softmax(dim=0)(logits).cpu().detach().numpy()
                likelihoods.append(probs[int(batch_labels[graph_index, 5].item())])

    return torch.Tensor(likelihoods).clip(min_clip)


def calc_pos_error(pred_xy, target_xy, aggfunc="mean"):
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_accuracy(y, y_hat, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(y_hat, dim=1) == y).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(y_hat, dim=1) == y).float().sum().item()


def calc_binary_metrics(y, y_hat, threshold=0.5):
    y_pred = y_hat > threshold
    precision = precision_score(y, y_pred) if np.sum(y_pred) > 0 else 0
    recall = recall_score(y, y_pred) if np.sum(y) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y, y_pred) if precision > 0 and recall > 0 else 0,
        "roc_auc": roc_auc_score(y, y_hat) if np.sum(y) > 0 else 0.5,
        "brier": brier_score_loss(y, y_hat),
        "log_loss": log_loss(y, y_hat) if np.sum(y) > 0 else np.nan,
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def adjust_dests(labels: torch.Tensor) -> torch.Tensor:
    start_xy = labels[:, 8:10]
    end_xy = labels[:, 10:12]
    intent_xy = labels[:, 12:14]

    # Masks for failed passes with valid coordinates
    is_failed_pass = (labels[:, 1] == 1) & (labels[:, -5] == 0)
    has_valid_xy = (
        (start_xy[:, 0] >= 0)
        & (start_xy[:, 1] >= 0)
        & (end_xy[:, 0] >= 0)
        & (end_xy[:, 1] >= 0)
        & (intent_xy[:, 0] >= 0)
        & (intent_xy[:, 1] >= 0)
    )
    adjust_mask = is_failed_pass & has_valid_xy

    intended_len = torch.linalg.norm(start_xy - intent_xy, dim=1)
    actual_len = torch.linalg.norm(start_xy - end_xy, dim=1).clamp_min(1e-6)
    scale = (intended_len / actual_len).unsqueeze(1)
    end_xy_adj = start_xy + scale * (end_xy - start_xy)

    dests = torch.where(adjust_mask.unsqueeze(1), end_xy_adj, end_xy).clone()
    dests[:, 0] = dests[:, 0].clamp(0.0, FIELD_SIZE[0])
    dests[:, 1] = dests[:, 1].clamp(0.0, FIELD_SIZE[1])

    return dests


def build_dest_features(graphs: Batch, dest_xy: torch.Tensor, oppo_aware=True) -> torch.Tensor:
    B = graphs.num_graphs  # batch_size
    G = dest_xy.size(0)  # grid_size

    feat: torch.Tensor = graphs.x
    batch: torch.Tensor = graphs.batch

    poss_mask = feat[:, 13] == 1
    team_mask = (feat[:, 0] == 1) & (feat[:, 2] == 0)
    oppo_mask = (feat[:, 0] == 0) & (feat[:, 2] == 0)

    dest_feat = []

    for graph_idx in range(B):
        graph_mask = batch == graph_idx
        poss_xy = feat[graph_mask & poss_mask, 3:5]  # [1, 2]
        team_xy = feat[graph_mask & team_mask, 3:5]  # [team, 2]
        oppo_xy = feat[graph_mask & oppo_mask, 3:5]  # [oppo, 2]

        poss_dx = dest_xy[:, 0] - poss_xy[0, 0]
        poss_dy = dest_xy[:, 1] - poss_xy[0, 1]

        # Distance to the nearest teammate
        team_dxy = dest_xy.unsqueeze(1) - team_xy.unsqueeze(0)  # [G, team, 2]
        team_nn_dist = (team_dxy**2).sum(-1).min(-1).values.sqrt()  # [G, team, 2] -> [G, team] -> [G]

        if oppo_aware:
            # Distance to the nearest opponent
            oppo_dxy = dest_xy.unsqueeze(1) - oppo_xy.unsqueeze(0)  # [G, oppo, 2]
            oppo_nn_dist = (oppo_dxy**2).sum(-1).min(-1).values.sqrt()  # [G, oppo, 2] -> [G, oppo] -> [G]

            # Distance from the possessor-cell line segment to its nearest opponent
            pass_xy = dest_xy - poss_xy  # [G, 2]
            pass_len = (pass_xy**2).sum(-1).clamp_min(1e-6)  # [G]
            dot = pass_xy @ (oppo_xy - poss_xy).t()  # [G, oppo]
            proj_ratio = (dot / pass_len.unsqueeze(1)).clamp(0.0, 1.0)  # [G, oppo]
            proj_point = poss_xy.unsqueeze(1) + proj_ratio.unsqueeze(-1) * pass_xy.unsqueeze(1)  # [G, oppo, 2]
            pass_oppo_dist = ((proj_point - oppo_xy.unsqueeze(0)) ** 2).sum(-1).sqrt()  # [G, oppo]
            pass_oppo_nn_dist, _ = pass_oppo_dist.min(dim=1)  # [G]

        else:
            oppo_nn_dist = torch.zeros(G, device=dest_xy.device)
            pass_oppo_nn_dist = torch.zeros(G, device=dest_xy.device)

        dest_feat_i = [dest_xy[:, 0], dest_xy[:, 1], poss_dx, poss_dy, team_nn_dist, oppo_nn_dist, pass_oppo_nn_dist]
        dest_feat.append(torch.stack(dest_feat_i, dim=1))  # [G, d]

    dest_feat = torch.cat(dest_feat, dim=0)  # [B*G, d]
    return dest_feat.reshape(B, G, -1).to(dest_xy.device)  # [B, G, d]


def get_grid_xy(grid_size: Tuple[int, int] = FIELD_SIZE, device="cuda") -> torch.Tensor:
    cell_x = FIELD_SIZE[0] / grid_size[0]
    cell_y = FIELD_SIZE[1] / grid_size[1]

    grid_size = (int(grid_size[0]), int(grid_size[1]))
    x_edges = torch.linspace(0, FIELD_SIZE[0], grid_size[0] + 1, device=device, dtype=torch.float32)
    y_edges = torch.linspace(FIELD_SIZE[1], 0, grid_size[1] + 1, device=device, dtype=torch.float32)
    x_centers = x_edges[:-1] + cell_x / 2
    y_centers = y_edges[1:] + cell_y / 2

    grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()]).T

    return grid_xy


def cartesian_to_polar(xy: torch.Tensor) -> torch.Tensor:
    goal_dx = FIELD_SIZE[0] - xy[:, 0]
    goal_dy = xy[:, 1] - FIELD_SIZE[1] / 2
    goal_dist = torch.sqrt(goal_dx**2 + goal_dy**2)
    goal_angle = torch.atan2(goal_dy, goal_dx)  # (-pi/2, pi/2) if x < 105
    return torch.stack([goal_dist, goal_angle], dim=1)


def find_cell_index(xy: torch.Tensor, grid_size: Tuple[int, int] = FIELD_SIZE) -> torch.Tensor:
    cell_x = FIELD_SIZE[0] / grid_size[0]
    cell_y = FIELD_SIZE[1] / grid_size[1]

    x_index = torch.floor(xy[:, 0] / cell_x).long().clamp(0, grid_size[0] - 1)
    y_index = (grid_size[1] - 1 - torch.floor(xy[:, 1] / cell_y)).long().clamp(0, grid_size[1] - 1)

    return y_index * grid_size[0] + x_index


def run_epoch(
    args: argparse.Namespace,
    model: nn.DataParallel,
    loader: DataLoader,
    optimizer: torch.optim.Adam = None,
    device: str = "cuda",
    pos_weight: float = 1.0,
    train: bool = False,
):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)
    pos_weight = torch.tensor(pos_weight)

    if args.gnn_task in ["node_binary", "graph_binary"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "f1": 0, "roc_auc": 0, "brier": 0}
    elif args.gnn_task in ["node_selection", "graph_multiclass"]:
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "accuracy": 0, "mrr": 0}
    elif args.gnn_task in ["node_regression", "graph_regression"]:
        metrics = {"count": 0, "mse_loss": 0, "l1_loss": 0}

    for batch_index, (batch_graphs, batch_labels, batch_ipw) in enumerate(loader):
        batch_graphs: Batch = batch_graphs.to(device)
        batch_ipw: torch.Tensor = batch_ipw.to(device)
        index_range = torch.unique(batch_graphs.batch)

        metrics["count"] += batch_graphs.num_graphs

        if args.include_out:
            # One node per player and one ball-out node per graph instance
            batch = torch.cat([batch_graphs.batch, index_range])
        else:
            batch = batch_graphs.batch

        batch_labels: torch.Tensor = batch_labels.to(device)
        batch_labels[batch_labels[:, 6] == -1, 6] = batch_labels[batch_labels[:, 6] == -1, 4]  # -1 to n_players

        if "dest" in args.task:
            if getattr(args, "adjust_dest", False):
                batch_dests = adjust_dests(batch_labels)
            else:
                batch_dests = batch_labels[:, 10:12].clone()

            if getattr(args, "normalize_dest", False):
                assert not args.task.endswith("dest")
                batch_dests[:, 0] /= float(FIELD_SIZE[0])
                batch_dests[:, 1] /= float(FIELD_SIZE[1])

            elif getattr(args, "polar_dest", False):
                polar_dests = cartesian_to_polar(batch_dests)
                batch_dests = torch.cat([batch_dests, polar_dests], axis=1)

        else:
            batch_dests = None

        if "pass_dest" in args.task:
            grid_xy = get_grid_xy(device=device)  # [G, 2]

            if getattr(args, "more_dest_features", False):
                oppo_aware = "oppo_agn" not in args.task
                grid_features = build_dest_features(batch_graphs, grid_xy, oppo_aware)  # [B, G, d]
            else:
                grid_features = grid_xy.clone()  # [G, 2]

            if train:
                out: torch.Tensor = model.module.forward_grid(batch_graphs, grid_features)
            else:
                with torch.no_grad():
                    out: torch.Tensor = model.module.forward_grid(batch_graphs, grid_features)

        else:
            if train:
                out: torch.Tensor = model(batch_graphs, batch_dests)
            else:
                with torch.no_grad():
                    out: torch.Tensor = model(batch_graphs, batch_dests)

        if args.gnn_task == "node_selection":  # {pass/action}_intent, {success/failure}_receiver
            if args.task.split("_")[1] == "intent":
                target = batch_labels[:, 5].clone().long()
            elif args.task.split("_")[1] == "receiver":
                target = batch_labels[:, 6].clone().long()

            loss_fn = nn.CrossEntropyLoss()
            pred_loss = 0
            accuracy = 0

            for graph_index in index_range:
                if args.task in ["pass_intent", "pass_intent_oppo_agn", "action_intent", "success_receiver"]:
                    # Only take teammate nodes
                    assert not args.include_out
                    pred_i = out[(batch == graph_index) & (batch_graphs.x[:, 0] == 1)]  # [N_i]
                    target_i = target[graph_index]

                elif args.task == "failure_receiver":
                    # Only take opponent nodes
                    if args.include_out:
                        ball_out_mask = torch.ones(batch_graphs.num_graphs).bool().to(device)
                        failure_mask = torch.cat([batch_graphs.x[:, 0] == 0, ball_out_mask])
                    else:
                        failure_mask = batch_graphs.x[:, 0] == 0
                    pred_i = out[(batch == graph_index) & failure_mask]
                    n_teammates = ((batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)).sum()
                    target_i = target[graph_index] - n_teammates

                else:  # pass_receiver, dest_receiver
                    pred_i = out[batch == graph_index]
                    target_i = target[graph_index]

                pred_loss += loss_fn(pred_i.unsqueeze(0), target_i.unsqueeze(0))
                accuracy += (pred_i.argmax() == target_i).float()

                rank = (pred_i.argsort(descending=True) == target_i).nonzero(as_tuple=True)[0].item() + 1
                metrics["mrr"] += 1.0 / rank

            pred_loss /= index_range.shape[0]
            metrics["accuracy"] += accuracy.item()

        elif args.gnn_task == "node_binary":  # {pass/action}_success, outcome_{scoring/conceding}, intent_return
            intent = batch_labels[:, 5].clone().long()
            scoring = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
            conceding = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

            if args.task in ["pass_success", "action_success"]:
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index]])
                pred = torch.stack(pred)

                target = batch_labels[:, -5]
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = target.cpu().detach().numpy()
                threshold = 0.5 if args.task.endswith("success") else 0.1
                batch_metrics = calc_binary_metrics(y, y_hat, threshold)

            elif args.task in ["outcome_scoring", "outcome_conceding"]:
                outcome = batch_labels[:, -5].clone().long()
                pred = []
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index], outcome[graph_index]])
                pred = torch.stack(pred)

                target = scoring if args.task.endswith("scoring") else conceding
                pred_loss = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = batch_labels[:, -4] if args.task.endswith("scoring") else batch_labels[:, -2]
                y = y.cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

            elif args.task in ["intent_return", "intent_return_oppo_agn"]:
                pred_s = []
                pred_c = []
                for graph_index in index_range:
                    pred_s.append(out[batch == graph_index][intent[graph_index], 0])
                    pred_c.append(out[batch == graph_index][intent[graph_index], 1])
                pred_s = torch.stack(pred_s)
                pred_c = torch.stack(pred_c)

                pred_loss_s = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred_s, scoring)
                pred_loss_c = nn.BCEWithLogitsLoss(weight=batch_ipw, pos_weight=pos_weight)(pred_c, conceding)
                pred_loss = pred_loss_s + pred_loss_c

                # Calculate performance metrics only for goal-scoring prediction for simplicity
                y_hat = torch.sigmoid(pred_s).cpu().detach().numpy()
                y = batch_labels[:, -4].cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

        elif args.gnn_task == "node_regression":  # outcome_return
            intent = batch_labels[:, 5].clone().long()
            outcome = batch_labels[:, -5].clone().long()
            scoring = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
            conceding = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

            pred = []
            for graph_index in index_range:
                pred.append(out[batch == graph_index][intent[graph_index], outcome[graph_index]])
            pred = torch.stack(pred) * 2 - 1  # Transform output to range from [0, 1] to [-1, 1]

            target = scoring - conceding
            pred_loss = nn.MSELoss()(pred, target)
            metrics["mse_loss"] += pred_loss.item() * batch_graphs.num_graphs

        elif args.gnn_task == "graph_binary":
            # overll_{scoring/conceding}, dest_{outcome/scoring/conceding}, shot_blocking
            if args.task in ["shot_blocking", "dest_success"]:
                target = batch_labels[:, -5] if args.task.endswith("success") else batch_labels[:, -6]
                pred_loss = nn.BCEWithLogitsLoss()(out, target)

                y_hat = torch.sigmoid(out).cpu().detach().numpy()
                y = target.cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.5)

            elif args.task.split("_")[0] in ["overall", "dest"]:  # {overall/dest}_{scoring/conceding}
                scoring = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
                conceding = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

                if args.task.startswith("dest"):
                    outcome = batch_labels[:, -5].clone().long()
                    pred = out[tuple([list(range(batch_graphs.num_graphs)), outcome])]
                else:
                    pred = out

                target = scoring if args.task.endswith("scoring") else conceding
                pred_loss = nn.BCEWithLogitsLoss()(pred, target)

                y_hat = torch.sigmoid(pred).cpu().detach().numpy()
                y = batch_labels[:, -4] if args.task.endswith("scoring") else batch_labels[:, -2]
                y = y.cpu().detach().numpy()
                batch_metrics = calc_binary_metrics(y, y_hat, 0.1)

            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

        elif args.gnn_task == "graph_multiclass":  # pass_dest
            if args.task == "pass_dest":
                sigma = getattr(args, "dest_sigma", 3.0)
                dist_to_target = ((grid_xy.unsqueeze(0) - batch_dests.unsqueeze(1)) ** 2).sum(-1).float()  # [B, G]
                soft_target = torch.exp(-dist_to_target / (2 * sigma**2))
                soft_target = soft_target / soft_target.sum(dim=1, keepdim=True)  # [B, G]

                log_probs = F.log_softmax(out, dim=-1)
                pred_loss = F.kl_div(log_probs, soft_target, reduction="batchmean")

                pred_xy = grid_xy[out.argmax(dim=1)]  # [B, G] to [B, 2]
                dist_error = torch.linalg.norm(pred_xy - batch_dests, dim=1)
                metrics["accuracy"] += (dist_error <= sigma).float().sum().item()

                target = find_cell_index(batch_dests)  # [B]
                pos = (out.argsort(dim=1, descending=True) == target.unsqueeze(1)).nonzero(as_tuple=False)  # [B, 2]
                rank = pos[:, 1].to(torch.float32) + 1.0
                metrics["mrr"] += (1.0 / rank).sum().item()

        elif args.gnn_task == "graph_regression":  # overall_return
            scoring = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
            conceding = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]
            target = scoring - conceding

            pred = torch.sigmoid(out) * 2 - 1  # Transform output to range from [0, 1] to [-1, 1]
            pred_loss = nn.MSELoss()(pred, target)
            metrics["mse_loss"] += pred_loss.item() * batch_graphs.num_graphs

        if "ce_loss" in metrics:
            metrics["ce_loss"] += pred_loss.item() * batch_graphs.num_graphs

        l1_loss = l1_regularizer(model, lambda_l1=args.lambda_l1)
        metrics["l1_loss"] += l1_loss.item() * batch_graphs.num_graphs

        if train:
            optimizer.zero_grad()
            loss = pred_loss + l1_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
            optimizer.step()

        if train and batch_index % args.print_freq == 0:
            interim_metrics = dict()
            for key, value in metrics.items():
                if key == "count":
                    continue
                interim_metrics[key] = value / metrics["count"]
            print(f"[{batch_index:>{len(str(n_batches))}d}/{n_batches}]  {get_losses_str(interim_metrics)}")

    for key, value in metrics.items():
        if key == "count":
            continue
        metrics[key] = value / metrics["count"]

    return metrics
