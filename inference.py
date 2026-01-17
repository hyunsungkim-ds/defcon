import re
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from xgboost import XGBClassifier

from datatools.config import FIELD_SIZE, TASK_CONFIG
from datatools.match import Match
from datatools.utils import (
    filter_features_and_labels,
    find_active_players,
    player_sort_key,
)
from models.gnn import GNN


def inference_boost(
    match: Match,
    model: Union[XGBClassifier, CatBoostClassifier],
    post_action: bool = False,
    pad_own_half: bool = True,  # Zero-pad xG values for events occurring in the team's own half
    event_indices: pd.Index = None,
) -> pd.Series:
    features = match.tabular_features_0 if not post_action else match.tabular_features_1

    event_indices = event_indices if event_indices is not None else match.actions.index
    mask = match.actions.index.isin(event_indices)
    features = features.numpy()[mask, :20]

    probs = model.predict_proba(features)[:, 1]
    if pad_own_half:
        own_half_mask = features[:, 2] < FIELD_SIZE[0] / 2
        probs[own_half_mask] = 0.0

    return pd.Series(probs, index=event_indices, dtype=float)


def inference_gnn(
    match: Match,
    model: GNN,
    device: str = "cuda",
    post_action: bool = False,
    event_indices: pd.Index = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gnn_task = TASK_CONFIG.at[model.args["task"], "gnn_task"]
    include_goals = TASK_CONFIG.at[model.args["task"], "include_goals"]
    out_filter = TASK_CONFIG.at[model.args["task"], "out_filter"]

    if not post_action:
        graphs, labels = filter_features_and_labels(match.graph_features_0, match.labels, model.args, event_indices)
    else:
        graphs, labels = filter_features_and_labels(match.graph_features_1, match.labels, model.args, event_indices)

    graphs = Batch.from_data_list(graphs).to(device)
    graphs.x = graphs.x[:, : model.args["node_in_dim"]]

    two_case_tasks = ["outcome_scoring", "outcome_conceding", "outcome_return", "intent_return"]
    if model.args["task"] in two_case_tasks:
        probs_0 = []
        probs_1 = []
    else:
        probs = []

    with torch.no_grad():
        if model.args["task"] == "shot_blocking":
            out = torch.sigmoid(model(graphs)).cpu().detach().numpy()  # [B,]
            event_indices = labels[:, 0].cpu().detach().numpy().astype(int)
            return pd.Series(out, index=event_indices), None

        else:  # model.args["task"].startswith("node")
            batch = graphs.batch
            out = model(graphs)

            if TASK_CONFIG.at[model.args["task"], "out_filter"] == "teammates":
                # Select components corresponding to teammates
                batch = batch[graphs.x[:, 0] == 1]
                out = out[graphs.x[:, 0] == 1]  # [N',]

            if "receiver" in model.args["task"] and model.args["include_out"]:
                batch = torch.cat([batch, torch.unique(graphs.batch)])

    players = set()

    for i in tqdm(range(graphs.num_graphs), desc=model.args["task"]):
        event_index = int(labels[i, 0].item())

        if post_action:
            frame = int(match.actions.at[event_index, "end_frame_id"])
            team = match.actions.at[event_index, "end_player_id"][:4]
        else:
            frame = int(match.actions.at[event_index, "frame_id"])
            team = match.actions.at[event_index, "object_id"][:4]

        active_players = find_active_players(match.tracking, frame, team, include_goals=include_goals)

        if gnn_task == "node_selection":
            probs_i = torch.softmax(out[batch == i], dim=0).cpu().detach().numpy()
        elif gnn_task in ["node_binary", "graph_binary"]:
            probs_i = torch.sigmoid(out[batch == i]).cpu().detach().numpy()
        elif gnn_task == "node_regression":
            probs_i = torch.sigmoid(out[batch == i]).cpu().detach().numpy() * 2 - 1

        if out_filter == "teammates":
            player_indices_i = active_players[0]
        elif out_filter == "all":  # "receiver" in model.args["task"]
            player_indices_i = active_players[0] + active_players[1]
            if model.args["include_out"]:
                player_indices_i.append("out")

        players = players | set(player_indices_i)

        if model.args["task"] in two_case_tasks:
            probs_i0 = dict(zip(player_indices_i, probs_i[:, 0].tolist()))
            probs_i1 = dict(zip(player_indices_i, probs_i[:, 1].tolist()))
            probs_0.append(dict(**probs_i0, **{"index": event_index}))
            probs_1.append(dict(**probs_i1, **{"index": event_index}))
        else:
            probs_i = dict(zip(player_indices_i, probs_i.tolist()))
            probs.append(dict(**probs_i, **{"index": event_index}))

    players = sorted(list(players), key=player_sort_key)

    if model.args["task"] in two_case_tasks:
        probs_0 = pd.DataFrame(probs_0).set_index("index")[players]
        probs_1 = pd.DataFrame(probs_1).set_index("index")[players]
        return probs_0, probs_1
    else:
        return pd.DataFrame(probs).set_index("index")[players], None


def inference_gnn_posterior(
    match: Match,
    model: GNN = None,
    device="cuda",
    event_indices: pd.Index = None,
    melt: bool = True,
) -> pd.DataFrame:
    graphs, labels = filter_features_and_labels(match.graph_features_0, match.labels, model.args, event_indices)
    include_goals = (graphs[0].x[:, 2] == 1).any().item()
    posteriors = []

    for data_index in tqdm(range(len(graphs)), desc="failure_posterior"):
        graph_i = graphs[data_index].to(device)

        event_index = int(labels[data_index, 0].item())
        frame = int(match.actions.at[event_index, "frame_id"])
        team = match.actions.at[event_index, "object_id"][:4]
        active_players = find_active_players(match.tracking, frame, team, include_goals=include_goals)
        n_teammates = len(active_players[0])

        intended_graphs = []
        for intent_index in range(n_teammates):
            intent_onehot = torch.zeros(graph_i.x.shape[0]).to(device)
            intent_onehot[intent_index] = 1
            intended_nodes = torch.cat([graph_i.x, intent_onehot.unsqueeze(1)], -1)
            intended_graph = Data(x=intended_nodes, edge_index=graph_i.edge_index, edge_attr=graph_i.edge_attr)
            intended_graphs.append(intended_graph)
        intended_graphs = Batch.from_data_list(intended_graphs).to(device)

        with torch.no_grad():
            logits = model(intended_graphs)  # [12 * 24 + 12,] if include_out else [12 * 24,]

        if model.args["include_out"]:
            receive_logits = logits[:-n_teammates].reshape(n_teammates, -1)  # [12, 24]
            ballout_logits = logits[-n_teammates:].unsqueeze(1)  # [12, 1]
            logits = torch.cat([receive_logits, ballout_logits], 1)  # [12, 25]
            posteriors_i = torch.softmax(logits[:, n_teammates:], dim=1).cpu().detach().numpy()  # [12, 13]
            posteriors_i = pd.DataFrame(posteriors_i, index=active_players[0], columns=active_players[1] + ["out"])

        else:
            logits = logits.reshape(n_teammates, -1)  # [12, 24]
            posteriors_i = torch.softmax(logits[:, n_teammates:], dim=1).cpu().detach().numpy()  # [12, 12]
            posteriors_i = pd.DataFrame(posteriors_i, index=active_players[0], columns=active_players[1])

        posteriors_i["index"] = event_index
        posteriors_i.index.name = "option"
        posteriors.append(posteriors_i)

    valid_tracking = match.tracking.dropna(axis=1, how="all")
    home_players = [c[:-2] for c in valid_tracking.columns if re.match(r"home_\d+_x", c)]
    away_players = [c[:-2] for c in valid_tracking.columns if re.match(r"away_\d+_x", c)]

    if model.args["include_out"]:
        posteriors = pd.concat(posteriors)[["index"] + home_players + away_players + ["out"]].reset_index()
    else:
        posteriors = pd.concat(posteriors)[["index"] + home_players + away_players].reset_index()

    if melt:
        posteriors = posteriors.melt(id_vars=["index", "option"], var_name="defender", value_name="posterior")
        return posteriors.dropna(subset=["posterior"]).reset_index(drop=True).copy()
    else:
        return posteriors


def inference_gnn_grid(match: Match, model: GNN, device="cuda") -> Dict[int, torch.Tensor]:
    assert "dest" in model.args["task"]

    grid_size = (int(FIELD_SIZE[0]), int(FIELD_SIZE[1]))
    grid = np.mgrid[0 : grid_size[0], grid_size[1] - 1 : -1 : -1] + 0.5  # [2, 105, 68]
    grid = np.transpose(grid, (0, 2, 1)).reshape(2, -1)  # [2, 68 * 105]
    dest_tensor = torch.tensor(grid.T, dtype=torch.float32).to(device)  # [68 * 105, 2]
    n_cells = dest_tensor.shape[0]  # G = 68 * 105

    graphs, labels = filter_features_and_labels(match.graph_features_0, match.labels, model.args)
    receive_probs = dict()
    success_probs = dict()

    for data_index in tqdm(range(len(graphs)), desc="dest_receiver"):
        graph_i = graphs[data_index].to(device)

        with torch.no_grad():
            graph_i = Batch.from_data_list([graph_i]).to(device)
            node_emb, graph_emb = model.encoder(graph_i)  # [P, z], [1, z]

            node_feat_rep = graph_i.x.repeat(n_cells, 1)  # [G * P, x]
            node_emb_rep = node_emb.repeat(n_cells, 1)  # [G * P, z]
            graph_emb_rep = graph_emb.repeat(n_cells, 1)  # [G, z]
            batch_indices = torch.arange(n_cells, device=device).repeat_interleave(graph_i.num_nodes)

            logits_i = model.decoder(node_feat_rep, node_emb_rep, graph_emb_rep, batch_indices, dest_tensor)

        if model.args["include_out"]:
            node_logits = logits_i[:-n_cells].view(n_cells, -1)  # [G, P]
            out_logits = logits_i[-n_cells:].view(n_cells, 1)  # [G, 1]
            logits_i = torch.cat([node_logits, out_logits], dim=-1)  # [G, P + 1]
        else:
            logits_i = logits_i.view(n_cells, -1)  # [G, P]

        event_index = int(labels[data_index, 0].item())

        receive_probs_i = F.softmax(logits_i, dim=-1)  # [G, P(+1)]
        n_teammates = torch.sum(graph_i.x[:, 0] == 1).item()
        receive_probs[event_index] = receive_probs_i.reshape(grid_size[1], grid_size[0], -1)

        success_probs_i = receive_probs_i[:, :n_teammates].sum(axis=1)
        success_probs[event_index] = success_probs_i.reshape(grid_size[1], grid_size[0])

    return receive_probs, success_probs
