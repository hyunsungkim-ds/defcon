import argparse
import os
import sys
from datetime import datetime

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch

from datatools.defcon import DEFCON
from datatools.event_xg import EventXGModel
from datatools.match import Match
from datatools.tabular_feature import construct_tabular_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot_success", type=str, default="unblocked", choices=["goal", "on_target", "unblocked"])
    parser.add_argument("--load_saved", action="store_true", default=False, help="load saved components")
    parser.add_argument("--component_path", type=str, required=False, default="data/ajax/defcon_components")
    parser.add_argument("--result_path", type=str, required=False, default="data/ajax/player_scores.parquet")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    defcon_args = {
        "shot_success": args.shot_success,
        "select_model_id": "action_intent/00",
        "pass_success_model_id": "pass_success/20",
        "score_model_id": "outcome_scoring/20",
        "concede_model_id": "outcome_conceding/20",
        "posterior_model_id": "failure_receiver/21",
        "device": device,
    }

    if args.shot_success == "unblocked":
        event_xg_model = EventXGModel(unblocked=True)
        event_xg_model.train(verbose=False)
        defcon_args["event_xg_model"] = event_xg_model
        defcon_args["shot_block_model_id"] = "shot_blocking/01"
    else:  # args.shot_success in ["goal", "on_target"]
        defcon_args["shot_xg_model_id"] = "shot_success/01"

    event_files = np.sort(os.listdir("data/ajax/event_synced"))
    match_ids = np.sort([f.split(".")[0] for f in event_files if f.endswith(".csv")])

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet").sort_values("game_date", ignore_index=True)
    lineups["game_id"] = lineups["stats_perform_match_id"]
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    match_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]

    test_match_ids = match_dates[(match_dates.index.isin(match_ids)) & (match_dates >= datetime(2024, 8, 1))].index
    n_test_matches = len(test_match_ids)

    for i, match_id in enumerate(test_match_ids):
        events = pd.read_csv(f"data/ajax/event_synced/{match_id}.csv", header=0, parse_dates=["utc_timestamp"])
        tracking = pd.read_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
        match_lineup = lineups.loc[lineups["stats_perform_match_id"] == match_id]

        match_date = match_lineup["game_date"].iloc[0].date()
        match_name = " vs ".join(match_lineup["contestant_name"].unique())
        print(f"\n[{i+1}/{n_test_matches}] {match_id}: {match_name} on {match_date}")

        match = Match(events, tracking, match_lineup, action_type="all", include_goals=True)
        print("Constructing labels...")
        match.labels = match.construct_labels(discount_xg=True)
        match.graph_features_0 = torch.load(f"data/ajax/features/action_graphs/{match_id}.pt", weights_only=False)
        match.graph_features_1 = torch.load(f"data/ajax/features/post_action_graphs/{match_id}.pt", weights_only=False)
        match.actions = match.label_post_actions(match.actions)

        if args.shot_success in ["goal", "on_target"]:
            print("Constructing tabular features for actions...")
            match.tabular_features_0 = construct_tabular_features(match, post_action=False)
            print("Constructing tabular features for post-actions...")
            match.tabular_features_1 = construct_tabular_features(match, post_action=True)

        print("\nValuing defensive contributions...")
        defcon = DEFCON(match, **defcon_args)

        if args.load_saved:
            defcon.load_components(args.component_path)
            # defcon.estimate_shot_components()
        else:
            defcon.estimate_components()
            defcon.save_components(args.component_path)

        player_scores = defcon.evaluate_players()

        if not os.path.exists(args.result_path):
            player_scores.to_parquet(args.result_path, engine="pyarrow", index=False)
        else:
            saved_scores = pd.read_parquet(args.result_path, engine="pyarrow")
            saved_scores = pd.concat([saved_scores, player_scores], ignore_index=True)
            saved_scores.to_parquet(args.result_path, engine="pyarrow", index=False)
