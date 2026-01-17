import os
import re
import sys
from fnmatch import fnmatch
from typing import List

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config, utils


def is_home_on_left(period_tracking: pd.DataFrame):
    home_x_cols = [c for c in period_tracking.columns if re.match(r"home_.*_x", c)]
    away_x_cols = [c for c in period_tracking.columns if re.match(r"away_.*_x", c)]

    home_gk = (period_tracking[home_x_cols].mean() - config.FIELD_SIZE[0] / 2).abs().idxmax()[:-2]
    away_gk = (period_tracking[away_x_cols].mean() - config.FIELD_SIZE[0] / 2).abs().idxmax()[:-2]

    home_gk_x = period_tracking[f"{home_gk}_x"].mean()
    away_gk_x = period_tracking[f"{away_gk}_x"].mean()

    return home_gk_x < away_gk_x


def label_frames_and_episodes(tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
    tracking = tracking.sort_values(["period_id", "timestamp"], ignore_index=True)

    tracking["frame_id"] = (tracking["timestamp"] * fps).round().astype(int)
    tracking["episode_id"] = 0
    n_prev_frames = 0
    n_prev_episodes = 0

    for i in tracking["period_id"].unique():
        period_tracking = tracking[tracking["period_id"] == i].copy()
        tracking.loc[period_tracking.index, "frame_id"] += n_prev_frames
        n_prev_frames += period_tracking["frame_id"].max() + 1

        alive_tracking = period_tracking[period_tracking["ball_state"] == "alive"].copy()
        frame_diffs = np.diff(alive_tracking["frame_id"].values, prepend=-5)
        period_episode_ids = (frame_diffs >= 5).astype(int).cumsum() + n_prev_episodes
        tracking.loc[alive_tracking.index, "episode_id"] = period_episode_ids
        n_prev_episodes = period_episode_ids.max()

    return tracking.set_index("frame_id")


def summarize_playing_times(tracking: pd.DataFrame) -> pd.DataFrame:
    players = [c[:-2] for c in tracking.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    play_records = dict()

    for p in players:
        player_x = tracking[f"{p}_x"].dropna()
        if not player_x.empty:
            play_records[p] = {"in_frame_id": player_x.index[0], "out_frame_id": player_x.index[-1]}

    return pd.DataFrame(play_records).T


def summarize_phases(tracking: pd.DataFrame, keepers: List[str] = None) -> pd.DataFrame:
    if keepers is None:
        keepers = []
    else:
        keepers = list(keepers)

    play_records = summarize_playing_times(tracking)
    player_in_frames = play_records["in_frame_id"].unique().tolist()
    player_out_frames = (play_records["out_frame_id"].unique() + 1).tolist()
    period_start_frames = tracking.reset_index().groupby("period_id")["frame_id"].first().values.tolist()
    phase_changes = np.sort(np.unique(player_in_frames + player_out_frames + period_start_frames))

    phases = []

    for i, start_frame in enumerate(phase_changes[:-1]):
        end_frame = phase_changes[i + 1] - 1

        alive_tracking = tracking[tracking["ball_state"] == "alive"].loc[start_frame:end_frame].copy()
        if alive_tracking.empty:
            continue

        active_players = utils.find_active_players(alive_tracking)
        home_keepers = [p for p in keepers if p in active_players[0]]
        away_keepers = [p for p in keepers if p in active_players[1]]
        home_x_cols = [f"{p}_x" for p in active_players[0]]
        away_x_cols = [f"{p}_x" for p in active_players[1]]
        home_keeper = home_keepers[0] if home_keepers else alive_tracking[home_x_cols].mean().idxmin()[:-2]
        away_keeper = away_keepers[0] if away_keepers else alive_tracking[away_x_cols].mean().idxmax()[:-2]

        phase_dict = {
            "phase_id": i + 1,
            "period_id": alive_tracking["period_id"].iloc[0],
            "start_frame": start_frame,
            "end_frame": end_frame,
            "active_players": active_players[0] + active_players[1],
            "active_keepers": [home_keeper, away_keeper],
        }
        phases.append(phase_dict)

    phases = pd.DataFrame(phases).set_index("phase_id")
    return phases


def calc_physical_features(tracking: pd.DataFrame, fps=25) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    if "episode_id" not in tracking.columns:
        tracking = label_frames_and_episodes(tracking)

    home_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"home_.*_x", c)]
    away_players = [c[:-2] for c in tracking.dropna(axis=1, how="all").columns if re.match(r"away_.*_x", c)]
    objects = home_players + away_players + ["ball"]
    physical_features = ["x", "y", "vx", "vy", "speed", "accel"]

    tqdm_desc = "Calculating physical features"
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

    for p in tqdm(objects, desc=tqdm_desc, bar_format=bar_format):
        new_features = pd.DataFrame(np.nan, index=tracking.index, columns=[f"{p}_{f}" for f in physical_features[2:]])
        tracking = pd.concat([tracking, new_features], axis=1)

        for i in tracking["period_id"].unique():
            x: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_x"].dropna()
            y: pd.Series = tracking.loc[tracking["period_id"] == i, f"{p}_y"].dropna()
            if x.empty:
                continue

            vx = savgol_filter(np.diff(x.values) * fps, window_length=15, polyorder=2)
            vy = savgol_filter(np.diff(y.values) * fps, window_length=15, polyorder=2)
            ax = savgol_filter(np.diff(vx) * fps, window_length=9, polyorder=2)
            ay = savgol_filter(np.diff(vy) * fps, window_length=9, polyorder=2)

            tracking.loc[x.index[1:], f"{p}_vx"] = vx
            tracking.loc[x.index[1:], f"{p}_vy"] = vy
            tracking.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            tracking.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)

            tracking.at[x.index[0], f"{p}_vx"] = tracking.at[x.index[1], f"{p}_vx"]
            tracking.at[x.index[0], f"{p}_vy"] = tracking.at[x.index[1], f"{p}_vy"]
            tracking.at[x.index[0], f"{p}_speed"] = tracking.at[x.index[1], f"{p}_speed"]
            tracking.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0

    state_cols = ["period_id", "timestamp", "episode_id", "ball_state", "ball_owning_home_away"]
    feature_cols = [f"{p}_{f}" for p in objects for f in physical_features] + ["ball_z"]

    return tracking[state_cols + feature_cols].copy()


if __name__ == "__main__":
    EVENT_PATH = "data/ajax/event/event_v3.parquet"
    TRACKING_DIR = "data/ajax/tracking_v3"

    events = pd.read_parquet(EVENT_PATH)
    events["utc_timestamp"] = pd.to_datetime(events["time_stamp"])
    events["game_date"] = events["utc_timestamp"].dt.date
    events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)

    # Only process data from valid matches without duplicated dates
    match_dates = events[["stats_perform_match_id", "game_date"]].drop_duplicates()
    match_counts = match_dates["stats_perform_match_id"].value_counts()
    match_ids = match_counts[match_counts == 1].index
    os.makedirs("data/ajax/tracking_processed", exist_ok=True)

    for i, match_id in enumerate(match_ids[-1:]):
        if not os.path.exists(f"data/ajax/tracking_v3/{match_id}_new.parquet"):
            continue

        print(f"\n[{i}] {match_id}")
        tracking = pd.read_parquet(f"data/ajax/tracking_v3/{match_id}_new.parquet")

        tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)
        tracking["ball_z"] = (tracking["ball_z"].astype(float) / 100).round(2)  # centimeters to meters

        tracking = label_frames_and_episodes(tracking)
        tracking = calc_physical_features(tracking)

        tracking.to_parquet(f"data/ajax/tracking_processed/{match_id}.parquet")
