import numpy as np
import pandas as pd

FIELD_SIZE = (105.0, 68.0)
GOAL_SIZE = 7.32
GOAL_XY = np.array(
    [
        (0, FIELD_SIZE[1] / 2 - GOAL_SIZE / 2),
        (0, FIELD_SIZE[1] / 2 + GOAL_SIZE / 2),
        (FIELD_SIZE[0], FIELD_SIZE[1] / 2 - GOAL_SIZE / 2),
        (FIELD_SIZE[0], FIELD_SIZE[1] / 2 + GOAL_SIZE / 2),
    ]
)

LINEUP_HEADER = [
    "stats_perform_match_id",
    "game_date",
    "contestant_name",
    "player_id",
    "object_id",
    "shirt_number",
    "match_name",
    "formation",
    "advanced_position",
    "mins_played",
    "start_time",
    "end_time",
]

DEFENSE = [
    "interception",
    "tackle",
    "shot_block",
    "keeper_save",
    "induce_out",
    "prevent",
    "concede",
]

# DEFCON_HEADER = LINEUP_HEADER + ["attack_time", "defend_time", "defcon", "defcon_normal"] + SCORE_COLS

# Categories for SPADL event types
PASS = [
    "pass",
    "cross",
    "throw_in",
    "goalkick",
    "corner_short",
    "corner_crossed",
    "freekick_short",
    "freekick_crossed",
]

INCOMING = [
    "interception",
    "ball_recovery",
    "shot_block",
    "keeper_save",
    "keeper_punch",
    "keeper_claim",
    "keeper_pick_up",
    "keeper_sweeper",
]

SHOT = ["shot", "shot_freekick", "shot_penalty"]

DEFENSIVE_TOUCH = ["interception", "bad_touch", "shot_block", "keeper_save", "keeper_punch"]

SET_PIECE_OOP = ["throw_in", "goalkick", "corner_short", "corner_crossed"]
SET_PIECE_FOUL = ["freekick_short", "freekick_crossed", "shot_freekick", "shot_penalty"]
SET_PIECE = SET_PIECE_OOP + SET_PIECE_FOUL

TASK_CONFIG = pd.DataFrame(
    {
        # Player-level option valuation
        "overall_scoring": ["graph_binary", True, True, True, False, False, None, None, 1],
        "overall_conceding": ["graph_binary", True, True, True, False, False, None, None, 1],
        "overall_return": ["graph_regression", True, True, True, False, False, None, None, 1],
        "action_intent": ["node_selection", True, True, True, True, True, None, "teammates", 1],
        "pass_intent": ["node_selection", True, True, False, True, False, None, "teammates", 1],
        "pass_intent_oppo_agn": ["node_selection", True, False, False, True, False, None, "teammates", 1],
        "action_success": ["node_binary", True, True, True, True, True, "intent", "teammates", 1],
        "pass_success": ["node_binary", True, True, False, True, False, "intent", "teammates", 1],
        "shot_blocking": ["graph_binary", True, False, True, False, True, None, None, 1],
        "intent_return": ["node_binary", True, True, False, True, False, "intent", "teammates", 2],
        "intent_return_oppo_agn": ["node_binary", True, True, False, True, False, "intent", "teammates", 2],
        "outcome_scoring": ["node_binary", True, True, False, True, False, "success", "teammates", 2],
        "outcome_conceding": ["node_binary", True, True, False, True, False, "success", "teammates", 2],
        "outcome_return": ["node_regression", True, True, False, True, False, "success", "teammates", 2],
        "success_receiver": ["node_selection", True, False, False, True, False, "direction", "teammates", 1],
        "failure_receiver": ["node_selection", True, True, True, True, True, "intent", "opponents", 1],
        # Space-level option valuation
        "pass_dest": ["graph_multiclass", True, False, False, True, False, None, None, 1],
        "pass_dest_oppo_agn": ["graph_multiclass", True, False, False, True, False, None, None, 1],
        "dest_receiver": ["node_selection", True, False, False, True, False, "destination", None, 1],
        "dest_scoring": ["graph_binary", True, False, False, True, False, "success", None, 2],
        "dest_conceding": ["graph_binary", True, False, False, True, False, "success", None, 2],
    },
    index=["gnn_task", "pass", "dribble", "shot", "intended", "include_goals", "condition", "out_filter", "out_dim"],
).T
