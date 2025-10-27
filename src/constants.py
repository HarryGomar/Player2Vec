from __future__ import annotations

NUMERIC_STATS = [
    "shots_per90", "goals_per90", "xg_total", "xg_per90", "passes_per90", "carries_per90", "dribbles_per90",
    "interceptions_per90", "blocks_per90", "clearances_per90", "duels_per90", "recoveries_per90", "pass_len_avg",
    "pass_len_p25", "pass_len_p50", "pass_len_p75", "pass_comp_rate", "pass_expected_success_mean", "xa_proxy",
    "obv_total_net_sum", "obv_for_net_sum", "obv_against_net_sum", "obv_total_net_per90", "obv_for_net_per90",
    "obv_against_net_per90", "gk_actions", "gk_actions_per90", "gk_success_total", "gk_success_rate",
    "gk_success_in_play", "gk_success_out", "gk_in_play_rate", "gk_out_rate", "gk_saved_to_post",
    "gk_saved_off_target", "gk_punched_out",
]

ID_COLS = [
    "ps_index", "player_id", "player_name", "season_id", "team_id", "team_season_id", "team_name",
    "position_mode", "position_mode_coarse", "pos_group", "minutes", "events_count"
]

DEFAULT_MIN_MINUTES = 300

SEASON_LABELS = {
    108: "2021/2022",
    "108": "2021/2022",
    235: "2022/2023",
    "235": "2022/2023",
    281: "2023/2024",
    "281": "2023/2024",
    317: "2024/2025",
    "317": "2024/2025",
}

CATEGORICAL_DIMENSIONS = [
    "team_name", "season_id", "season_label", "position_mode", "position_mode_coarse", "pos_group"
]

DEFAULT_HOVER_COLUMNS = [
    "player_name", "team_name", "season_label", "season_id", "minutes", "position_mode"
]

EMBED_COL = "z_style_unit"