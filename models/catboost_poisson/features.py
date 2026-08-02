"""Feature engineering for the CatBoost Poisson model."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from models.common.teams import normalize_team_name


@dataclass(frozen=True)
class FeatureConfig:
    feature_columns: Tuple[str, ...]
    cat_feature_names: Tuple[str, ...]
    goal_cap: int


DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "team_home",
    "team_away",
    "saison",
    "spieltag",
    "wochentag",
    "home_market_value",
    "away_market_value",
    "mv_diff",
    "mv_ratio_log",
    "home_home_gf_mean",
    "home_home_ga_mean",
    "away_away_gf_mean",
    "away_away_ga_mean",
    "home_season_points_per_game",
    "home_season_gf_per_game",
    "home_season_ga_per_game",
    "home_recent_points_per_game",
    "home_recent_gf_per_game",
    "home_recent_ga_per_game",
    "away_season_points_per_game",
    "away_season_gf_per_game",
    "away_season_ga_per_game",
    "away_recent_points_per_game",
    "away_recent_gf_per_game",
    "away_recent_ga_per_game",
    "home_elo",
    "away_elo",
    "elo_diff",
)

DEFAULT_CAT_FEATURES: Tuple[str, ...] = ("team_home", "team_away", "wochentag")


def parse_result(result_str: str) -> Tuple[int, int]:
    home, away = str(result_str).split(":")
    return int(home), int(away)


def prepare_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["team_home_norm"] = out["Team Home"].map(normalize_team_name)
    out["team_away_norm"] = out["Team Away"].map(normalize_team_name)
    goals = out["Ergebnis"].map(parse_result)
    out["home_goals"] = [g[0] for g in goals]
    out["away_goals"] = [g[1] for g in goals]
    return out


def _best_fuzzy_match(needle: str, haystack: Iterable[str]) -> Tuple[str | None, float]:
    best_name = None
    best_score = 0.0
    for candidate in haystack:
        score = SequenceMatcher(None, needle, candidate).ratio()
        if score > best_score:
            best_name = candidate
            best_score = score
    return best_name, best_score


def build_mv_alias_map(
    match_teams: Iterable[str],
    mv_teams: Iterable[str],
    min_score: float = 0.92,
) -> Dict[str, str]:
    mv_set = set(mv_teams)
    alias_map: Dict[str, str] = {}

    manual = {
        normalize_team_name("Erzgebirge Aue"): normalize_team_name("FC Erzgebirge Aue"),
    }
    for k, v in manual.items():
        if v in mv_set:
            alias_map[k] = v

    for t in match_teams:
        if t in mv_set:
            alias_map.setdefault(t, t)
            continue
        if t in alias_map:
            continue
        best, score = _best_fuzzy_match(t, mv_set)
        if best is not None and score >= min_score:
            alias_map[t] = best

    return alias_map


def compute_team_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    df = train_df.copy()

    home_stats = (
        df.groupby("team_home_norm")
        .agg(
            home_gf_mean=("home_goals", "mean"),
            home_ga_mean=("away_goals", "mean"),
            home_matches=("home_goals", "size"),
        )
        .reset_index()
    )

    away_stats = (
        df.groupby("team_away_norm")
        .agg(
            away_gf_mean=("away_goals", "mean"),
            away_ga_mean=("home_goals", "mean"),
            away_matches=("away_goals", "size"),
        )
        .reset_index()
        .rename(columns={"team_away_norm": "team_home_norm"})
    )

    team_stats = home_stats.merge(away_stats, on="team_home_norm", how="outer")
    team_stats = team_stats.rename(columns={"team_home_norm": "team_norm"})

    global_home_gf = df["home_goals"].mean()
    global_home_ga = df["away_goals"].mean()
    global_away_gf = df["away_goals"].mean()
    global_away_ga = df["home_goals"].mean()

    team_stats["home_gf_mean"] = team_stats["home_gf_mean"].fillna(global_home_gf)
    team_stats["home_ga_mean"] = team_stats["home_ga_mean"].fillna(global_home_ga)
    team_stats["away_gf_mean"] = team_stats["away_gf_mean"].fillna(global_away_gf)
    team_stats["away_ga_mean"] = team_stats["away_ga_mean"].fillna(global_away_ga)
    team_stats["home_matches"] = team_stats["home_matches"].fillna(0).astype(int)
    team_stats["away_matches"] = team_stats["away_matches"].fillna(0).astype(int)

    return team_stats


def compute_team_matchday_form_table(train_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = train_df.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)

    def empty_state() -> dict:
        return {
            "played": 0,
            "points": 0,
            "gf": 0,
            "ga": 0,
            "recent_points": [],
            "recent_gf": [],
            "recent_ga": [],
        }

    rows = []

    for saison, df_s in df.groupby("Saison", sort=True):
        state: Dict[str, dict] = {}
        matchdays = sorted(df_s["Spieltag"].unique())

        for spieltag in matchdays:
            for team, st in state.items():
                played = st["played"]
                gf = st["gf"]
                ga = st["ga"]
                pts = st["points"]
                recent_pts = st["recent_points"][-window:]
                recent_gf = st["recent_gf"][-window:]
                recent_ga = st["recent_ga"][-window:]
                rows.append(
                    {
                        "team_norm": team,
                        "Saison": int(saison),
                        "Spieltag": int(spieltag),
                        "season_played": played,
                        "season_points_per_game": (pts / played) if played > 0 else np.nan,
                        "season_gf_per_game": (gf / played) if played > 0 else np.nan,
                        "season_ga_per_game": (ga / played) if played > 0 else np.nan,
                        "recent_played": len(recent_pts),
                        "recent_points_per_game": (
                            (sum(recent_pts) / len(recent_pts)) if recent_pts else np.nan
                        ),
                        "recent_gf_per_game": (
                            (sum(recent_gf) / len(recent_gf)) if recent_gf else np.nan
                        ),
                        "recent_ga_per_game": (
                            (sum(recent_ga) / len(recent_ga)) if recent_ga else np.nan
                        ),
                    }
                )

            md_matches = df_s[df_s["Spieltag"] == spieltag]
            for _, r in md_matches.iterrows():
                home = r["team_home_norm"]
                away = r["team_away_norm"]
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])

                state.setdefault(home, empty_state())
                state.setdefault(away, empty_state())

                if hg > ag:
                    home_pts, away_pts = 3, 0
                elif hg < ag:
                    home_pts, away_pts = 0, 3
                else:
                    home_pts, away_pts = 1, 1

                st_h = state[home]
                st_h["played"] += 1
                st_h["points"] += home_pts
                st_h["gf"] += hg
                st_h["ga"] += ag
                st_h["recent_points"].append(home_pts)
                st_h["recent_gf"].append(hg)
                st_h["recent_ga"].append(ag)

                st_a = state[away]
                st_a["played"] += 1
                st_a["points"] += away_pts
                st_a["gf"] += ag
                st_a["ga"] += hg
                st_a["recent_points"].append(away_pts)
                st_a["recent_gf"].append(ag)
                st_a["recent_ga"].append(hg)

        next_spieltag = int(max(matchdays)) + 1
        for team, st in state.items():
            played = st["played"]
            gf = st["gf"]
            ga = st["ga"]
            pts = st["points"]
            recent_pts = st["recent_points"][-window:]
            recent_gf = st["recent_gf"][-window:]
            recent_ga = st["recent_ga"][-window:]
            rows.append(
                {
                    "team_norm": team,
                    "Saison": int(saison),
                    "Spieltag": next_spieltag,
                    "season_played": played,
                    "season_points_per_game": (pts / played) if played > 0 else np.nan,
                    "season_gf_per_game": (gf / played) if played > 0 else np.nan,
                    "season_ga_per_game": (ga / played) if played > 0 else np.nan,
                    "recent_played": len(recent_pts),
                    "recent_points_per_game": (
                        (sum(recent_pts) / len(recent_pts)) if recent_pts else np.nan
                    ),
                    "recent_gf_per_game": (
                        (sum(recent_gf) / len(recent_gf)) if recent_gf else np.nan
                    ),
                    "recent_ga_per_game": (
                        (sum(recent_ga) / len(recent_ga)) if recent_ga else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def compute_team_matchday_elo_table(
    train_df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 50.0,
    base_rating: float = 1500.0,
) -> pd.DataFrame:
    df = train_df.sort_values(["Saison", "Spieltag"]).reset_index(drop=True)
    rows = []

    def expected_score(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    for saison, df_s in df.groupby("Saison", sort=True):
        ratings: Dict[str, float] = {}
        matchdays = sorted(df_s["Spieltag"].unique())

        for spieltag in matchdays:
            for team, r in ratings.items():
                rows.append(
                    {
                        "team_norm": team,
                        "Saison": int(saison),
                        "Spieltag": int(spieltag),
                        "elo": float(r),
                    }
                )

            md_matches = df_s[df_s["Spieltag"] == spieltag]
            for _, r in md_matches.iterrows():
                home = r["team_home_norm"]
                away = r["team_away_norm"]
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])

                ratings.setdefault(home, base_rating)
                ratings.setdefault(away, base_rating)

                r_home = ratings[home] + home_advantage
                r_away = ratings[away]

                exp_home = expected_score(r_home, r_away)
                exp_away = 1.0 - exp_home

                if hg > ag:
                    act_home, act_away = 1.0, 0.0
                elif hg < ag:
                    act_home, act_away = 0.0, 1.0
                else:
                    act_home, act_away = 0.5, 0.5

                ratings[home] = ratings[home] + k_factor * (act_home - exp_home)
                ratings[away] = ratings[away] + k_factor * (act_away - exp_away)

        next_spieltag = int(max(matchdays)) + 1
        for team, r in ratings.items():
            rows.append(
                {
                    "team_norm": team,
                    "Saison": int(saison),
                    "Spieltag": next_spieltag,
                    "elo": float(r),
                }
            )

    return pd.DataFrame(rows)


def build_features(
    X: pd.DataFrame,
    mv_df: pd.DataFrame,
    mv_alias_map: Dict[str, str],
    team_aggs: pd.DataFrame,
    team_form: pd.DataFrame,
    team_elo: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = X.copy()

    df["team_home_norm"] = df["Team Home"].map(normalize_team_name)
    df["team_away_norm"] = df["Team Away"].map(normalize_team_name)

    df["team_home_mv"] = df["team_home_norm"].map(lambda x: mv_alias_map.get(x, x))
    df["team_away_mv"] = df["team_away_norm"].map(lambda x: mv_alias_map.get(x, x))

    mv = mv_df.copy()
    if "MarketValue" in mv.columns and "market_value" not in mv.columns:
        mv = mv.rename(columns={"MarketValue": "market_value"})

    home_mv = mv.rename(columns={"team_norm": "team_home_mv"})
    away_mv = mv.rename(columns={"team_norm": "team_away_mv"})

    df = df.merge(
        home_mv[["team_home_mv", "Saison", "market_value"]].rename(
            columns={"market_value": "home_market_value"}
        ),
        on=["team_home_mv", "Saison"],
        how="left",
    )
    df = df.merge(
        away_mv[["team_away_mv", "Saison", "market_value"]].rename(
            columns={"market_value": "away_market_value"}
        ),
        on=["team_away_mv", "Saison"],
        how="left",
    )

    df["home_market_value"] = df["home_market_value"].astype(float)
    df["away_market_value"] = df["away_market_value"].astype(float)
    df["mv_diff"] = df["home_market_value"] - df["away_market_value"]

    eps = 1e-6
    df["mv_ratio_log"] = np.log((df["home_market_value"] + eps) / (df["away_market_value"] + eps))

    aggs = team_aggs.copy()
    df = df.merge(
        aggs.add_prefix("home_").rename(columns={"home_team_norm": "team_home_norm"}),
        on="team_home_norm",
        how="left",
    )
    df = df.merge(
        aggs.add_prefix("away_").rename(columns={"away_team_norm": "team_away_norm"}),
        on="team_away_norm",
        how="left",
    )

    for col in [
        "home_home_gf_mean",
        "home_home_ga_mean",
        "home_away_gf_mean",
        "home_away_ga_mean",
        "away_home_gf_mean",
        "away_home_ga_mean",
        "away_away_gf_mean",
        "away_away_ga_mean",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    te = team_elo.copy()
    home_te = te.rename(columns={"team_norm": "team_home_norm", "elo": "home_elo"})
    away_te = te.rename(columns={"team_norm": "team_away_norm", "elo": "away_elo"})
    df = df.merge(home_te, on=["team_home_norm", "Saison", "Spieltag"], how="left")
    df = df.merge(away_te, on=["team_away_norm", "Saison", "Spieltag"], how="left")
    df["home_elo"] = df["home_elo"].fillna(1500.0)
    df["away_elo"] = df["away_elo"].fillna(1500.0)
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    tf = team_form.copy()
    home_tf = tf.rename(columns={"team_norm": "team_home_norm"})
    home_tf = home_tf.rename(
        columns={
            c: f"home_{c}"
            for c in home_tf.columns
            if c not in {"team_home_norm", "Saison", "Spieltag"}
        }
    )
    away_tf = tf.rename(columns={"team_norm": "team_away_norm"})
    away_tf = away_tf.rename(
        columns={
            c: f"away_{c}"
            for c in away_tf.columns
            if c not in {"team_away_norm", "Saison", "Spieltag"}
        }
    )

    df = df.merge(home_tf, on=["team_home_norm", "Saison", "Spieltag"], how="left")
    df = df.merge(away_tf, on=["team_away_norm", "Saison", "Spieltag"], how="left")

    for col in [
        "home_season_points_per_game",
        "home_season_gf_per_game",
        "home_season_ga_per_game",
        "home_recent_points_per_game",
        "home_recent_gf_per_game",
        "home_recent_ga_per_game",
        "away_season_points_per_game",
        "away_season_gf_per_game",
        "away_season_ga_per_game",
        "away_recent_points_per_game",
        "away_recent_gf_per_game",
        "away_recent_ga_per_game",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    cols = tuple(feature_columns) if feature_columns is not None else DEFAULT_FEATURE_COLUMNS
    features = pd.DataFrame(
        {
            "team_home": df["team_home_norm"],
            "team_away": df["team_away_norm"],
            "saison": df["Saison"].astype(int),
            "spieltag": df["Spieltag"].astype(int),
            "wochentag": df["Wochentag"].astype(str),
            "home_market_value": df["home_market_value"],
            "away_market_value": df["away_market_value"],
            "mv_diff": df["mv_diff"],
            "mv_ratio_log": df["mv_ratio_log"],
            "home_home_gf_mean": df["home_home_gf_mean"],
            "home_home_ga_mean": df["home_home_ga_mean"],
            "away_away_gf_mean": df["away_away_gf_mean"],
            "away_away_ga_mean": df["away_away_ga_mean"],
            "home_season_points_per_game": df["home_season_points_per_game"],
            "home_season_gf_per_game": df["home_season_gf_per_game"],
            "home_season_ga_per_game": df["home_season_ga_per_game"],
            "home_recent_points_per_game": df["home_recent_points_per_game"],
            "home_recent_gf_per_game": df["home_recent_gf_per_game"],
            "home_recent_ga_per_game": df["home_recent_ga_per_game"],
            "away_season_points_per_game": df["away_season_points_per_game"],
            "away_season_gf_per_game": df["away_season_gf_per_game"],
            "away_season_ga_per_game": df["away_season_ga_per_game"],
            "away_recent_points_per_game": df["away_recent_points_per_game"],
            "away_recent_gf_per_game": df["away_recent_gf_per_game"],
            "away_recent_ga_per_game": df["away_recent_ga_per_game"],
            "home_elo": df["home_elo"],
            "away_elo": df["away_elo"],
            "elo_diff": df["elo_diff"],
        }
    )
    return features[list(cols)]
