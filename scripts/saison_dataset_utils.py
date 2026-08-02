"""Build season-level Kicktipp outlook labels and pre-season features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from dataset_utils import (
    DEFAULT_HOLDOUT_SEASON,
    DEFAULT_START_SEASON,
    discover_season_years,
    load_market_values,
)
from models.common.teams import normalize_team_name

POINTS_PER_CORRECT = 6
MAX_SAISON_SCORE = 24
HERBST_MATCHDAY = 17
FINAL_MATCHDAY = 34


def _load_raw_matches(data_dir: Path, start_season: int = DEFAULT_START_SEASON) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in discover_season_years(data_dir, start_season=start_season):
        path = data_dir / f"match_df_{year}.pck"
        raw = pickle.load(open(path, "rb"))
        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError(f"No match_df_*.pck found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


def _load_goals(data_dir: Path, start_season: int = DEFAULT_START_SEASON) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in discover_season_years(data_dir, start_season=start_season):
        path = data_dir / f"goals_df_{year}.pck"
        if not path.exists():
            continue
        raw = pickle.load(open(path, "rb"))
        df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
        if len(df):
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No goals_df_*.pck in {data_dir}. Run scripts/reload_goals.py first."
        )
    return pd.concat(frames, ignore_index=True)


def encode_bottom3(teams: Iterable[str]) -> str:
    """Canonical unordered trio encoding (sorted, pipe-separated)."""
    cleaned = sorted({normalize_team_name(t) for t in teams if t})
    return "|".join(cleaned)


def decode_bottom3(encoded: str) -> list[str]:
    if not encoded or (isinstance(encoded, float) and np.isnan(encoded)):
        return []
    return [t for t in str(encoded).split("|") if t]


def compute_table(matches: pd.DataFrame, max_matchday: int | None = None) -> pd.DataFrame:
    """Build a league table from finished matches (3/1/0, GD, GF tiebreak)."""
    df = matches[matches["status"] == "finished"].copy()
    if max_matchday is not None:
        df = df[df["matchDay"] <= max_matchday]
    if df.empty:
        return pd.DataFrame(
            columns=["Team", "Played", "Points", "GF", "GA", "GD", "Rank"]
        )

    records: dict[str, dict[str, float]] = {}

    def _row(team: str) -> dict[str, float]:
        if team not in records:
            records[team] = {"Played": 0, "Points": 0, "GF": 0, "GA": 0}
        return records[team]

    for _, m in df.iterrows():
        home = normalize_team_name(m["teamHomeName"])
        away = normalize_team_name(m["teamAwayName"])
        gh = int(m["goalsHome"])
        ga = int(m["goalsAway"])
        h = _row(home)
        a = _row(away)
        h["Played"] += 1
        a["Played"] += 1
        h["GF"] += gh
        h["GA"] += ga
        a["GF"] += ga
        a["GA"] += gh
        if gh > ga:
            h["Points"] += 3
        elif gh < ga:
            a["Points"] += 3
        else:
            h["Points"] += 1
            a["Points"] += 1

    table = pd.DataFrame(
        [
            {
                "Team": team,
                "Played": int(stats["Played"]),
                "Points": int(stats["Points"]),
                "GF": int(stats["GF"]),
                "GA": int(stats["GA"]),
                "GD": int(stats["GF"] - stats["GA"]),
            }
            for team, stats in records.items()
        ]
    )
    table = table.sort_values(
        ["Points", "GD", "GF", "Team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    table["Rank"] = np.arange(1, len(table) + 1)
    return table


def top_scorer_team(
    goals: pd.DataFrame,
    final_table: pd.DataFrame,
) -> str | None:
    """
    Team of the top goalscorer (own goals excluded).

    Tie-break: player with more goals wins; if still tied, prefer the player
    whose team finished higher in the final table (lower Rank).
    """
    if goals.empty or final_table.empty:
        return None
    g = goals[~goals["isOwnGoal"].fillna(False).astype(bool)].copy()
    g = g[g["goalGetterName"].notna() & (g["goalGetterName"].astype(str).str.strip() != "")]
    if g.empty:
        return None

    g["scoringTeamName"] = g["scoringTeamName"].map(normalize_team_name)
    counts = (
        g.groupby(["goalGetterId", "goalGetterName", "scoringTeamName"], dropna=False)
        .size()
        .reset_index(name="Goals")
    )
    rank_map = {
        normalize_team_name(r["Team"]): int(r["Rank"]) for _, r in final_table.iterrows()
    }
    counts["TeamRank"] = counts["scoringTeamName"].map(rank_map).fillna(999)
    counts = counts.sort_values(
        ["Goals", "TeamRank", "goalGetterName"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    team = counts.iloc[0]["scoringTeamName"]
    return team if isinstance(team, str) and team else None


def season_labels_for_year(
    matches: pd.DataFrame,
    goals: pd.DataFrame,
    season: int,
    league: str = "bl1",
) -> dict | None:
    season_matches = matches[
        (matches["season"] == season) & (matches["league"].astype(str).str.lower() == league)
    ]
    finished = season_matches[season_matches["status"] == "finished"]
    if finished.empty:
        return None

    final = compute_table(finished, max_matchday=FINAL_MATCHDAY)
    herbst = compute_table(finished, max_matchday=HERBST_MATCHDAY)
    if len(final) < 3:
        return None

    season_goals = goals[
        (goals["season"] == season) & (goals["league"].astype(str).str.lower() == league)
    ]
    scorer_team = top_scorer_team(season_goals, final)
    bottom = final.tail(3)["Team"].tolist()

    return {
        "Saison": int(season),
        "Liga": league,
        "champion": final.iloc[0]["Team"],
        "herbstmeister": herbst.iloc[0]["Team"] if len(herbst) else None,
        "bottom3": encode_bottom3(bottom),
        "top_scorer_team": scorer_team,
        "n_teams": int(len(final)),
        "n_matches": int(len(finished)),
    }


def build_season_tables(
    matches: pd.DataFrame,
    league: str = "bl1",
) -> pd.DataFrame:
    """Final tables for every finished season (for prior-year features)."""
    rows: list[pd.DataFrame] = []
    seasons = sorted(matches["season"].dropna().unique())
    for season in seasons:
        season_matches = matches[
            (matches["season"] == season)
            & (matches["league"].astype(str).str.lower() == league)
        ]
        finished = season_matches[season_matches["status"] == "finished"]
        if finished.empty:
            continue
        table = compute_table(finished, max_matchday=FINAL_MATCHDAY)
        table["Saison"] = int(season)
        table["Liga"] = league
        rows.append(table)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def bl1_teams_for_season(matches: pd.DataFrame, season: int) -> list[str]:
    m = matches[
        (matches["season"] == season)
        & (matches["league"].astype(str).str.lower() == "bl1")
    ]
    teams = set(m["teamHomeName"].map(normalize_team_name)) | set(
        m["teamAwayName"].map(normalize_team_name)
    )
    return sorted(t for t in teams if t)


def build_team_features(
    matches: pd.DataFrame,
    mv_df: pd.DataFrame,
    seasons: list[int],
) -> pd.DataFrame:
    """Pre-season team features for each BL1 season (causal: prior year only)."""
    bl1_tables = build_season_tables(matches, league="bl1")
    bl2_tables = build_season_tables(matches, league="bl2")

    mv = mv_df.copy()
    mv["Team"] = mv["Team"].map(normalize_team_name)
    mv["Saison"] = mv["Saison"].astype(int)

    # Median BL1 prior GF as fallback for promoted sides without BL2 history
    bl1_prior_gf_by_season: dict[int, float] = {}
    if not bl1_tables.empty:
        for s, grp in bl1_tables.groupby("Saison"):
            bl1_prior_gf_by_season[int(s)] = float(grp["GF"].median())

    rows: list[dict] = []
    for season in seasons:
        teams = bl1_teams_for_season(matches, season)
        if len(teams) < 18:
            # Incomplete / future season — still emit rows if we have teams
            if not teams:
                continue

        prior = season - 1
        bl1_prior = (
            bl1_tables[bl1_tables["Saison"] == prior].set_index("Team")
            if not bl1_tables.empty
            else pd.DataFrame()
        )
        bl2_prior = (
            bl2_tables[bl2_tables["Saison"] == prior].set_index("Team")
            if not bl2_tables.empty
            else pd.DataFrame()
        )
        mv_season = mv[mv["Saison"] == season].set_index("Team")["MarketValue"]

        for team in teams:
            is_promoted = team not in bl1_prior.index if len(bl1_prior) else True
            if team in bl1_prior.index:
                prior_rank = float(bl1_prior.loc[team, "Rank"])
                prior_points = float(bl1_prior.loc[team, "Points"])
                prior_gf = float(bl1_prior.loc[team, "GF"])
            elif team in bl2_prior.index:
                # Map BL2 finish into a weak BL1 prior (ranks 16–20-ish)
                bl2_rank = float(bl2_prior.loc[team, "Rank"])
                prior_rank = 15.0 + min(bl2_rank, 3.0)
                prior_points = float(bl2_prior.loc[team, "Points"]) * 0.5
                prior_gf = float(bl2_prior.loc[team, "GF"]) * 0.7
            else:
                prior_rank = 18.0
                prior_points = 20.0
                prior_gf = bl1_prior_gf_by_season.get(prior, 40.0)

            market_value = float(mv_season.get(team, np.nan)) if team in mv_season.index else np.nan
            rows.append(
                {
                    "Saison": int(season),
                    "Team": team,
                    "MarketValue": market_value,
                    "prior_rank": prior_rank,
                    "prior_points": prior_points,
                    "prior_gf": prior_gf,
                    "is_promoted": bool(is_promoted),
                }
            )

    feat = pd.DataFrame(rows)
    if feat.empty:
        return feat

    # Fill missing market values with season median (or global)
    for season, grp_idx in feat.groupby("Saison").groups.items():
        idx = list(grp_idx)
        vals = feat.loc[idx, "MarketValue"]
        med = float(vals.median()) if vals.notna().any() else float(feat["MarketValue"].median())
        if np.isnan(med):
            med = 1.0
        feat.loc[idx, "MarketValue"] = vals.fillna(med)

    feat["log_mv"] = np.log(feat["MarketValue"].clip(lower=1.0))
    feat["mv_rank"] = feat.groupby("Saison")["MarketValue"].rank(
        ascending=False, method="average"
    )
    return feat.sort_values(["Saison", "mv_rank", "Team"]).reset_index(drop=True)


def build_saison_dataset(
    data_dir: Path,
    *,
    holdout_season: int = DEFAULT_HOLDOUT_SEASON,
    start_season: int = DEFAULT_START_SEASON,
    datasets_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (labels_df, team_features_df, split labels into train/holdout via season).

    Actually returns (train_labels, holdout_labels, team_features).
    """
    if datasets_dir is None:
        datasets_dir = data_dir.parent / "datasets"

    matches = _load_raw_matches(data_dir, start_season=start_season)
    goals = _load_goals(data_dir, start_season=start_season)
    mv_df = load_market_values(data_dir, datasets_dir=datasets_dir)

    # Normalize names on matches for consistent joins
    matches = matches.copy()
    matches["teamHomeName"] = matches["teamHomeName"].map(normalize_team_name)
    matches["teamAwayName"] = matches["teamAwayName"].map(normalize_team_name)
    goals = goals.copy()
    if "scoringTeamName" in goals.columns:
        goals["scoringTeamName"] = goals["scoringTeamName"].map(normalize_team_name)

    seasons = sorted(
        s
        for s in matches.loc[
            matches["league"].astype(str).str.lower() == "bl1", "season"
        ]
        .dropna()
        .unique()
    )
    seasons = [int(s) for s in seasons if int(s) >= start_season]

    label_rows: list[dict] = []
    for season in seasons:
        if season > holdout_season:
            # Future season without finished labels
            continue
        labels = season_labels_for_year(matches, goals, season, league="bl1")
        if labels is None:
            continue
        # Require finished season roughly (at least ~300 matches) for labels
        if season < holdout_season and labels["n_matches"] < 250:
            print(f"  Warning: season {season} has only {labels['n_matches']} finished matches")
        if labels["top_scorer_team"] is None:
            print(f"  Warning: season {season} missing top_scorer_team")
        label_rows.append(labels)

    labels_df = pd.DataFrame(label_rows).sort_values("Saison").reset_index(drop=True)
    feature_seasons = [s for s in seasons if s <= holdout_season]
    # Also include holdout even if we only need features for prediction years
    team_features = build_team_features(matches, mv_df, feature_seasons)

    train_labels = labels_df[labels_df["Saison"] < holdout_season].copy()
    holdout_labels = labels_df[labels_df["Saison"] == holdout_season].copy()
    return train_labels, holdout_labels, team_features
