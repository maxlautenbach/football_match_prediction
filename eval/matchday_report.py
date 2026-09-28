"""Live matchday performance vs archived tips / expected points."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "scripts"))

from eval.metrics import kicktipp_points_one, parse_result
from tip_archive import list_tip_archives, load_tips


def _result_str(home_goals: Any, away_goals: Any) -> Optional[str]:
    try:
        if home_goals is None or away_goals is None:
            return None
        if isinstance(home_goals, float) and math.isnan(home_goals):
            return None
        if isinstance(away_goals, float) and math.isnan(away_goals):
            return None
        return f"{int(home_goals)}:{int(away_goals)}"
    except (TypeError, ValueError):
        return None


def _match_key(home: str, away: str) -> tuple[str, str]:
    return str(home).strip().lower(), str(away).strip().lower()


def score_matchday(
    season: int,
    matchday: int,
    match_df: pd.DataFrame,
    *,
    tips_dir: Path | str | None = None,
) -> Optional[dict[str, Any]]:
    """Score archived tips for one matchday against finished results.

    Returns None if no tip archive exists.
    """
    archive = load_tips(season, matchday, tips_dir=tips_dir)
    if archive is None:
        return None

    subset = match_df[
        (match_df["matchDay"].astype(int) == int(matchday))
        & (match_df["season"].astype(int) == int(season))
    ].copy()
    if "league" in subset.columns:
        bl1 = subset[subset["league"].astype(str).str.lower() == "bl1"]
        if len(bl1) > 0:
            subset = bl1

    by_teams: dict[tuple[str, str], pd.Series] = {}
    for _, row in subset.iterrows():
        by_teams[_match_key(row["teamHomeName"], row["teamAwayName"])] = row

    rows: list[dict[str, Any]] = []
    points_total = 0
    expected_total = 0.0
    variance_total = 0.0
    n_scored = 0
    n_exact = 0
    n_outcome = 0
    n_pending = 0
    has_expected = False

    for m in archive.get("matches", []):
        home = str(m["home_team"])
        away = str(m["away_team"])
        tip = str(m["tip"])
        exp = m.get("expected_points")
        var = m.get("variance")
        try:
            exp_f = float(exp) if exp is not None else float("nan")
        except (TypeError, ValueError):
            exp_f = float("nan")
        try:
            var_f = float(var) if var is not None else float("nan")
        except (TypeError, ValueError):
            var_f = float("nan")

        result_row = by_teams.get(_match_key(home, away))
        result = None
        status = "missing"
        if result_row is not None:
            status = str(result_row.get("status", ""))
            result = _result_str(result_row.get("goalsHome"), result_row.get("goalsAway"))
            if result is None and status == "finished":
                status = "no_score"

        points: int | None = None
        if result is not None and status == "finished":
            points = kicktipp_points_one(result, tip)
            if points is not None:
                n_scored += 1
                points_total += points
                th, ta = parse_result(result)
                ph, pa = parse_result(tip)
                if th is not None and ph is not None:
                    if th == ph and ta == pa:
                        n_exact += 1
                    true_out = 1 if th > ta else (0 if th == ta else -1)
                    pred_out = 1 if ph > pa else (0 if ph == pa else -1)
                    if true_out == pred_out:
                        n_outcome += 1
                if exp_f == exp_f:
                    expected_total += exp_f
                    has_expected = True
                if var_f == var_f:
                    variance_total += max(0.0, var_f)
        else:
            n_pending += 1

        rows.append(
            {
                "home_team": home,
                "away_team": away,
                "tip": tip,
                "result": result,
                "status": status,
                "points": points,
                "expected_points": exp_f,
                "variance": var_f,
            }
        )

    delta = float("nan")
    z_score = float("nan")
    if has_expected and n_scored > 0:
        delta = float(points_total) - expected_total
        if variance_total > 1e-12:
            z_score = delta / math.sqrt(variance_total)
        elif abs(delta) <= 1e-12:
            z_score = 0.0

    return {
        "season": int(season),
        "matchday": int(matchday),
        "n_matches": len(rows),
        "n_scored": n_scored,
        "n_pending": n_pending,
        "points": int(points_total),
        "expected": float(expected_total) if has_expected else float("nan"),
        "delta": delta,
        "z_score": z_score,
        "n_exact": n_exact,
        "n_outcome": n_outcome,
        "matches": rows,
        "complete": n_pending == 0 and n_scored == len(rows) and len(rows) > 0,
    }


def season_standing(
    season: int,
    match_df: pd.DataFrame,
    *,
    tips_dir: Path | str | None = None,
    up_to_matchday: int | None = None,
) -> dict[str, Any]:
    """Aggregate scored tip archives for a season."""
    paths = list_tip_archives(season, tips_dir=tips_dir)
    reports: list[dict[str, Any]] = []
    points = 0
    expected = 0.0
    variance = 0.0
    n_scored = 0
    has_expected = False

    for path in paths:
        # season_YYYY_md_N.json
        try:
            md = int(path.stem.split("_md_")[-1])
        except ValueError:
            continue
        if up_to_matchday is not None and md > int(up_to_matchday):
            continue
        report = score_matchday(season, md, match_df, tips_dir=tips_dir)
        if report is None or report["n_scored"] == 0:
            continue
        reports.append(report)
        points += int(report["points"])
        n_scored += int(report["n_scored"])
        if report["expected"] == report["expected"]:
            expected += float(report["expected"])
            has_expected = True
        for m in report["matches"]:
            var = m.get("variance", float("nan"))
            if m.get("points") is not None and var == var:
                variance += max(0.0, float(var))

    delta = float("nan")
    z_score = float("nan")
    if has_expected and n_scored > 0:
        delta = float(points) - expected
        if variance > 1e-12:
            z_score = delta / math.sqrt(variance)
        elif abs(delta) <= 1e-12:
            z_score = 0.0

    return {
        "season": int(season),
        "n_matchdays": len(reports),
        "n_scored": n_scored,
        "points": int(points),
        "expected": float(expected) if has_expected else float("nan"),
        "delta": delta,
        "z_score": z_score,
        "matchdays": [r["matchday"] for r in reports],
    }


def find_previous_scorable_matchday(
    season: int,
    current_tip_matchday: int,
    match_df: pd.DataFrame,
    *,
    tips_dir: Path | str | None = None,
) -> Optional[dict[str, Any]]:
    """Score the most recent archived matchday before the newly tipped one."""
    for md in range(int(current_tip_matchday) - 1, 0, -1):
        report = score_matchday(season, md, match_df, tips_dir=tips_dir)
        if report is not None:
            return report
    return None
