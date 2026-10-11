"""Persist submitted tips so matchdays can be scored later."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DEFAULT_TIPS_DIR = BASE_DIR / "data" / "tips"


def tip_archive_path(
    season: int,
    matchday: int,
    *,
    tips_dir: Path | str | None = None,
) -> Path:
    root = Path(tips_dir) if tips_dir is not None else DEFAULT_TIPS_DIR
    return root / f"season_{int(season)}_md_{int(matchday)}.json"


def _match_key(home: str, away: str) -> str:
    return f"{home.strip().lower()}|{away.strip().lower()}"


def save_tips(
    results_df: pd.DataFrame,
    *,
    tips_dir: Path | str | None = None,
    created_at: Optional[datetime] = None,
) -> Path:
    """Write tip archive JSON for the matchday in ``results_df``.

    Merges into an existing archive by home/away so a partial mid-matchday
    re-tip cannot wipe earlier tips for the same matchday.

    Expected columns: Home_Team, Away_Team, Date, Matchday, Season, Prediction,
    and optionally Expected_Points, Variance.
    """
    if len(results_df) == 0:
        raise ValueError("Cannot archive empty tip sheet")

    season = int(results_df["Season"].iloc[0])
    matchday = int(results_df["Matchday"].iloc[0])
    path = tip_archive_path(season, matchday, tips_dir=tips_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = created_at or datetime.now(timezone.utc)
    by_key: dict[str, dict[str, Any]] = {}

    existing = load_tips(season, matchday, tips_dir=tips_dir)
    if existing:
        for m in existing.get("matches") or []:
            key = _match_key(str(m.get("home_team", "")), str(m.get("away_team", "")))
            if key != "|":
                by_key[key] = m

    for _, row in results_df.iterrows():
        date_val = row.get("Date")
        if hasattr(date_val, "isoformat"):
            date_str = date_val.isoformat()
        else:
            date_str = str(date_val)

        exp = row.get("Expected_Points", float("nan"))
        var = row.get("Variance", float("nan"))
        try:
            exp_f = float(exp)
        except (TypeError, ValueError):
            exp_f = float("nan")
        try:
            var_f = float(var)
        except (TypeError, ValueError):
            var_f = float("nan")

        home = str(row["Home_Team"])
        away = str(row["Away_Team"])
        by_key[_match_key(home, away)] = {
            "home_team": home,
            "away_team": away,
            "date": date_str,
            "tip": str(row["Prediction"]),
            "expected_points": exp_f,
            "variance": var_f,
        }

    matches = list(by_key.values())
    payload = {
        "season": season,
        "matchday": matchday,
        "created_at": ts.isoformat(),
        "n_matches": len(matches),
        "matches": matches,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_tips(
    season: int,
    matchday: int,
    *,
    tips_dir: Path | str | None = None,
) -> Optional[dict[str, Any]]:
    path = tip_archive_path(season, matchday, tips_dir=tips_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_tip_archives(
    season: int,
    *,
    tips_dir: Path | str | None = None,
) -> list[Path]:
    root = Path(tips_dir) if tips_dir is not None else DEFAULT_TIPS_DIR
    if not root.exists():
        return []
    prefix = f"season_{int(season)}_md_"
    paths = sorted(root.glob(f"{prefix}*.json"))
    return paths
