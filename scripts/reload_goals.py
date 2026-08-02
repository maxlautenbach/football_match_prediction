"""Reload BL1 goal-scorer events from OpenLigaDB into data/goals_df_{year}.pck.

Uses the season endpoint (goals are included per match) — no per-match calls.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from api import openligadb
from data_loader import DATA_DIR, extract_goal_events


def load_season_goals(league: str, season: int) -> list[dict]:
    rows: list[dict] = []
    for match_json in openligadb.get_all_season_matches(league, season):
        rows.extend(extract_goal_events(match_json, league=league))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reload goal events from OpenLigaDB")
    parser.add_argument("--start", "--from-year", dest="start", type=int, default=2009)
    parser.add_argument("--end", "--to-year", dest="end", type=int, default=2025)
    parser.add_argument("--leagues", nargs="+", default=["bl1"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing goals_df_*.pck files",
    )
    args = parser.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start, args.end + 1):
        out = args.data_dir / f"goals_df_{year}.pck"
        if out.exists() and not args.force:
            print(f"  {year}: {out.name} exists — skip (use --force to overwrite)")
            continue

        rows: list[dict] = []
        for league in args.leagues:
            try:
                league_rows = load_season_goals(league, year)
                rows.extend(league_rows)
                print(f"  {year} {league}: {len(league_rows)} goals")
            except Exception as e:
                print(f"  {year} {league}: failed — {e}")

        if not rows:
            print(f"  {year}: no goals, skipping")
            continue

        pickle.dump(pd.DataFrame(rows), open(out, "wb"))
        print(f"  Saved {out.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
