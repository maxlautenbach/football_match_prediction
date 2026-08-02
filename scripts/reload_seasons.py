"""Reload historical seasons from OpenLigaDB into data/match_df_{year}.pck."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from data_loader import DATA_DIR, extract_match_data
from api import openligadb


def load_season(league: str, season: int) -> list[dict]:
    rows = []
    json_list = openligadb.get_all_season_matches(league, season)
    for match_json in json_list:
        rows.append(extract_match_data(match_json, league=league))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reload seasons from OpenLigaDB")
    parser.add_argument("--from-year", type=int, default=2009)
    parser.add_argument("--to-year", type=int, default=2025)
    parser.add_argument("--leagues", nargs="+", default=["bl1", "bl2"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.from_year, args.to_year + 1):
        rows: list[dict] = []
        for league in args.leagues:
            try:
                league_rows = load_season(league, year)
                rows.extend(league_rows)
                print(f"  {year} {league}: {len(league_rows)} matches")
            except Exception as e:
                print(f"  {year} {league}: failed — {e}")
        if not rows:
            print(f"  {year}: no data, skipping")
            continue
        out = args.data_dir / f"match_df_{year}.pck"
        pickle.dump(pd.DataFrame(rows), open(out, "wb"))
        print(f"  Saved {out.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
