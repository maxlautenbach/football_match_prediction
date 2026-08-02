"""Create train/holdout CSV datasets from pickle files (season holdout)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from data_loader import update_match_data_delta, update_next_matchday_df
from dataset_utils import DEFAULT_HOLDOUT_SEASON, generate_datasets_from_pickle

DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create train/holdout CSV datasets")
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--skip-delta", action="store_true")
    args = parser.parse_args(argv)

    DATASETS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Dataset Creation (holdout season={args.holdout_season})")
    print("=" * 60)

    if not args.skip_delta:
        print("\nStep 1: Running delta update...")
        update_match_data_delta(data_dir=DATA_DIR, verbose=True)
        update_next_matchday_df(data_dir=DATA_DIR, verbose=True)
    else:
        print("\nStep 1: Skipping delta update")

    print("\nStep 2: Generating datasets...")
    train_df, holdout_df, mv_df = generate_datasets_from_pickle(
        DATA_DIR,
        holdout_season=args.holdout_season,
        datasets_dir=DATASETS_DIR,
    )

    train_df.to_csv(DATASETS_DIR / "train.csv", index=False)
    holdout_df.to_csv(DATASETS_DIR / "test.csv", index=False)
    mv_df.to_csv(DATASETS_DIR / "TeamMarketValues.csv", index=False)

    print(f"\nSaved train.csv ({len(train_df)}), test.csv ({len(holdout_df)}), MV ({len(mv_df)})")


if __name__ == "__main__":
    main()
