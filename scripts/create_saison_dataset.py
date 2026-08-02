"""Create season-outlook train/holdout CSVs and team feature table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from dataset_utils import DEFAULT_HOLDOUT_SEASON, DEFAULT_START_SEASON
from saison_dataset_utils import build_saison_dataset

DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Kicktipp saison outlook datasets")
    parser.add_argument("--holdout-season", type=int, default=DEFAULT_HOLDOUT_SEASON)
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    args = parser.parse_args(argv)

    args.datasets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building saison dataset (holdout={args.holdout_season})...")
    train_labels, holdout_labels, team_features = build_saison_dataset(
        args.data_dir,
        holdout_season=args.holdout_season,
        start_season=args.start_season,
        datasets_dir=args.datasets_dir,
    )

    train_path = args.datasets_dir / "saison_train.csv"
    holdout_path = args.datasets_dir / "saison_holdout.csv"
    feats_path = args.datasets_dir / "saison_team_features.csv"

    train_labels.to_csv(train_path, index=False)
    holdout_labels.to_csv(holdout_path, index=False)
    team_features.to_csv(feats_path, index=False)

    print(f"  Train seasons: {len(train_labels)} → {train_path.name}")
    print(f"  Holdout seasons: {len(holdout_labels)} → {holdout_path.name}")
    print(f"  Team features: {len(team_features)} → {feats_path.name}")
    if len(holdout_labels):
        row = holdout_labels.iloc[0]
        print(
            f"  Holdout {int(row['Saison'])}: champion={row['champion']}, "
            f"herbst={row['herbstmeister']}, bottom3={row['bottom3']}, "
            f"scorer={row['top_scorer_team']}"
        )


if __name__ == "__main__":
    main()
