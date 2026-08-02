"""Promote an MLflow model (run or registry URI) into artifacts/ for production."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.mlflow_utils import (
    REGISTERED_MODEL_NAME,
    download_model_uri,
    download_run_artifacts,
    set_model_alias,
    setup_mlflow,
)
from mlflow.tracking import MlflowClient

ARTIFACTS_DIR = BASE_DIR / "artifacts"


def _find_bundle(root: Path) -> Path:
    """Locate directory containing meta.json inside a downloaded model tree."""
    if (root / "meta.json").exists():
        return root
    # pyfunc layout: .../model/artifacts/bundle/meta.json
    candidates = list(root.rglob("meta.json"))
    if not candidates:
        raise FileNotFoundError(f"No meta.json under {root}")
    return candidates[0].parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote MLflow model into artifacts/ and optionally set registry alias"
    )
    parser.add_argument("--run-id", default=None, help="MLflow run id (legacy path)")
    parser.add_argument(
        "--model-uri",
        default=None,
        help="e.g. models:/bundesliga-kicktipp@candidate or models:/bundesliga-kicktipp/1",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help=f"Promote registry model '{REGISTERED_MODEL_NAME}' by alias (e.g. candidate)",
    )
    parser.add_argument(
        "--set-production-alias",
        action="store_true",
        help="After promote, set @production on the chosen version",
    )
    parser.add_argument("--dst", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args(argv)

    if sum(x is not None for x in (args.run_id, args.model_uri, args.alias)) != 1:
        parser.error("Specify exactly one of --run-id, --model-uri, or --alias")

    setup_mlflow()
    dst: Path = args.dst

    if args.backup and dst.exists():
        backup = dst.parent / f"{dst.name}_prev"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(dst, backup)
        print(f"Backed up existing artifacts to {backup}")

    tmp = BASE_DIR / ".mlflow_promote_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)

    version_for_alias: str | None = None
    if args.alias:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{args.alias}"
        print(f"Downloading {model_uri} ...")
        local = download_model_uri(model_uri, tmp)
        client = MlflowClient()
        mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, args.alias)
        version_for_alias = mv.version
    elif args.model_uri:
        print(f"Downloading {args.model_uri} ...")
        local = download_model_uri(args.model_uri, tmp)
        # Try parse models:/name/version
        if args.model_uri.startswith("models:/") and "/" in args.model_uri.split(":", 1)[1]:
            parts = args.model_uri.replace("models:/", "").split("/")
            if len(parts) == 2 and parts[1].isdigit():
                version_for_alias = parts[1]
    else:
        print(f"Downloading run {args.run_id} artifacts ...")
        local = download_run_artifacts(args.run_id, tmp, artifact_path="model")

    source = _find_bundle(local if local.exists() else tmp)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Promoted → {dst}")

    if args.set_production_alias:
        if version_for_alias is None:
            raise SystemExit("--set-production-alias requires --alias or models:/name/version URI")
        set_model_alias(REGISTERED_MODEL_NAME, "production", version_for_alias)
        print(f"Set alias @{REGISTERED_MODEL_NAME}@production → v{version_for_alias}")


if __name__ == "__main__":
    main()
