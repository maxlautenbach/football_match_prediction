"""Promote an MLflow model (run or registry URI) into artifacts/ for production."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
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
from model import Model
from models.contract import read_bundle_json

ARTIFACTS_DIR = BASE_DIR / "artifacts"


def _find_bundle(root: Path) -> Path:
    """Locate directory containing bundle.json (or legacy meta.json) inside a download."""
    if (root / "bundle.json").exists() or (root / "meta.json").exists():
        return root
    # Prefer bundle.json over legacy meta.json
    for name in ("bundle.json", "meta.json"):
        candidates = list(root.rglob(name))
        if candidates:
            return candidates[0].parent
    raise FileNotFoundError(f"No bundle.json or meta.json under {root}")


def _validate_bundle(bundle_dir: Path) -> None:
    """Validate bundle metadata and that Model can load + smoke-predict."""
    if (bundle_dir / "bundle.json").exists():
        meta = read_bundle_json(bundle_dir)
        print(
            f"bundle.json: model_type={meta['model_type']} "
            f"recipe={meta.get('recipe_name')} schema={meta.get('schema_version')}"
        )
        for fname in meta.get("files") or []:
            if not (bundle_dir / fname).exists():
                raise FileNotFoundError(f"bundle.json lists missing file: {fname}")
    else:
        print("WARNING: no bundle.json — accepting legacy meta.json bundle")

    model = Model(artifacts_dir=bundle_dir)
    import pandas as pd

    smoke = pd.DataFrame(
        {
            "Team Home": ["FC Bayern München"],
            "Team Away": ["Borussia Dortmund"],
            "Saison": [2025],
            "Spieltag": [1],
            "Wochentag": ["Saturday"],
        }
    )
    preds = model.predict(smoke)
    print(f"Smoke prediction OK: {preds[0]}")


def _atomic_replace(src: Path, dst: Path) -> None:
    """Replace dst with src contents via temp dir + rename."""
    parent = dst.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".artifacts_staging_", dir=str(parent)))
    staging_bundle = staging / "bundle"
    shutil.copytree(src, staging_bundle)

    backup_existing = None
    if dst.exists():
        backup_existing = parent / f".{dst.name}_swap_old"
        if backup_existing.exists():
            shutil.rmtree(backup_existing)
        dst.rename(backup_existing)

    try:
        staging_bundle.rename(dst)
    except Exception:
        if backup_existing is not None and backup_existing.exists() and not dst.exists():
            backup_existing.rename(dst)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup_existing is not None and backup_existing.exists():
            shutil.rmtree(backup_existing, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Promote MLflow model into artifacts/ and optionally set registry alias"
    )
    parser.add_argument("--run-id", default=None, help="MLflow run id (legacy path)")
    parser.add_argument(
        "--model-uri",
        default=None,
        help=f"e.g. models:/{REGISTERED_MODEL_NAME}@candidate or models:/name/1",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help=f"Promote registry model '{REGISTERED_MODEL_NAME}' by alias (e.g. candidate)",
    )
    parser.add_argument(
        "--registered-model",
        default=REGISTERED_MODEL_NAME,
        help="Registered model name when using --alias / --set-production-alias",
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
    registered_name = args.registered_model

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
    try:
        if args.alias:
            model_uri = f"models:/{registered_name}@{args.alias}"
            print(f"Downloading {model_uri} ...")
            local = download_model_uri(model_uri, tmp)
            client = MlflowClient()
            mv = client.get_model_version_by_alias(registered_name, args.alias)
            version_for_alias = mv.version
        elif args.model_uri:
            print(f"Downloading {args.model_uri} ...")
            local = download_model_uri(args.model_uri, tmp)
            if args.model_uri.startswith("models:/"):
                parts = args.model_uri.replace("models:/", "").split("/")
                if len(parts) == 2 and parts[1].isdigit():
                    version_for_alias = parts[1]
                    # Prefer name from URI when setting production alias
                    registered_name = parts[0].split("@")[0]
        else:
            print(f"Downloading run {args.run_id} artifacts ...")
            local = download_run_artifacts(args.run_id, tmp, artifact_path="model")

        source = _find_bundle(local if local.exists() else tmp)
        print(f"Validating bundle at {source} ...")
        _validate_bundle(source)

        print(f"Promoting → {dst}")
        _atomic_replace(source, dst)
        print(f"Promoted → {dst}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if args.set_production_alias:
        if version_for_alias is None:
            raise SystemExit("--set-production-alias requires --alias or models:/name/version URI")
        set_model_alias(registered_name, "production", version_for_alias)
        print(f"Set alias {registered_name}@production → v{version_for_alias}")


if __name__ == "__main__":
    main()
