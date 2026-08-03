"""Container entrypoint: seed data/artifacts volumes, then start the scheduler.

CapRover persistent directories can mount empty volumes over COPY'd paths.
Match data is filled in only when missing (volumes keep live refreshes).
Production artifacts are always synced from the image so model/MV updates
take effect on every redeploy without manually wiping the volume.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# When launched as `python scripts/docker_entrypoint.py`, sys.path[0] is
# scripts/ — put the app root first so `run_scheduler` / `models` resolve.
APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

DATA = APP / "data"
ARTIFACTS = APP / "artifacts"
DEFAULT_DATA = APP / ".image_data"
DEFAULT_ARTIFACTS = APP / ".image_artifacts"


def _is_empty_dir(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        next(path.iterdir())
        return False
    except StopIteration:
        return True


def _seed_missing(src: Path, dst: Path) -> None:
    """Copy image defaults into dst only where the target path is absent."""
    if not src.exists():
        print(f"[entrypoint] No image defaults at {src}, skipping seed for {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    if not _is_empty_dir(dst):
        print(f"[entrypoint] Seeding missing defaults into non-empty {dst} ...")
    else:
        print(f"[entrypoint] Seeding {dst} from {src} ...")
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"[entrypoint] Seed complete for {dst}")


def _sync_tree(src: Path, dst: Path) -> None:
    """Overwrite dst with src contents (files + directories)."""
    if not src.exists():
        print(f"[entrypoint] No image defaults at {src}, skipping sync for {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[entrypoint] Syncing {dst} from {src} (overwrite) ...")
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"[entrypoint] Sync complete for {dst}")


def main() -> None:
    print(f"[entrypoint] APP={APP}")
    _seed_missing(DEFAULT_DATA, DATA)
    _sync_tree(DEFAULT_ARTIFACTS, ARTIFACTS)

    # Replace this process with the scheduler so signals/PID stay clean.
    scheduler = APP / "run_scheduler.py"
    if not scheduler.exists():
        raise FileNotFoundError(f"Missing scheduler entrypoint: {scheduler}")
    print(f"[entrypoint] Starting {scheduler}")
    os.execv(sys.executable, [sys.executable, str(scheduler)])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[entrypoint] Fatal: {e}", file=sys.stderr)
        raise
