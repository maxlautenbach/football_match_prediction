"""Container entrypoint: seed data/artifacts volumes, then start the scheduler.

CapRover persistent directories can mount empty volumes over COPY'd paths.
If the mount is empty, we restore the image defaults baked at build time.
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


def _seed(src: Path, dst: Path, marker: str) -> None:
    if not src.exists():
        print(f"[entrypoint] No image defaults at {src}, skipping seed for {dst}")
        return
    dst.mkdir(parents=True, exist_ok=True)
    marker_path = dst / marker
    if marker_path.exists():
        return
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


def main() -> None:
    print(f"[entrypoint] APP={APP}")
    _seed(DEFAULT_DATA, DATA, "match_df_2026.pck")
    _seed(DEFAULT_ARTIFACTS, ARTIFACTS, "bundle.json")

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
