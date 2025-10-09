from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable
from tqdm import tqdm


def rsync_move_dirs(
    source_dirs: Iterable[str | Path],
    target_root: str | Path,
    *,
    overwrite: bool = False,   # if False, refuse when target/<name> exists
) -> None:
    """
    Move each directory in `source_dirs` into `target_root` using one `rsync` per dir.

    - Uses: rsync -a --remove-source-files src_dir/ target_root/src_dir.name/
    - If overwrite=False, raises if the destination directory already exists.
    - After rsync, removes empty source directories (since --remove-source-files leaves dirs).

    Requires `rsync` on PATH.
    """
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    src_dirs = [Path(d) for d in source_dirs]
    for s in src_dirs:
        if not s.exists():
            raise FileNotFoundError(f"Source does not exist: {s}")
        if not s.is_dir():
            raise ValueError(f"Not a directory: {s}")

    for s in tqdm(src_dirs, desc="Moving directories (rsync)", unit="dir"):
        dst = target_root / s.name
        if not overwrite and dst.exists():
            raise FileExistsError(f"Destination already exists: {dst}")

        dst.mkdir(parents=True, exist_ok=True)

        # One rsync per directory; trailing slashes copy contents into the named folder.
        cmd = ["rsync", "-a", "--remove-source-files", "--", str(s) + "/", str(dst) + "/"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError("rsync not found. Install it and ensure it's on your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"rsync failed for {s}:\n{e.stderr.decode(errors='ignore')}") from e

        # Clean up now-empty directories under s (rsync leaves empty dirs).
        # Walk bottom-up so children are removed before parents.
        for root, dirs, files in os.walk(s, topdown=False):
            if not dirs and not files:
                try:
                    Path(root).rmdir()
                except OSError:
                    pass

        # Finally, try to remove the source dir itself (if empty).
        try:
            s.rmdir()
        except OSError:
            # Not empty (e.g., excluded files) — leave it.
            pass


if __name__ == '__main__':
    rsync_move_dirs(
        ["/data/projectA", "/data/projectB"],
        "/backup/2025-10-09",
        overwrite=False,   # set True to merge into existing destinations
    )

