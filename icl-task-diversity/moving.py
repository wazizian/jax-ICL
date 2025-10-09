from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Dict, List, Tuple
from tqdm import tqdm


def copy_and_verify_files(
    paths: Iterable[str | Path],
    target_dir: str | Path,
) -> Dict[Path, Path]:
    """
    Copy files to `target_dir` with a tqdm progress bar and verify by SHA-256.

    - No filename auto-suffixing: raises if a destination path already exists.
    - Uses shutil.copy2 to preserve basic metadata.

    Returns
    -------
    dict[Path, Path]
        Mapping from source path -> destination path.

    Raises
    ------
    FileNotFoundError, ValueError, FileExistsError, RuntimeError
    """
    # Normalize & validate sources
    src_files: List[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Source does not exist: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")
        src_files.append(p)

    # Prepare destination dir
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # Plan destinations and ensure no collisions/overwrites
    mapping: Dict[Path, Path] = {}
    for src in src_files:
        dst = target / src.name
        if dst.exists():
            raise FileExistsError(f"Destination already exists, refusing to overwrite: {dst}")
        mapping[src] = dst

    # Copy with progress bar
    for src in tqdm(src_files, desc="Copying files", unit="file"):
        shutil.copy2(src, mapping[src])

    # Helper: SHA-256
    def sha256(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()

    # Verify by existence, size, then SHA-256
    failures: List[Tuple[Path, str]] = []
    for src, dst in tqdm(mapping.items(), desc="Verifying copies", unit="file"):
        if not dst.exists():
            failures.append((src, "destination_missing"))
            continue
        if src.stat().st_size != dst.stat().st_size:
            failures.append((src, "size_mismatch"))
            continue
        if sha256(src) != sha256(dst):
            failures.append((src, "hash_mismatch"))
            continue

    if failures:
        details = "\n".join(f"- {s} -> {mapping.get(s, 'N/A')} [{reason}]" for s, reason in failures)
        raise RuntimeError(f"Verification failed for {len(failures)} file(s):\n{details}")

    return mapping

if __name__ == '__main__':
    mapping = copy_and_verify_files(
        ["/data/a.csv", "/data/b.csv", "/reports/summary.pdf"],
        "/backup/2025-10-09",
    )
    print("Copied:")
    for s, d in mapping.items():
        print(f"{s} -> {d}")

