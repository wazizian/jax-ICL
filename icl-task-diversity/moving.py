from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import shlex

def print_rsync_batch(
    source_dirs: Iterable[str | Path],
    target_root: str | Path,
    *,
    move: bool = False,   # if True, add --remove-source-files + cleanup of empty dirs
) -> str:
    target = Path(target_root)
    dirs = [Path(d) for d in source_dirs]

    rsync_base = 'rsync -a -vv --progress --info=progress2 --stats --human-readable'
    if move:
        rsync_base += ' --remove-source-files'

    header = [
        "set -euo pipefail",
        f'TARGET={shlex.quote(str(target))}',
        'mkdir -p "$TARGET"',
        f'RSYNC="{rsync_base}"',
    ]

    lines = []
    for d in dirs:
        src_q = shlex.quote(str(d))  # IMPORTANT: no trailing slash -> creates $TARGET/<basename>
        lines.append(f'$RSYNC -- {src_q} "$TARGET/"')
        if move:
            # remove any now-empty directories left behind
            # (no-op if something excluded remains)
            lines.append(f'find {src_q} -type d -empty -delete || true')

    script = "bash -lc '" + "\\n".join(header + lines) + "'"
    print(script)
    return script


if __name__ == '__main__':
    rsync_move_dirs(
        ["/data/projectA", "/data/projectB"],
        "/backup/2025-10-09",
        overwrite=False,   # set True to merge into existing destinations
    )

