"""Keep controller source data and feedback outside local task UID access."""

import os
import subprocess
from pathlib import Path

TASK_UIDS = (63111, 63112)


def private_controller_directory(path, *, create=False):
    path = Path(path).absolute()
    if os.geteuid() != 0:
        raise ValueError("Local collection requires root to run separate low-privilege task UIDs")
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Controller directory does not exist: {path}")
    for uid in TASK_UIDS:
        result = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-c",
                "import os,sys; sys.exit(0 if os.access(sys.argv[1], os.X_OK) else 1)",
                str(path),
            ],
            user=uid,
            group=uid,
            extra_groups=[],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            raise ValueError(
                f"Controller directory accessible to task UID {uid}: {path}. "
                "Place source data and output under a root-owned private directory (mode0700). "
                "Only public task files may be copied to the actor workspace."
            )
        if result.returncode != 1:
            raise ValueError("Cannot establish controller path privacy: " + result.stderr.decode())
    return path
