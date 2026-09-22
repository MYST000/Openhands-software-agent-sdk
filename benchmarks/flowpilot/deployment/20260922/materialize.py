"""Restore versioned deployment templates into a new project root, without downloads."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    # Root enters Python/TOML string literals in the existing configuration templates.
    if any(char in str(root) for char in ('"', "\\", "\n", "\r")):
        parser.error("Project path must not contain quotes, backslashes or newlines")
    source = Path(__file__).resolve().parent
    files = []
    for line in (source / "SNAPSHOT_SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe snapshot path")
        raw = (source / "snapshot" / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Snapshot checksum mismatch: {name}")
        target = root / name
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Destination escapes project root: {name}")
        if name.startswith("configs/c4/"):
            raw = raw.replace(b"/path/to/flowpilot_predictor", str(root).encode())
        if target.is_symlink() or (target.exists() and target.read_bytes() != raw):
            raise ValueError(f"Refusing to overwrite different existing file: {target}")
        files.append((target, raw))
    # Verify the whole plan before creating files. Existing identical files allow resume.
    root.mkdir(parents=True, exist_ok=True)
    for target, raw in files:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
    for name in ("downloads", "logs", "data", "models", "evidence/preparation"):
        (root / name).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"root": str(root), "verified_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
