"""Check runtime contents while allowing benchmark-only repository changes."""

import importlib.metadata
import json
import subprocess
from pathlib import Path

CORE_PATHS = (
    "openhands-sdk",
    "openhands-tools",
    "openhands-workspace",
    "openhands-agent-server",
    "pyproject.toml",
    "uv.lock",
)


def sdk_source_state(repo, expected_commit):
    repo = Path(repo).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    head = git("rev-parse", "HEAD")
    git("rev-parse", "--verify", expected_commit + "^{commit}")
    committed = git("diff", "--name-only", expected_commit, "HEAD", "--", *CORE_PATHS)
    working = git("diff", "--name-only", "HEAD", "--", *CORE_PATHS)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *CORE_PATHS)
    if committed or working or untracked:
        raise ValueError(
            "SDK runtime differs from pinned baseline: "
            + "\n".join(part for part in (committed, working, untracked) if part)
        )
    return {
        "checkout_commit": head,
        "runtime_base_commit": expected_commit,
        "runtime_tree_entries": git("ls-tree", expected_commit, "--", *CORE_PATHS).splitlines(),
        "runtime_matches_baseline": True,
    }


def check_sdk(runtime):
    state = sdk_source_state(runtime.sdk_path, runtime.sdk_commit)
    dist = importlib.metadata.distribution("openhands-sdk")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    expected = (Path(runtime.sdk_path) / "openhands-sdk").resolve().as_uri()
    if direct.get("url") != expected or not direct.get("dir_info", {}).get("editable"):
        raise ValueError("Installed SDK must be editable from the configured checkout")
    return {**state, "package_version": dist.version, "editable_url": expected}
