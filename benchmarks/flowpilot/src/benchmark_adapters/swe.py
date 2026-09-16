import hashlib
import json
import re
import shlex
from pathlib import Path

from .contracts import Task

PUBLIC_COLUMNS = ["instance_id", "repo", "base_commit", "problem_statement"]


def checked_dataset_digest(config):
    with Path(config.dataset.path).open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if config.dataset.sha256 and digest != config.dataset.sha256:
        raise ValueError("Dataset checksum mismatch")
    return digest


def select_records(records, id_key, ids=None, limit=0):
    lookup = {}
    for row in records:
        key = str(row[id_key])
        if key in lookup:
            raise ValueError(f"Duplicate task ID: {key}")
        lookup[key] = row
    if ids:
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate selected task IDs")
        if missing := set(ids) - lookup.keys():
            raise ValueError(f"Selected task IDs missing from dataset: {sorted(missing)}")
        selected = [lookup[key] for key in ids]
    else:
        selected = list(lookup.values())
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    return selected[:limit] if limit else selected


def load_records(path, columns=None):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path, columns=columns).to_pylist()
    if path.suffix == ".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain an array")
    return data


class SWEAdapter:
    kind = "swe"

    @staticmethod
    def from_record(row, *, dataset_id, revision, split):
        for field in PUBLIC_COLUMNS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or invalid SWE field: {field}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError("base_commit must be a full 40-character commit SHA")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repo"]):
            raise ValueError("Invalid repository name")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+", row["instance_id"]):
            raise ValueError("Invalid SWE instance ID")
        instruction = (
            "Fix the issue below in the repository provided by your workspace tools. "
            "Inspect the code, make the necessary changes, and run relevant tests. "
            "Your actual repository changes will be collected as the submission patch. "
            "Do not modify tests merely to hide a failure. Finish with a concise summary.\n\n"
            f"Repository: {row['repo']}\nBase commit: {row['base_commit']}\n\n"
            f"Issue:\n{row['problem_statement']}"
        )
        return Task(
            dataset_id,
            revision,
            split,
            row["instance_id"],
            instruction,
            {k: row[k] for k in ("repo", "base_commit")},
        )

    @classmethod
    def load(cls, path, *, dataset_id, revision, split, ids=None, limit=0):
        rows = select_records(load_records(path, PUBLIC_COLUMNS), "instance_id", ids, limit)
        return [
            cls.from_record(r, dataset_id=dataset_id, revision=revision, split=split) for r in rows
        ]

    @staticmethod
    def submission(task, patch, model):
        return dict(instance_id=task.task_id, model_name_or_path=model, model_patch=patch)


def export_patch_command(base_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", base_commit):
        raise ValueError("Invalid base commit")
    return (
        "set -e\n"
        "adapter_index=$(mktemp)\n"
        'trap \'rm -f "$adapter_index" "$adapter_index.lock"\' EXIT\n'
        # Preserve index mtime so Git's racy-clean detection still examines same-size edits.
        'cp -p -- "$(git rev-parse --git-path index)" "$adapter_index"\n'
        'export GIT_INDEX_FILE="$adapter_index"\n'
        "git add -A -- .\n"
        f"git --no-pager diff --cached --binary --no-ext-diff --no-color {shlex.quote(base_commit)} -- .\n"
    )


def image_name(task, template):
    return template.format(
        instance_id=task.task_id, image_id=task.task_id.replace("__", "_1776_").lower()
    )
