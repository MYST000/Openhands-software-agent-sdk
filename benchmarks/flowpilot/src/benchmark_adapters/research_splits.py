"""Validate a frozen LiveCodeBench research manifest before assigning task splits.

The legacy v1 JSONL identifies its benchmark through question IDs and group prefixes.
Its dataset revision is therefore bound to the adapter's pinned release_v6 lock.
Manifest content is controller metadata; only validated split/group IDs are projected
into an existing task. Public instructions, files, and private evaluation data remain
those produced by the dataset adapter.
"""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .code_tasks import CodeBundle

RESEARCH_SPLITS = frozenset({"fit", "tune", "calibration", "test", "historical_dev"})
_DATASET_ID = "livecodebench/code_generation_lite"
_METADATA_FIELDS = ("platform", "difficulty", "contest_date", "statement_sha256")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{field} requires an explicit SHA256 hexadecimal digest")
    return value.lower()


def _identifier(value: Any, field: str, pattern: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(pattern, value) or ".." in value:
        raise ValueError(f"Invalid manifest {field}: expected a safe identifier")
    return value


def _split(value: Any) -> str:
    if not isinstance(value, str) or value not in RESEARCH_SPLITS:
        raise ValueError(f"Invalid research split (research_split): {value!r}")
    return value


def _claim(mapping: dict, key: str, value: str, description: str) -> None:
    if key in mapping and mapping[key] != value:
        raise ValueError(f"Manifest {description} conflict: {key}")
    mapping[key] = value


def _read_manifest(content: bytes, revision: str) -> dict[str, dict[str, Any]]:
    rows = {}
    group_splits = {}
    statement_splits = {}
    contest_groups = {}
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("Research manifest must be UTF-8 JSONL") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid research manifest JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Research manifest line {line_number} must be an object")
        task_id = _identifier(
            row.get("question_id"), "question_id", r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
        if task_id in rows:
            raise ValueError(f"Duplicate manifest task ID: {task_id}")
        group = _identifier(
            row.get("task_group_id"),
            "task_group_id",
            r"livecodebench/[A-Za-z0-9][A-Za-z0-9_-]{0,31}:[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
        )
        split = _split(row.get("research_split"))
        historical = row.get("historically_selected")
        if type(historical) is not bool:
            raise ValueError("Manifest historically_selected must be an explicit boolean")
        if historical and split != "historical_dev":
            raise ValueError(f"Manifest historical exposure requires historical_dev: {task_id}")
        for field in ("platform", "difficulty", "contest_date"):
            value = row.get(field)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"Invalid manifest {field}: expected a nonempty string")
        try:
            datetime.fromisoformat(row["contest_date"])
        except ValueError as exc:
            raise ValueError("Invalid manifest contest_date: expected an ISO date") from exc
        fingerprint = _sha256(row.get("statement_sha256"), "manifest statement_sha256")
        for field, expected in (
            ("benchmark", "livecodebench"),
            ("dataset_id", _DATASET_ID),
            ("revision", revision),
            ("dataset_revision", revision),
        ):
            if field in row and row[field] != expected:
                raise ValueError(f"Incompatible manifest {field}: expected {expected}")
        _claim(group_splits, group, split, "task group crosses research split")
        _claim(statement_splits, fingerprint, split, "exact statement crosses research split")
        # A renamed group must not let siblings from one contest bypass isolation.
        contest = row["platform"] + ":" + task_id.rsplit("_", 1)[0]
        _claim(contest_groups, contest, group, "contest assigned to different task groups")
        rows[task_id] = {**row, "statement_sha256": fingerprint}
    if not rows:
        raise ValueError("Research manifest must be nonempty")
    return rows


def apply_research_split(
    bundles: Sequence[CodeBundle],
    manifest_path: str | Path,
    expected_sha256: str,
    research_split: str,
) -> tuple[list[CodeBundle], dict[str, Any]]:
    """Return newly assigned bundles and provenance, or fail before any assignment.

    Validation covers every manifest row, including unselected tasks, to detect
    holdout leakage outside the current batch. This v1 policy supports only the
    pinned LiveCodeBench release; QuixBugs retains its development-only policy.
    """
    research_split = _split(research_split)
    expected_sha256 = _sha256(expected_sha256, "Research manifest SHA256")
    selected_ids = [bundle.task.task_id for bundle in bundles]
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Research selection must contain nonempty, unique task IDs")
    path = Path(manifest_path).resolve()
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Research manifest SHA256 mismatch")
    lock = json.loads((Path(__file__).parent / "data/lcb_release_v6.lock.json").read_text())
    revision = lock["revision"]
    rows = _read_manifest(content, revision)
    assigned = []
    for bundle in bundles:
        task = bundle.task
        if bundle.kind != "livecodebench":
            raise ValueError("Research manifest policy currently supports LiveCodeBench only")
        if task.dataset_id != _DATASET_ID or task.revision != revision:
            raise ValueError("Research manifest requires the pinned LiveCodeBench dataset/revision")
        row = rows.get(task.task_id)
        if row is None:
            raise ValueError(f"Selected task is not in research manifest: {task.task_id}")
        if row["research_split"] != research_split:
            raise ValueError(
                f"Task {task.task_id} belongs to {row['research_split']}, not {research_split}"
            )
        for field in _METADATA_FIELDS:
            if task.public_metadata.get(field) != row[field]:
                raise ValueError(f"Research manifest {field} differs from task {task.task_id}")
        metadata = dict(task.public_metadata)
        metadata.pop("pilot_split", None)
        metadata.update(
            research_split=research_split,
            task_group_id=row["task_group_id"],
            upstream_split="test",
        )
        assigned.append(
            replace(bundle, task=replace(task, split=research_split, public_metadata=metadata))
        )
    provenance = {
        "manifest_path": str(path),
        "manifest_sha256": expected_sha256,
        "manifest_rows": len(rows),
        "research_split": research_split,
        "dataset_id": _DATASET_ID,
        "dataset_revision": revision,
        "selected_task_ids": selected_ids,
        "task_group_ids": [bundle.task.public_metadata["task_group_id"] for bundle in assigned],
    }
    return assigned, provenance
