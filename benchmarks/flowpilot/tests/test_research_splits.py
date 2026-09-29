import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_adapters.code_tasks import CodeBundle
from benchmark_adapters.contracts import Task

REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
DATASET_ID = "livecodebench/code_generation_lite"


def manifest_row(task_id="abc400_a", split="fit", **updates):
    return {
        "question_id": task_id,
        "platform": "atcoder",
        "difficulty": "easy",
        "contest_date": "2025-04-05T00:00:00",
        "statement_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
        "task_group_id": "livecodebench/atcoder:" + task_id.rsplit("_", 1)[0],
        "research_split": split,
        "historically_selected": False,
        **updates,
    }


def bundle_for(row):
    return CodeBundle(
        kind="livecodebench",
        task=Task(
            DATASET_ID,
            REVISION,
            "dev",
            row["question_id"],
            "ORIGINAL PUBLIC PROBLEM",
            {
                "protocol": "lcb-release-v6-stdin-python-agentic-v1",
                "upstream_split": "test",
                "pilot_split": "development-only",
                **{
                    key: row[key]
                    for key in ("platform", "difficulty", "contest_date", "statement_sha256")
                },
            },
        ),
        public_files={"solution.py": "# implement\n"},
        solution_path="solution.py",
        private={"hidden_cases": "CONTROLLER_ONLY"},
    )


def write_manifest(tmp_path, rows):
    path = tmp_path / "research.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def apply(bundles, path, digest, split="fit"):
    from benchmark_adapters.research_splits import apply_research_split

    return apply_research_split(bundles, path, digest, split)


def test_assigns_frozen_split_without_importing_manifest_content_into_actor(tmp_path):
    row = manifest_row(
        instruction="INJECTED_INSTRUCTION",
        hidden_tests="INJECTED_PRIVATE_DATA",
        public_metadata={"solution_path": "INJECTED_PATH"},
    )
    original = bundle_for(row)
    path, digest = write_manifest(tmp_path, [row])

    assigned, provenance = apply([original], path, digest)

    assert original.task.split == "dev"
    assert "research_split" not in original.task.public_metadata
    assert assigned[0] is not original
    task = assigned[0].task
    assert task.split == "fit"
    assert task.public_metadata["research_split"] == "fit"
    assert task.public_metadata["task_group_id"] == row["task_group_id"]
    assert task.public_metadata["upstream_split"] == "test"
    assert "pilot_split" not in task.public_metadata
    assert task.instruction == original.task.instruction
    assert assigned[0].public_files == original.public_files
    assert assigned[0].private == original.private
    assert "INJECTED" not in json.dumps(task.to_dict())
    assert "CONTROLLER_ONLY" not in json.dumps(task.to_dict())
    assert provenance["manifest_sha256"] == digest
    assert provenance["manifest_path"] == str(path.resolve())
    assert provenance["research_split"] == "fit"
    assert provenance["dataset_id"] == DATASET_ID
    assert provenance["dataset_revision"] == REVISION
    assert provenance["selected_task_ids"] == [row["question_id"]]
    assert provenance["task_group_ids"] == [row["task_group_id"]]


@pytest.mark.parametrize("split", ["fit", "tune", "calibration", "test", "historical_dev"])
def test_accepts_each_research_split_and_preserves_explicit_selection_order(tmp_path, split):
    rows = [manifest_row("abc400_a", split), manifest_row("abc401_a", split)]
    path, digest = write_manifest(tmp_path, rows)
    bundles = [bundle_for(row) for row in reversed(rows)]
    assigned, provenance = apply(bundles, path, digest, split)
    assert [bundle.task.task_id for bundle in assigned] == ["abc401_a", "abc400_a"]
    assert all(bundle.task.split == split for bundle in assigned)
    assert provenance["selected_task_ids"] == ["abc401_a", "abc400_a"]


@pytest.mark.parametrize("digest", [None, "", "z" * 64, "a" * 63, "0" * 64])
def test_requires_explicit_matching_manifest_sha256(tmp_path, digest):
    row = manifest_row()
    path, _ = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="SHA256"):
        apply([bundle_for(row)], path, digest)


@pytest.mark.parametrize("split", ["dev", "validation", "", None, ["fit"]])
def test_rejects_unknown_research_split(tmp_path, split):
    row = manifest_row()
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="research split"):
        apply([bundle_for(row)], path, digest, split)


def test_rejects_task_not_in_manifest_or_in_different_split(tmp_path):
    row = manifest_row()
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="not in.*manifest"):
        apply([bundle_for(manifest_row("abc401_a"))], path, digest)
    with pytest.raises(ValueError, match="belongs to.*fit"):
        apply([bundle_for(row)], path, digest, "test")


def test_rejects_duplicate_or_empty_selection(tmp_path):
    row = manifest_row()
    path, digest = write_manifest(tmp_path, [row])
    bundle = bundle_for(row)
    with pytest.raises(ValueError, match="unique"):
        apply([bundle, bundle], path, digest)
    with pytest.raises(ValueError, match="nonempty"):
        apply([], path, digest)


@pytest.mark.parametrize("content", ["", "\n", "[]\n", "null\n", "{invalid}\n"])
def test_rejects_empty_or_malformed_manifest(tmp_path, content):
    path = tmp_path / "research.jsonl"
    path.write_text(content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="manifest"):
        apply([bundle_for(manifest_row())], path, digest)


def test_rejects_duplicate_ids_in_whole_manifest_including_unselected_rows(tmp_path):
    selected = manifest_row()
    other = manifest_row("abc401_a")
    path, digest = write_manifest(tmp_path, [selected, other, other])
    with pytest.raises(ValueError, match="Duplicate.*ID"):
        apply([bundle_for(selected)], path, digest)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"task_group_id": "livecodebench/atcoder:abc400"}, "group.*split"),
        ({"statement_sha256": manifest_row()["statement_sha256"]}, "statement.*split"),
    ],
)
def test_rejects_cross_split_group_or_statement_anywhere_in_manifest(tmp_path, updates, match):
    row = manifest_row()
    conflicting = manifest_row("abc401_a", "test", **updates)
    path, digest = write_manifest(tmp_path, [row, conflicting])
    with pytest.raises(ValueError, match=match):
        apply([bundle_for(row)], path, digest)


def test_contest_siblings_cannot_evade_split_isolation_by_renaming_group(tmp_path):
    row = manifest_row()
    sibling = manifest_row("abc400_b", "test", task_group_id="livecodebench/atcoder:fake")
    path, digest = write_manifest(tmp_path, [row, sibling])
    with pytest.raises(ValueError, match="contest.*group|contest.*split"):
        apply([bundle_for(row)], path, digest)


def test_historical_exposure_forces_entire_group_into_historical_dev(tmp_path):
    historical = manifest_row("abc400_a", "historical_dev", historically_selected=True)
    sibling = manifest_row("abc400_b", "historical_dev")
    path, digest = write_manifest(tmp_path, [historical, sibling])
    assigned, _ = apply([bundle_for(sibling)], path, digest, "historical_dev")
    assert assigned[0].task.split == "historical_dev"

    for split in ("fit", "tune", "calibration", "test"):
        exposed = dict(historical, research_split=split)
        path, digest = write_manifest(tmp_path, [exposed])
        with pytest.raises(ValueError, match="historical"):
            apply([bundle_for(exposed)], path, digest, split)


@pytest.mark.parametrize(
    "field,value",
    [
        ("question_id", "../../escape"),
        ("question_id", 123),
        ("task_group_id", "quixbugs/abc400"),
        ("task_group_id", "livecodebench/abc400"),
        ("task_group_id", "livecodebench/../escape"),
        ("task_group_id", "livecodebench/atcoder:abc400\nINJECT"),
        ("research_split", "dev"),
        ("historically_selected", "false"),
        ("statement_sha256", "bad-hash"),
        ("platform", []),
        ("difficulty", ""),
        ("contest_date", "not-a-date"),
    ],
)
def test_rejects_invalid_schema_even_for_unselected_rows(tmp_path, field, value):
    row = manifest_row()
    invalid = manifest_row("abc401_a", **{field: value})
    path, digest = write_manifest(tmp_path, [row, invalid])
    with pytest.raises(ValueError, match=field):
        apply([bundle_for(row)], path, digest)


@pytest.mark.parametrize(
    "field", ["task_group_id", "historically_selected", "statement_sha256", "platform"]
)
def test_rejects_missing_required_manifest_fields(tmp_path, field):
    row = manifest_row()
    incomplete = dict(row)
    incomplete.pop(field)
    path, digest = write_manifest(tmp_path, [incomplete])
    with pytest.raises(ValueError, match=field):
        apply([bundle_for(row)], path, digest)


@pytest.mark.parametrize("field", ["platform", "difficulty", "contest_date", "statement_sha256"])
def test_matches_manifest_metadata_to_selected_original_task(tmp_path, field):
    row = manifest_row()
    bundle = bundle_for(row)
    bundle.task.public_metadata[field] = "CHANGED_ORIGINAL"
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match=field):
        apply([bundle], path, digest)


@pytest.mark.parametrize(
    "field,value",
    [("dataset_id", "other/dataset"), ("revision", "unfrozen")],
)
def test_rejects_bundle_from_wrong_dataset_or_revision(tmp_path, field, value):
    row = manifest_row()
    bundle = bundle_for(row)
    bundle.task = replace(bundle.task, **{field: value})
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="dataset|revision"):
        apply([bundle], path, digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("benchmark", "quixbugs"),
        ("dataset_id", "other/dataset"),
        ("revision", "unfrozen"),
        ("dataset_revision", "unfrozen"),
    ],
)
def test_rejects_incompatible_explicit_manifest_benchmark_or_revision(tmp_path, field, value):
    row = manifest_row(**{field: value})
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match=field):
        apply([bundle_for(row)], path, digest)


def test_rejects_quixbugs_until_it_has_a_separate_research_policy(tmp_path):
    row = manifest_row()
    bundle = bundle_for(row)
    bundle.kind = "quixbugs"
    path, digest = write_manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="LiveCodeBench"):
        apply([bundle], path, digest)


def test_bundled_lock_matches_test_revision():
    import benchmark_adapters.code_tasks as module

    lock = Path(module.__file__).parent / "data/lcb_release_v6.lock.json"
    assert json.loads(lock.read_text())["revision"] == REVISION


def test_accepts_duplicate_statements_merged_into_another_contest_group(tmp_path):
    original = manifest_row()
    duplicate = manifest_row(
        "abc401_a",
        task_group_id=original["task_group_id"],
        statement_sha256=original["statement_sha256"],
    )
    path, digest = write_manifest(tmp_path, [original, duplicate])
    assigned, _ = apply([bundle_for(original), bundle_for(duplicate)], path, digest)
    assert [bundle.task.public_metadata["task_group_id"] for bundle in assigned] == [
        original["task_group_id"],
        original["task_group_id"],
    ]
