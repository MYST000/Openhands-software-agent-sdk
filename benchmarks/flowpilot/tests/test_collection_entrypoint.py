import json
import os
import subprocess

import pytest

from benchmark_adapters.config import Config, DatasetConfig, RuntimeConfig


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def checkout(tmp_path):
    repo = tmp_path / "checkout"
    (repo / "openhands-sdk").mkdir(parents=True)
    (repo / "openhands-sdk/core.py").write_text("value = 1\n")
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "SDK base")
    return repo, git(repo, "rev-parse", "HEAD")


def test_adapter_only_commits_allowed_but_runtime_changes_rejected(checkout):
    from benchmark_adapters.sdk_provenance import sdk_source_state

    repo, base = checkout
    (repo / "benchmarks").mkdir()
    (repo / "benchmarks/adapter.py").write_text("value = 2\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Add adapter")
    state = sdk_source_state(repo, base)
    assert state["checkout_commit"] != base
    assert state["runtime_base_commit"] == base
    (repo / "openhands-sdk/core.py").write_text("value = 3\n")
    with pytest.raises(ValueError, match="runtime"):
        sdk_source_state(repo, base)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Change runtime")
    with pytest.raises(ValueError, match="runtime"):
        sdk_source_state(repo, base)


def test_untracked_runtime_module_is_rejected(checkout):
    from benchmark_adapters.sdk_provenance import sdk_source_state

    repo, base = checkout
    (repo / "openhands-sdk/shadow.py").write_text("pass\n")
    with pytest.raises(ValueError, match="runtime"):
        sdk_source_state(repo, base)


@pytest.mark.skipif(os.geteuid() != 0, reason="Local collector must probe separate task UIDs")
def test_explicit_quix_selection_restores_protocol_and_freezes_input(tmp_path):
    from benchmark_adapters.code_collection import prepare_collection

    repo = tmp_path / "quix"
    for dirname in ("python_programs", "python_testcases", "correct_python_programs"):
        (repo / dirname).mkdir(parents=True)
    for name in ("a", "b"):
        (repo / f"python_programs/{name}.py").write_text("def solve(): return 0\n")
        (repo / f"correct_python_programs/{name}.py").write_text("REFERENCE_SECRET")
        (repo / f"python_testcases/test_{name}.py").write_text("PUBLIC_TEST")
    (repo / "conftest.py").write_text("")
    (repo / "python_testcases/load_testdata.py").write_text("")
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Dataset")
    cfg = Config(
        dataset=DatasetConfig(
            kind="quixbugs",
            id="jkoppel/QuixBugs",
            path=str(repo),
            revision=git(repo, "rev-parse", "HEAD"),
            setting="openhands-python-code-v1",
        ),
        runtime=RuntimeConfig(runs_dir=str(tmp_path / "runs")),
    )
    bundles, root = prepare_collection(cfg, ["b"], "/usr/bin", "pilot")
    assert [b.task.task_id for b in bundles] == ["b"]
    text = (root / "selection.json").read_text()
    assert "REFERENCE_SECRET" not in text
    assert bundles[0].public_files["python_testcases/test_b.py"] == "PUBLIC_TEST"
    prepare_collection(cfg, ["b"], "/usr/bin", "pilot")
    with pytest.raises(ValueError, match="Frozen selection"):
        prepare_collection(cfg, ["a"], "/usr/bin", "pilot")
    for bad_ids in ([], ["a", "a"], ["missing"]):
        with pytest.raises(ValueError):
            prepare_collection(cfg, bad_ids, "/usr/bin", "other")
    with pytest.raises(ValueError, match="run_id"):
        prepare_collection(cfg, ["b"], "/usr/bin", "../escape")


def test_lcb_explicit_ids_do_not_expose_private_data_or_silently_change_subset(tmp_path):
    from benchmark_adapters.code_collection import select_lcb_records

    rows = [
        dict(
            question_id="q1",
            question_title="Sum",
            question_content="Read and sum",
            starter_code="",
            difficulty="easy",
            platform="atcoder",
            contest_date="2023-01-01T00:00:00",
            metadata="{}",
            public_test_cases=json.dumps([dict(input="1 2", output="3", testtype="stdin")]),
            private_test_cases=json.dumps(
                [dict(input="SECRET", output="HIDDEN", testtype="stdin")]
            ),
        )
    ]
    path = tmp_path / "test.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    bundles = select_lcb_records([path], ["q1"], revision="v", checker_source="# checker")
    assert "SECRET" not in json.dumps(bundles[0].public_files)
    assert "SECRET" not in json.dumps(bundles[0].task.to_dict())
    for ids in (["unknown"], ["q1", "q1"]):
        with pytest.raises(ValueError):
            select_lcb_records([path], ids, revision="v", checker_source="# checker")
    rows[0]["difficulty"] = "hard"
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="easy/medium"):
        select_lcb_records([path], ["q1"], revision="v", checker_source="# checker")


def test_public_controller_source_rejected_and_new_output_is_private(tmp_path):
    import os
    import tempfile
    from pathlib import Path

    from benchmark_adapters.controller_paths import private_controller_directory

    if os.geteuid() != 0:
        pytest.skip("Local collection privacy preflight requires root to drop UID")
    with tempfile.TemporaryDirectory(prefix="flowpilot-private-path-test-", dir="/tmp") as name:
        public = Path(name)
        public.chmod(0o755)
        with pytest.raises(ValueError, match="accessible"):
            private_controller_directory(public)
        private = public / "new-output"
        private_controller_directory(private, create=True)
        assert private.stat().st_mode & 0o777 == 0o700
        private.chmod(0o755)
        with pytest.raises(ValueError, match="accessible"):
            private_controller_directory(private, create=True)
