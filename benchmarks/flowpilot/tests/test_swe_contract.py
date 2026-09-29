import json
import subprocess

import pytest


def test_swe_input_excludes_gold_and_validates_commit():
    from benchmark_adapters.swe import SWEAdapter

    raw = dict(
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="a" * 40,
        problem_statement="Fix the reported problem.",
        patch="GOLD_SECRET",
        test_patch="TEST_SECRET",
        FAIL_TO_PASS='["hidden_test"]',
    )
    task = SWEAdapter.from_record(
        raw, dataset_id="princeton-nlp/SWE-bench", revision="locked", split="dev"
    )
    assert "GOLD_SECRET" not in json.dumps(task.to_dict())
    assert "TEST_SECRET" not in task.instruction
    assert "hidden_test" not in task.instruction
    with pytest.raises(ValueError):
        SWEAdapter.from_record(
            {**raw, "base_commit": "main; echo bad"},
            dataset_id="princeton-nlp/SWE-bench",
            revision="x",
            split="dev",
        )


def test_local_parquet_selection_is_exact(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from benchmark_adapters.swe import SWEAdapter

    records = [
        dict(
            instance_id=f"org__repo-{i}",
            repo="org/repo",
            base_commit="a" * 40,
            problem_statement=f"Issue {i}",
            patch="SECRET",
        )
        for i in range(3)
    ]
    p = tmp_path / "dev.parquet"
    pq.write_table(pa.Table.from_pylist(records), p)
    tasks = SWEAdapter.load(
        p,
        dataset_id="princeton-nlp/SWE-bench",
        revision="x",
        split="dev",
        ids=["org__repo-2", "org__repo-0"],
        limit=0,
    )
    assert [x.task_id for x in tasks] == ["org__repo-2", "org__repo-0"]
    with pytest.raises(ValueError, match="missing"):
        SWEAdapter.load(
            p,
            dataset_id="princeton-nlp/SWE-bench",
            revision="x",
            split="dev",
            ids=["not-present"],
            limit=0,
        )


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True)


def test_patch_covers_commits_index_worktree_new_and_deleted_files(tmp_path):
    from benchmark_adapters.swe import export_patch_command

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "test@example.invalid")
    git(repo, "config", "user.name", "Test")
    for name in ["committed", "staged", "unstaged", "deleted"]:
        (repo / name).write_text("before\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD").strip()
    (repo / "committed").write_text("after commit\n")
    git(repo, "add", "committed")
    git(repo, "commit", "-qm", "agent commit")
    (repo / "staged").write_text("after staged\n")
    git(repo, "add", "staged")
    (repo / "unstaged").write_text("after unstaged\n")
    (repo / "new file").write_text("new\n")
    (repo / ".gitignore").write_text("ignored.py\n")
    (repo / "ignored.py").write_text("explicit staged file\n")
    git(repo, "add", "-f", "ignored.py")
    (repo / "deleted").unlink()
    index_before = (repo / ".git/index").read_bytes()
    patch = subprocess.check_output(["bash", "-c", export_patch_command(base)], cwd=repo, text=True)
    assert (repo / ".git/index").read_bytes() == index_before
    target = tmp_path / "target"
    subprocess.run(["git", "clone", "-q", str(repo), str(target)], check=True)
    git(target, "checkout", "-q", base)
    subprocess.run(
        ["git", "-C", str(target), "apply", "--binary", "-"], input=patch, text=True, check=True
    )
    for name in ["committed", "staged", "unstaged", "new file", "ignored.py"]:
        assert (target / name).read_bytes() == (repo / name).read_bytes()
    assert not (target / "deleted").exists()


def test_config_rejects_unknown_keys_and_verified_train(tmp_path):
    from benchmark_adapters.config import load_config

    p = tmp_path / "bad.toml"
    p.write_text(
        '[dataset]\nkind="swe"\nsplit="train"\nid="princeton-nlp/SWE-bench_Verified"\npath="x"\nrevision="x"\n'
    )
    with pytest.raises(ValueError, match="Verified"):
        load_config(p)
    p.write_text(
        '[dataset]\nkind="swe"\nsplit="dev"\nid="princeton-nlp/SWE-bench"\npath="x"\nrevision="x"\n[llm]\ntemperatur=0\n'
    )
    with pytest.raises(ValueError, match="temperatur"):
        load_config(p)


@pytest.mark.parametrize(
    "extra,match",
    [('[docker]\nrepo_dir="relative"\n', "absolute"), ("[docker]\nnano_cpus=-1\n", "positive")],
)
def test_invalid_container_configuration(tmp_path, extra, match):
    from benchmark_adapters.config import load_config

    p = tmp_path / "config.toml"
    p.write_text(
        '[dataset]\nkind="swe"\nid="princeton-nlp/SWE-bench"\npath="x"\nrevision="x"\n' + extra
    )
    with pytest.raises(ValueError, match=match):
        load_config(p)


def test_unknown_dataset_identity_is_rejected(tmp_path):
    from benchmark_adapters.config import load_config

    p = tmp_path / "config.toml"
    p.write_text('[dataset]\nkind="swe"\nid="unknown/dataset"\npath="x"\nrevision="x"\n')
    with pytest.raises(ValueError, match="dataset"):
        load_config(p)


def test_evaluation_python_preserves_virtualenv_symlink(tmp_path):
    import sys

    from benchmark_adapters.config import load_config

    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    path = tmp_path / "config.toml"
    path.write_text(
        '[dataset]\npath="tasks.jsonl"\nrevision="fixed"\n[evaluation]\npython="venv/bin/python"\n'
    )
    cfg = load_config(path)
    assert cfg.evaluation.python == str(python)


def test_config_rejects_unimplemented_setting(tmp_path):
    from benchmark_adapters.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        '[dataset]\nkind="hotpot"\nid="hotpotqa"\npath="x"\n'
        'revision="fixed"\nsetting="distractor-pilot-v1"\n'
    )
    with pytest.raises(ValueError, match="setting"):
        load_config(path)


def test_patch_detects_same_size_edit_with_racy_git_index(tmp_path):
    import os
    import time

    from benchmark_adapters.swe import export_patch_command

    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "core.trustctime", "false")
    source = tmp_path / "source.py"
    source.write_text("x=0\n")
    stamp = time.time_ns() - 10_000_000_000
    os.utime(source, ns=(stamp, stamp))
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD").strip()
    # Reproduce a coarse filesystem timestamp: contents changed, cached size/mtime match.
    source.write_text("x=1\n")
    os.utime(source, ns=(stamp, stamp))
    os.utime(tmp_path / ".git/index", ns=(stamp, stamp))
    patch = subprocess.check_output(
        ["bash", "-c", export_patch_command(base)], cwd=tmp_path, text=True
    )
    assert "+x=1" in patch


def test_non_native_tool_calling_is_rejected_for_prediction_tracing(tmp_path):
    from benchmark_adapters.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('[dataset]\npath="x"\nrevision="fixed"\n[llm]\nnative_tool_calling=false\n')
    with pytest.raises(ValueError, match="native_tool_calling"):
        load_config(path)
