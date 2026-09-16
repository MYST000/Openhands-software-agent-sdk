import json
import subprocess

import pytest

from benchmark_adapters.code_tasks import (
    ClassEvalAdapter,
    LiveCodeBenchAdapter,
    QuixBugsAdapter,
    evaluation_files,
    export_code,
    materialize_workspace,
)
from benchmark_adapters.contracts import CommandResult


def test_class_public_projection_and_evaluation_are_separate(tmp_path):
    row = dict(
        task_id="ClassEval_0",
        skeleton="class Counter:\n    pass\n",
        class_name="Counter",
        import_statement=[],
        class_description="counter",
        solution_code="REFERENCE_SECRET",
        test="PRIVATE_TEST_SECRET",
        test_classes=["CounterTest"],
    )
    bundle = ClassEvalAdapter.from_record(
        row,
        revision="v",
        harness_files={"test_pipeline.py": "# fixture", "path_util.py": "# fixture"},
    )
    assert "SECRET" not in json.dumps(bundle.task.to_dict())
    assert "SECRET" not in json.dumps(bundle.public_files)
    task = materialize_workspace(bundle, tmp_path / "repo")
    assert len(task.public_metadata["base_commit"]) == 40
    assert not list((tmp_path / "repo").glob("*test*"))
    files, command = evaluation_files(bundle, "class Counter:\n    value = 1\n")
    assert "PRIVATE_TEST_SECRET" in files["test_submission.py"]
    assert "REFERENCE_SECRET" not in files["test_submission.py"]
    assert "value = 1" in files["test_submission.py"]
    assert "unittest" in command


def test_lcb_public_cases_do_not_expose_private_cases():
    row = dict(
        question_id="q1",
        question_title="Sum",
        question_content="Read and sum",
        starter_code="",
        difficulty="easy",
        platform="atcoder",
        contest_date="2024-01-01T00:00:00",
        metadata="{}",
        public_test_cases=json.dumps([dict(input="1 2", output="3", testtype="stdin")]),
        private_test_cases=json.dumps([dict(input="SECRET", output="HIDDEN", testtype="stdin")]),
    )
    bundle = LiveCodeBenchAdapter.from_record(row, revision="v", checker_source="# checker fixture")
    assert "SECRET" not in json.dumps(bundle.task.to_dict()) + json.dumps(bundle.public_files)
    assert "1 2" in bundle.public_files["public_cases.json"]
    files, command = evaluation_files(bundle, "print(3)")
    assert "SECRET" in files["cases.json"]
    assert "print(3)" == files["solution.py"]
    assert files["testing_util.py"] == "# checker fixture"


def test_quix_evaluation_restores_tests_and_exports_only_selected_source(tmp_path):
    root = tmp_path / "quix"
    for name in (
        "python_programs",
        "correct_python_programs",
        "python_testcases",
        "json_testcases",
    ):
        (root / name).mkdir(parents=True)
    for name, text in {
        "python_programs/bitcount.py": "def bitcount(n): return 0",
        "correct_python_programs/bitcount.py": "REFERENCE_SECRET",
        "python_testcases/test_bitcount.py": "ORIGINAL_TEST",
        "python_testcases/load_testdata.py": "# loader",
        "conftest.py": "# config",
        "json_testcases/bitcount.json": "[1,1]",
    }.items():
        (root / name).write_text(text)
    bundle = QuixBugsAdapter.load_one(root, "bitcount", revision="v")
    assert "REFERENCE_SECRET" not in json.dumps(bundle.public_files)
    bundle.public_files["python_testcases/test_bitcount.py"] = "ATTEMPTED_TAMPER"
    files, _ = evaluation_files(bundle, "def bitcount(n): return 1")
    assert files["python_testcases/test_bitcount.py"] == "ORIGINAL_TEST"
    assert "REFERENCE_SECRET" not in json.dumps(files)
    assert files["python_programs/bitcount.py"] == "def bitcount(n): return 1"


def test_export_refuses_symlink_and_returns_actual_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solution.py").write_text("print(3)")

    class Env:
        repo_dir = str(repo)

        def execute(self, command, **kwargs):
            r = subprocess.run(["bash", "-c", command], cwd=repo, capture_output=True, text=True)
            return CommandResult(r.stdout, r.stderr, r.returncode, 1)

    assert export_code(Env(), "solution.py") == "print(3)"
    (repo / "solution.py").unlink()
    (repo / "solution.py").symlink_to("/etc/hostname")
    with pytest.raises(RuntimeError):
        export_code(Env(), "solution.py")


def test_command_supervisor_cannot_be_shadowed_by_actor_python_modules(tmp_path):
    from benchmark_adapters.environment import command_argv, file_operation_command

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solution.py").write_text("ACTUAL_CODE")
    fake = json.dumps(
        dict(
            stdout="FORGED_CODE",
            stderr="",
            exit_code=0,
            duration_ms=1,
            timed_out=False,
            truncated=False,
        )
    )
    (repo / "json.py").write_text("print(" + repr(fake) + "); raise SystemExit(0)")
    result = json.loads(
        subprocess.check_output(
            command_argv("cat solution.py", str(repo), 5, 10000), cwd=repo, text=True
        )
    )
    assert result["stdout"] == "ACTUAL_CODE"
    command = file_operation_command(str(repo), dict(command="view", path="solution.py"))
    result = json.loads(
        subprocess.check_output(command_argv(command, str(repo), 5, 10000), cwd=repo, text=True)
    )
    assert "ACTUAL_CODE" in result["stdout"]
    assert "FORGED_CODE" not in result["stdout"]


def test_submission_timeout_is_a_measured_failure_not_an_infrastructure_error(
    tmp_path, monkeypatch
):
    import benchmark_adapters.code_evaluation as evaluation
    from benchmark_adapters.code_tasks import CodeBundle
    from benchmark_adapters.config import Config
    from benchmark_adapters.contracts import Task

    class Env:
        def __init__(self, *a, **kw):
            pass

        def prepare(self):
            pass

        def execute(self, *a, **kw):
            return CommandResult("", "", -9, 1000, timed_out=True)

        def close(self):
            pass

    monkeypatch.setattr(evaluation, "LocalPilotEnvironment", Env)
    b = CodeBundle(
        "quixbugs",
        Task("quix", "v", "dev", "loop", "fix", {"protocol": "fixture"}),
        {"a.py": "while True:pass"},
        "a.py",
        {"evaluation_files": {"a.py": "while True:pass"}, "command": "python a.py"},
    )
    report = evaluation.evaluate_code(
        b, "while True:pass", Config(), tmp_path / "evaluation", python_bin="/usr/bin", timeout=1
    )
    assert report["status"] == "evaluated"
    assert report["passed"] is False
    assert report["details"]["failure_kind"] == "submission_timeout"


@pytest.mark.parametrize(
    "status,reason", [("budget_exhausted", "task_timeout"), ("llm_error", "Timeout")]
)
def test_audit_censors_deadline_timeout_without_inventing_a_next_action(tmp_path, status, reason):
    from benchmark_adapters.code_audit import audit_attempt
    from benchmark_adapters.tracing import TraceRecorder, write_json

    r = TraceRecorder(tmp_path, {"task_id": "fixture"})
    r.request(
        "r1",
        {"messages": [], "tools": []},
        logical_request_id="l1",
        request_role="actor",
        snapshot_stage="litellm_transport_input",
    )
    r.emit("llm_error", request_id="r1", error_type="Timeout")
    r.emit("task_error", error_type="Timeout", reason=reason)
    r.close()
    write_json(
        tmp_path / "result.json",
        dict(task_id="fixture", execution_status=status, termination_reason=reason),
    )
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/solution.py").write_text("pass")
    report = audit_attempt(tmp_path)
    assert report["audit_passed"]
    assert report["budget_censored_request_ids"] == ["r1"]
    assert (tmp_path / "prediction_samples.jsonl").read_text() == ""
