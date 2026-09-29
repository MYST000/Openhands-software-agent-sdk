import json

import pyarrow as pa
import pyarrow.parquet as pq


def test_evaluation_plan_uses_only_selected_local_gold(tmp_path):
    from benchmark_adapters.config import Config, DatasetConfig
    from benchmark_adapters.evaluation import prepare_swe_evaluation

    rows = [
        dict(
            instance_id=f"org__repo-{i}",
            repo="org/repo",
            base_commit="a" * 40,
            problem_statement="issue",
            patch=f"GOLD{i}",
            test_patch="TEST",
            version="1",
        )
        for i in range(3)
    ]
    data = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data)
    cfg = Config(dataset=DatasetConfig(path=str(data), revision="fixed"))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(dict(instance_id="org__repo-1", model_name_or_path="model", model_patch=""))
        + "\n"
    )
    plan = prepare_swe_evaluation(cfg, tmp_path / "run", ["org__repo-1"], predictions, "r")
    from pathlib import Path

    gold_path = Path(plan["cwd"]) / "gold.jsonl"
    gold = [json.loads(x) for x in gold_path.read_text().splitlines()]
    assert [x["instance_id"] for x in gold] == ["org__repo-1"]
    assert gold[0]["patch"] == "GOLD1"
    assert plan["command"][plan["command"].index("--dataset_name") + 1] == str(gold_path)
    assert "--instance_image_tag" in plan["command"]
    predictions.write_text(
        json.dumps(dict(instance_id="org__repo-1", model_name_or_path="model", model_patch="NEW"))
        + "\n"
    )
    changed = prepare_swe_evaluation(cfg, tmp_path / "run", ["org__repo-1"], predictions, "r")
    assert changed["cwd"] != plan["cwd"]


def test_swe_aggregate_distinguishes_empty_patches_and_evaluation_errors(tmp_path):
    from benchmark_adapters.evaluation import parse_swe_aggregate

    report = dict(
        total_instances=2,
        submitted_ids=["a", "b"],
        resolved_ids=[],
        unresolved_ids=[],
        empty_patch_ids=["a", "b"],
        error_ids=[],
        incomplete_ids=[],
    )
    (tmp_path / "model.eval.json").write_text(json.dumps(report))
    parsed = parse_swe_aggregate(tmp_path, ["a", "b"])
    assert parsed["evaluation_status"] == "scored"
    assert parsed["resolved_by_task"] == {"a": False, "b": False}
    report.update(empty_patch_ids=["a"], error_ids=["b"])
    (tmp_path / "model.eval.json").write_text(json.dumps(report))
    parsed = parse_swe_aggregate(tmp_path, ["a", "b"])
    assert parsed["evaluation_status"] == "partial_error"
    assert parsed["resolved_by_task"]["b"] is None


def test_export_keeps_failed_tasks_in_denominator(tmp_path):
    from benchmark_adapters.cli import export_run

    run = tmp_path / "run"
    run.mkdir()
    tasks = [dict(task_id=f"org__repo-{i}") for i in range(2)]
    manifest = dict(config=dict(dataset=dict(kind="swe"), llm=dict(model="model")), tasks=tasks)
    (run / "manifest.json").write_text(json.dumps(manifest))
    attempt = run / "tasks/org__repo-0/attempt-1"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(
        json.dumps(dict(execution_status="completed", artifact_status="valid"))
    )
    (attempt / "submission.json").write_text(
        json.dumps(dict(instance_id="org__repo-0", model_name_or_path="model", model_patch="PATCH"))
    )
    path = export_run(run)
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["instance_id"] == "org__repo-1" and rows[1]["model_patch"] == ""


def test_original_dev_image_build_uses_local_harness_and_selected_tasks(tmp_path):
    from benchmark_adapters.config import Config, DatasetConfig, DockerConfig, EvaluationConfig
    from benchmark_adapters.images import prepare_image_build
    from benchmark_adapters.swe import SWEAdapter

    record = dict(
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="a" * 40,
        problem_statement="issue",
        patch="gold",
    )
    data = tmp_path / "dev.json"
    data.write_text(json.dumps([record]))
    cfg = Config(
        dataset=DatasetConfig(path=str(data), revision="fixed"),
        docker=DockerConfig(image_template="sweb.eval.x86_64.{instance_id}:latest"),
        evaluation=EvaluationConfig(namespace="none"),
    )
    tasks = SWEAdapter.load(
        data, dataset_id="princeton-nlp/SWE-bench", revision="fixed", split="dev"
    )
    plan = prepare_image_build(cfg, tasks, tmp_path / "build")
    assert plan["command"][2] == "swebench.harness.prepare_images"
    assert plan["command"][plan["command"].index("--namespace") + 1] == "none"
    assert plan["expected_images"] == ["sweb.eval.x86_64.org__repo-1:latest"]


def test_image_build_reports_missing_image_even_when_harness_exits_zero(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import docker

    from benchmark_adapters import images
    from benchmark_adapters.config import Config, DatasetConfig, DockerConfig, EvaluationConfig
    from benchmark_adapters.swe import SWEAdapter

    data = tmp_path / "dev.json"
    data.write_text(
        json.dumps(
            [
                dict(
                    instance_id="org__repo-1",
                    repo="org/repo",
                    base_commit="a" * 40,
                    problem_statement="issue",
                    patch="gold",
                )
            ]
        )
    )
    cfg = Config(
        dataset=DatasetConfig(path=str(data), revision="fixed"),
        docker=DockerConfig(image_template="sweb.eval.x86_64.{instance_id}:latest"),
        evaluation=EvaluationConfig(namespace="none"),
    )
    tasks = SWEAdapter.load(data, dataset_id=cfg.dataset.id, revision="fixed", split="dev")
    plan = images.prepare_image_build(cfg, tasks, tmp_path / "build")
    # Simulate upstream builder reporting partial failure with process exit code 0.
    plan["command"] = [sys.executable, "-c", 'print("Failed to build 1 images")']
    monkeypatch.setattr(images, "verify_harness", lambda p: {"version": "4.1.0"})

    def missing(name):
        raise docker.errors.ImageNotFound(name)

    monkeypatch.setattr(
        docker,
        "from_env",
        lambda **kw: SimpleNamespace(
            ping=lambda: True, images=SimpleNamespace(get=missing), close=lambda: None
        ),
    )
    result = images.execute_image_build(plan)
    assert result["build_status"] == "build_error"
    assert result["missing_images"] == plan["expected_images"]


def test_build_images_cli_writes_offline_plan(tmp_path, capsys):
    from benchmark_adapters.cli import main

    data = tmp_path / "dev.json"
    data.write_text(
        json.dumps(
            [
                dict(
                    instance_id="org__repo-1",
                    repo="org/repo",
                    base_commit="a" * 40,
                    problem_statement="issue",
                    patch="gold",
                )
            ]
        )
    )
    config = tmp_path / "config.toml"
    config.write_text(f'''[dataset]
path = "{data}"
revision = "fixed"
[docker]
image_template = "sweb.eval.x86_64.{{instance_id}}:latest"
[evaluation]
namespace = "none"
[runtime]
runs_dir = "{tmp_path / "runs"}"
''')
    assert main(["build-images", "--config", str(config)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["build_status"] == "pending"
    assert plan["task_ids"] == ["org__repo-1"]


def test_resume_rejects_changed_adapter_source(tmp_path, monkeypatch):
    import pytest

    from benchmark_adapters import cli, runner
    from benchmark_adapters.config import Config, DatasetConfig, RuntimeConfig

    data = tmp_path / "tasks.json"
    data.write_text(
        json.dumps(
            [
                dict(
                    instance_id="org__repo-1",
                    repo="org/repo",
                    base_commit="a" * 40,
                    problem_statement="issue",
                )
            ]
        )
    )
    cfg = Config(
        dataset=DatasetConfig(path=str(data), revision="fixed"),
        runtime=RuntimeConfig(runs_dir=str(tmp_path / "runs")),
    )
    tasks = cli.tasks_for(cfg)

    def fake_run(config, task, attempt_dir, **kwargs):
        result = dict(execution_status="completed", artifact_status="empty")
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(runner, "run_task", fake_run)
    root, _ = cli.run_selected(cfg, tasks, "run")
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["adapter_source_sha256"]) == 64
    assert cli.run_selected(cfg, tasks, "run", resume=True)[0] == root
    monkeypatch.setattr(cli, "adapter_source_digest", lambda: "0" * 64)
    with pytest.raises(ValueError, match="source"):
        cli.run_selected(cfg, tasks, "run", resume=True)
