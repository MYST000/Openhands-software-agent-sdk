import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .swe import checked_dataset_digest, load_records, select_records
from .tracing import write_json

HARNESS_COMMIT = "726c5461e2ef52d83cf1ea2107870a8bb3328d57"


def _freeze(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen evaluation input changed: {path}")
    if not path.exists():
        path.write_bytes(data)


def prepare_swe_evaluation(config, run_dir, ids, predictions, run_id):
    dataset_hash = checked_dataset_digest(config)
    rows = select_records(load_records(config.dataset.path), "instance_id", ids)
    gold_bytes = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    prediction_bytes = Path(predictions).read_bytes()
    identity = dict(
        dataset_sha256=dataset_hash,
        gold_sha256=hashlib.sha256(gold_bytes).hexdigest(),
        predictions_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
        evaluator=asdict(config.evaluation),
        harness_commit=HARNESS_COMMIT,
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    evaluation_id = "eval_" + digest[:20]
    root = Path(run_dir).resolve() / "evaluations" / evaluation_id
    root.mkdir(parents=True, exist_ok=True)
    gold, prediction = root / "gold.jsonl", root / "predictions.jsonl"
    _freeze(gold, gold_bytes)
    _freeze(prediction, prediction_bytes)
    python = config.evaluation.python or sys.executable
    command = [
        python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(gold),
        "--predictions_path",
        str(prediction),
        "--run_id",
        evaluation_id,
        "--max_workers",
        str(config.evaluation.workers),
        "--timeout",
        str(config.evaluation.timeout),
        "--namespace",
        config.evaluation.namespace,
        "--instance_image_tag",
        config.evaluation.image_tag,
        "--cache_level",
        "env",
    ]
    plan = dict(
        evaluator="swebench==4.1.0",
        command=command,
        cwd=str(root),
        task_ids=ids,
        evaluation_status="pending",
        logs="evaluator.log",
        identity=identity,
        source_run_id=run_id,
        evaluation_id=evaluation_id,
        harness_path=config.evaluation.harness_path,
    )
    write_json(root / "evaluation_plan.json", plan)
    return plan


def verify_harness(plan):
    source = Path(plan["harness_path"]).resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        head != HARNESS_COMMIT
        or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Official SWE harness must be clean at the pinned v4.1.0 commit")
    code = "import importlib.metadata,json,swebench; d=importlib.metadata.distribution('swebench'); print(json.dumps({'version':d.version,'file':swebench.__file__,'direct_url':json.loads(d.read_text('direct_url.json') or '{}')}))"
    info = json.loads(subprocess.check_output([plan["command"][0], "-c", code], text=True))
    if info["version"] != "4.1.0" or info["direct_url"].get("url") != source.as_uri():
        raise ValueError(
            "Evaluation interpreter must install swebench==4.1.0 from configured local checkout"
        )
    installed_root = Path(info["file"]).parent
    for relative in [
        "harness/run_evaluation.py",
        "harness/grading.py",
        "harness/reporting.py",
        "harness/prepare_images.py",
        "harness/docker_build.py",
        "harness/test_spec/test_spec.py",
    ]:
        if (installed_root / relative).read_bytes() != (
            source / "swebench" / relative
        ).read_bytes():
            raise ValueError(f"Installed harness differs from configured source: {relative}")
    return info


def parse_swe_aggregate(root, ids):
    reports = []
    for path in Path(root).glob("*.json"):
        data = json.loads(path.read_text())
        if (
            isinstance(data, dict)
            and {"total_instances", "resolved_ids", "empty_patch_ids", "error_ids"} <= data.keys()
        ):
            reports.append((path, data))
    if len(reports) != 1:
        raise ValueError(f"Expected one official aggregate report, found {len(reports)}")
    path, report = reports[0]
    if report["total_instances"] != len(ids):
        raise ValueError("Official evaluator denominator differs from selected task count")
    categories = [
        "resolved_ids",
        "unresolved_ids",
        "empty_patch_ids",
        "error_ids",
        "incomplete_ids",
    ]
    if set().union(*(set(report.get(k, [])) for k in categories)) != set(ids):
        raise ValueError("Official aggregate task IDs differ from selection")
    errors = set(report.get("error_ids", [])) | set(report.get("incomplete_ids", []))
    resolved = {
        task_id: None if task_id in errors else task_id in report["resolved_ids"] for task_id in ids
    }
    return dict(
        evaluation_status="partial_error" if errors else "scored",
        resolved_by_task=resolved,
        official_aggregate=report,
        aggregate_path=str(path),
    )


def execute_evaluation(plan):
    root = Path(plan["cwd"])
    status = {**plan, "evaluation_status": "running"}
    write_json(root / "evaluation_status.json", status)
    try:
        status["harness_verified"] = verify_harness(plan)
        for name, field in [
            ("gold.jsonl", "gold_sha256"),
            ("predictions.jsonl", "predictions_sha256"),
        ]:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != plan["identity"][field]:
                raise ValueError(f"Frozen evaluation input checksum mismatch: {name}")
        with (root / "evaluator.log").open("w") as log:
            proc = subprocess.run(plan["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status["returncode"] = proc.returncode
        if proc.returncode:
            status["evaluation_status"] = "evaluator_error"
        else:
            status.update(parse_swe_aggregate(root, plan["task_ids"]))
    except (Exception, KeyboardInterrupt) as exc:
        status["evaluation_status"] = "evaluator_error"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
    write_json(root / "evaluation_status.json", status)
    return status
