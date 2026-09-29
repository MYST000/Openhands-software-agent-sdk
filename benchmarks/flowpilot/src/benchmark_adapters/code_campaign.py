"""Reproducible local development campaign. Run each stage explicitly and in order.

python -m benchmark_adapters.code_campaign --ops PATH --stage quixbugs --validate
python -m benchmark_adapters.code_campaign --ops PATH --stage quixbugs --run
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .cli import adapter_source_digest, task_key
from .code_audit import audit_attempt
from .code_evaluation import evaluate_code
from .code_tasks import (
    ClassEvalAdapter,
    LiveCodeBenchAdapter,
    QuixBugsAdapter,
    materialize_workspace,
)
from .config import Config, DatasetConfig, LLMConfig, RuntimeConfig
from .local_environment import LocalPilotEnvironment
from .runner import run_task
from .sdk_provenance import check_sdk
from .tracing import write_json

QUIX_REV = "4257f44b0ff1181dedaedee6a447e133219fcebf"
LCB_REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_CHECKER_REV = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
CLASS_REV = "eaeac44d0d5dcd8a95feec50726d66fedc73a98f"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundles(ops, stage):
    sources = ops / "sources"
    provenance = {}
    if stage == "quixbugs":
        repo = Path("/root/predictor_exp/repos/QuixBugs")
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if (
            actual != QUIX_REV
            or subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        ):
            raise ValueError("QuixBugs checkout must be clean at pinned commit")
        names = sorted(
            p.stem.removeprefix("test_") for p in (repo / "python_testcases").glob("test_*.py")
        )[:8]
        bundles = [QuixBugsAdapter.load_one(repo, name, revision=QUIX_REV) for name in names]
        provenance = {
            "revision": QUIX_REV,
            "selection_rule": "first 8 Python program tests sorted lexicographically",
        }
    elif stage == "livecodebench":
        paths = [
            sources / "lcb_release_v6" / name
            for name in ["test.jsonl"] + [f"test{i}.jsonl" for i in range(2, 7)]
        ]
        rows = []
        for p in paths:
            with p.open() as stream:
                rows.extend(json.loads(line) for line in stream)
        checker = (ops / "lcb_testing_util.reference.py").read_text()
        eligible = []
        for r in rows:
            metadata = json.loads(r["metadata"])
            public = json.loads(r["public_test_cases"])
            if (
                r["difficulty"] in {"easy", "medium"}
                and public
                and not metadata.get("func_name")
                and all(t["testtype"] == "stdin" for t in public)
            ):
                eligible.append(r)
        selected = []
        for diff in ("easy", "medium"):
            part = sorted(
                (r for r in eligible if r["difficulty"] == diff),
                key=lambda r: str(r["question_id"]),
            )[:12]
            if len(part) != 12:
                raise ValueError("Insufficient eligible tasks for fixed selection")
            selected.extend(part)
        bundles = [
            LiveCodeBenchAdapter.from_record(r, revision=LCB_REV, checker_source=checker)
            for r in selected
        ]
        provenance = {
            "revision": LCB_REV,
            "release": "release_v6",
            "total_release_tasks": len(rows),
            "files": {p.name: sha(p) for p in paths},
            "checker_commit": LCB_CHECKER_REV,
            "checker_sha256": sha(ops / "lcb_testing_util.reference.py"),
            "selection_rule": "12 easy + 12 medium; stdin-only; each group sorted question_id",
            "eligible_counts": dict(Counter(r["difficulty"] for r in eligible)),
        }
    else:
        path = sources / "ClassEval_data.json"
        rows = json.loads(path.read_text())
        eligible = []
        for r in sorted(rows, key=lambda r: r["task_id"]):
            modules = set()
            try:
                for source in (r["skeleton"], r["solution_code"], r["test"]):
                    for node in ast.walk(ast.parse(source)):
                        if isinstance(node, ast.Import):
                            modules.update(a.name.split(".")[0] for a in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            modules.add(node.module.split(".")[0])
            except SyntaxError:
                continue
            if modules <= sys.stdlib_module_names:
                eligible.append(r)
        harness_root = Path("/root/predictor_exp/repos/ClassEval/classeval_evaluation")
        harness = {
            name: (harness_root / name).read_text() for name in ("test_pipeline.py", "path_util.py")
        }
        bundles = [
            ClassEvalAdapter.from_record(r, revision=CLASS_REV, harness_files=harness)
            for r in eligible[:6]
        ]
        provenance = {
            "revision": CLASS_REV,
            "data_sha256": sha(path),
            "harness_sha256": {name: sha(harness_root / name) for name in harness},
            "selection_rule": "first 6 sorted task IDs with statically stdlib-only imports and parsable sources",
            "eligible_count": len(eligible),
        }
    return bundles, provenance


def make_config(bundle, ops):
    return Config(
        dataset=DatasetConfig(
            kind=bundle.kind,
            id=bundle.task.dataset_id,
            path=str(ops / "sources"),
            revision=bundle.task.revision,
            split="dev",
            setting="openhands-python-code-v1",
        ),
        llm=LLMConfig(max_output_tokens=16384, timeout=600, num_retries=0),
        runtime=replace(
            RuntimeConfig(),
            max_iterations=24,
            max_tool_calls=36,
            max_llm_requests=28,
            task_timeout=900,
            tool_timeout=30,
            runs_dir=str(ops),
        ),
    )


def freeze_selection(ops, stage, bundles, provenance, config):
    root = ops / stage
    root.mkdir(exist_ok=True)
    data = {
        "stage": stage,
        "provenance": provenance,
        "config": config.to_dict(),
        "tasks": [b.task.to_dict() for b in bundles],
        "public_files_sha256": {
            b.task.task_id: hashlib.sha256(
                json.dumps(b.public_files, sort_keys=True).encode()
            ).hexdigest()
            for b in bundles
        },
    }
    path = root / "selection.json"
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(
                "Frozen selection or configuration changed; use a new operations directory"
            )
    else:
        write_json(path, data)
    return root


def runtime_fingerprint(python_bin):
    command = "import sys,importlib.metadata as m,json; print(json.dumps({'python':sys.version,'packages':{p:m.version(p) for p in ('pytest','numpy','scipy','func_timeout')}},sort_keys=True))"
    return subprocess.check_output(
        [str(Path(python_bin) / "python"), "-I", "-c", command], text=True
    ).strip()


def validate(bundles, config, root, python_bin):
    if (root / "validation.json").exists():
        raise ValueError("Validation already exists; inspect it, do not overwrite")
    source_hash = adapter_source_digest()
    runtime_hash = runtime_fingerprint(python_bin)
    reports = []
    if bundles[0].kind == "livecodebench":
        # Known checker conformance cases, NOT a claim that dataset reference solutions were tested.
        base = bundles[0]
        fixtures = [
            ("correct_sum", "print(sum(map(int,input().split())))", True),
            ("wrong_answer", "print(99)", False),
            ("syntax_error", "def invalid(", False),
            ("timeout", "while True: pass", False),
        ]
        for name, code, expected in fixtures:
            b = replace(
                base,
                private={
                    **base.private,
                    "sample": {
                        "input_output": json.dumps(
                            {"inputs": ["1 2", "4 5"], "outputs": ["3", "9"], "fn_name": None}
                        )
                    },
                },
            )
            r = evaluate_code(
                b, code, config, root / "validation" / name, python_bin=python_bin, timeout=20
            )
            reports.append(
                {
                    "fixture": name,
                    "expected": expected,
                    "check_ok": r["status"] == "evaluated" and r["passed"] == expected,
                    "report": r,
                }
            )
    else:
        for b in bundles:
            ref = evaluate_code(
                b,
                b.private["reference_code"],
                config,
                root / "validation" / task_key(b.task.task_id) / "reference",
                python_bin=python_bin,
                timeout=60,
            )
            baseline = evaluate_code(
                b,
                b.public_files[b.solution_path],
                config,
                root / "validation" / task_key(b.task.task_id) / "baseline",
                python_bin=python_bin,
                timeout=10,
            )
            reports.append(
                {
                    "task_id": b.task.task_id,
                    "check_ok": ref["status"] == "evaluated" and ref["passed"],
                    "reference": ref,
                    "baseline": baseline,
                }
            )
            print("VALIDATE", b.task.task_id, ref["status"], ref["passed"], flush=True)
    report = {
        "stage": bundles[0].kind,
        "all_checks_passed": all(r["check_ok"] for r in reports),
        "selection_sha256": sha(root / "selection.json"),
        "reports": reports,
        "adapter_source_sha256": source_hash,
        "runtime_fingerprint": runtime_hash,
        "validation_kind": "upstream-checker-conformance-fixtures"
        if bundles[0].kind == "livecodebench"
        else "upstream-reference-and-buggy-or-skeleton-baseline",
    }
    write_json(root / "validation.json", report)
    return report


def summarize(root, results, *, finalized=False):
    tools = Counter()
    proposed = Counter()
    durations = []
    costs = []
    prompts = []
    completions = []
    tests = 0
    args_lengths = []
    for record in results:
        audit = record["audit"]
        proposed.update(audit["proposed_action_counts"])
        tools.update(audit["executed_tool_counts"])
        costs.append(record["actor"]["duration_s"])
        prompts.append(audit["prompt_tokens_sum"])
        completions.append(audit["completion_tokens_sum"])
        events = [
            json.loads(line)
            for line in (Path(record["attempt"]) / "events.jsonl").read_text().splitlines()
        ]
        task_tests = False
        for e in events:
            if e["event"] == "tool_end":
                durations.append(e["executor_duration_ms"])
            if e["event"] == "tool_start":
                args_lengths.append(len(json.dumps(e.get("arguments", {}), ensure_ascii=False)))
                cmd = e.get("arguments", {}).get("command", "")
                if e.get("tool_name") == "code_terminal" and any(
                    s in cmd
                    for s in (
                        "pytest",
                        "unittest",
                        "doctest",
                        "check_public.py",
                        "test_",
                        "assert ",
                    )
                ):
                    task_tests = True
        tests += task_tests

    def dist(xs):
        if not xs:
            return {}
        ys = sorted(xs)
        return {
            "min": min(xs),
            "median": statistics.median(xs),
            "p90": ys[min(len(ys) - 1, int(0.9 * len(ys)))],
            "max": max(xs),
            "sum": sum(xs),
        }

    n = len(results)
    summary = {
        "attempted": n,
        "completed": sum(r["actor"]["execution_status"] == "completed" for r in results),
        "benchmark_passed": sum(r["evaluation"]["passed"] for r in results),
        "evaluation_errors": sum(r["evaluation"]["status"] != "evaluated" for r in results),
        "trace_audit_passed": sum(r["audit"]["audit_passed"] for r in results),
        "task_tests_detected_heuristically": tests,
        "proposed_action_counts": dict(proposed),
        "executed_tool_counts": dict(tools),
        "llm_requests": sum(r["audit"]["request_count"] for r in results),
        "task_duration_s": dist(costs),
        "executor_duration_ms": dist(durations),
        "argument_chars": dist(args_lengths),
        "prompt_tokens_sum": sum(prompts),
        "completion_tokens_sum": sum(completions),
        "max_prompt_tokens": max((r["audit"]["max_prompt_tokens"] for r in results), default=0),
        "results": [
            {
                "task_id": r["actor"]["task_id"],
                "execution_status": r["actor"]["execution_status"],
                "passed": r["evaluation"]["passed"],
                "evaluation_status": r["evaluation"]["status"],
                "requests": r["audit"]["request_count"],
                "tools": r["audit"]["tool_count"],
                "duration_s": r["actor"]["duration_s"],
                "audit_passed": r["audit"]["audit_passed"],
            }
            for r in results
        ],
    }
    summary["planned_tasks"] = len(json.loads((root / "selection.json").read_text())["tasks"])
    summary["stage_complete"] = finalized and n == summary["planned_tasks"]
    summary["infrastructure_gate_passed"] = (
        bool(n) and summary["trace_audit_passed"] == n and summary["evaluation_errors"] == 0
    )
    summary["viability_gate_passed"] = (
        summary["infrastructure_gate_passed"]
        and summary["completed"] > 0
        and summary["benchmark_passed"] > 0
    )
    write_json(root / "summary.json", summary)
    return summary


def run_stage(bundles, config, root, python_bin, *, require_previous_stage=True, run_id=None):
    validation = json.loads((root / "validation.json").read_text())
    if (
        not validation["all_checks_passed"]
        or validation["selection_sha256"] != sha(root / "selection.json")
        or validation.get("adapter_source_sha256") != adapter_source_digest()
        or validation.get("runtime_fingerprint") != runtime_fingerprint(python_bin)
    ):
        raise ValueError("Reference/checker validation gate failed")
    stage = bundles[0].kind
    previous = {"livecodebench": "quixbugs", "classeval": "livecodebench"}.get(stage)
    if previous and require_previous_stage:
        prev = json.loads((root.parent / previous / "summary.json").read_text())
        if not prev.get("stage_complete") or not prev["viability_gate_passed"]:
            raise ValueError("Previous stage did not establish viable data collection")
    if (root / "manifest.json").exists():
        raise ValueError("Actor run already exists; never silently rerun failed tasks")
    sdk = check_sdk(config.runtime)
    source_hash = adapter_source_digest()
    write_json(
        root / "manifest.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "selection_sha256": sha(root / "selection.json"),
            "adapter_source_sha256": source_hash,
            "sdk_commit": sdk["checkout_commit"],
            "sdk_provenance": sdk,
            "model_service": {"model": config.llm.model, "base_url": config.llm.base_url},
            "python_bin": str(python_bin),
            "executor_python": subprocess.check_output(
                [str(Path(python_bin) / "python"), "--version"], text=True
            ).strip(),
            "packages": {
                p: importlib.metadata.version(p) for p in ("openhands-sdk", "litellm", "pydantic")
            },
            "single_attempt": True,
            "config": config.to_dict(),
            "purpose": "agentic development pilot, not original leaderboard protocol",
        },
    )
    with tarfile.open(root / "adapter_source_snapshot.tar.gz", "w:gz") as tar:
        tar.add(
            Path(__file__).parent,
            arcname="benchmark_adapters",
            filter=lambda t: None if "__pycache__" in t.name else t,
        )
    results = []
    for b in bundles:
        if adapter_source_digest() != source_hash:
            raise ValueError("Adapter source changed mid-stage; stopping")
        task_root = Path(tempfile.mkdtemp(prefix="flowpilot-code-local-actor-", dir="/tmp"))
        attempt = root / "tasks" / task_key(b.task.task_id) / "attempt-001"
        try:
            task = materialize_workspace(b, task_root / "repo")
            env = LocalPilotEnvironment(
                config,
                task,
                attempt / "artifacts",
                task_root=task_root,
                python_bin=python_bin,
                uid=63111,
            )
            print("ACTOR_START", stage, b.task.task_id, flush=True)
            actor = run_task(
                config, task, attempt, environment=env, run_id=run_id or root.name + "-20260915"
            )
            write_json(attempt / "public_files.json", b.public_files)
            if not (attempt / "artifacts/solution.py").exists():
                raise RuntimeError("No exported code; stop for infrastructure inspection")
            audit = audit_attempt(attempt)
            evaluation = evaluate_code(
                b,
                (attempt / "artifacts/solution.py").read_text(),
                config,
                attempt / "evaluation",
                python_bin=python_bin,
            )
            # Preserve final files including actor-authored tests; symlinks are stored without dereferencing.
            with tarfile.open(
                attempt / "artifacts/final_workspace.tar.gz", "w:gz", dereference=False
            ) as tar:
                tar.add(
                    task_root / "repo",
                    arcname="repo",
                    filter=lambda t: (
                        None
                        if any(
                            x in t.name.split("/") for x in (".git", "__pycache__", ".pytest_cache")
                        )
                        or t.size > 10000000
                        else t
                    ),
                )
            record = dict(actor=actor, audit=audit, evaluation=evaluation, attempt=str(attempt))
            results.append(record)
            summary = summarize(root, results)
            print("ACTOR_RESULT", json.dumps(summary["results"][-1]), flush=True)
            if not audit["audit_passed"] or evaluation["status"] != "evaluated":
                raise RuntimeError("Trace/evaluator gate failed; inspect retained attempt")
        finally:
            shutil.rmtree(task_root)
    return summarize(root, results, finalized=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ops", type=Path, required=True)
    p.add_argument("--stage", choices=["quixbugs", "livecodebench", "classeval"], required=True)
    p.add_argument(
        "--python-bin", type=Path, default=Path("/tmp/flowpilot-code-python-20260915/bin")
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = p.parse_args()
    bundles, provenance = load_bundles(args.ops, args.stage)
    config = make_config(bundles[0], args.ops)
    root = freeze_selection(args.ops, args.stage, bundles, provenance, config)
    result = (
        validate(bundles, config, root, args.python_bin)
        if args.validate
        else run_stage(bundles, config, root, args.python_bin)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.validate and not result["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
