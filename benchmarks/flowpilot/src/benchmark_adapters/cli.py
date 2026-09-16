import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .sdk_provenance import check_sdk
from .swe import SWEAdapter, checked_dataset_digest, image_name
from .tracing import write_json


def adapter_source_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def tasks_for(config, ids=None, limit=1):
    from .code_tasks import CODE_KINDS

    if config.dataset.kind in CODE_KINDS:
        raise ValueError(
            "Use python -m benchmark_adapters.code_campaign for the local Python code profiles"
        )
    if config.dataset.kind == "swe":
        adapter = SWEAdapter
    else:
        from .retrieval import BrowseCompAdapter, HotpotAdapter

        adapter = HotpotAdapter if config.dataset.kind == "hotpot" else BrowseCompAdapter
    return adapter.load(
        config.dataset.path,
        dataset_id=config.dataset.id,
        revision=config.dataset.revision,
        split=config.dataset.split,
        ids=ids,
        limit=limit,
    )


def task_key(task_id):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,160}", task_id) and ".." not in task_id:
        return task_id
    return "task-" + hashlib.sha256(task_id.encode()).hexdigest()[:24]


def doctor(config, *, offline=False, tasks=()):
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append(dict(name=name, ok=True, detail=detail))
        except Exception as exc:
            checks.append(dict(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}"))

    def data_check():
        path = Path(config.dataset.path)
        if not path.is_file():
            raise ValueError(f"Dataset not found: {path}")
        checked_dataset_digest(config)
        return f"{path} ({path.stat().st_size} bytes)"

    def sdk_check():
        return check_sdk(config.runtime)

    check("dataset", data_check)
    check("sdk", sdk_check)
    if config.dataset.kind != "swe":
        from .retrieval import RetrievalEnvironment

        def index_check():
            env = RetrievalEnvironment(config)
            try:
                return env.prepare()
            finally:
                env.close()

        check("retrieval_index", index_check)
    if not offline:
        if config.dataset.kind == "swe":

            def docker_check():
                import docker

                client = docker.from_env(timeout=5)
                try:
                    client.ping()
                    missing = []
                    for task in tasks:
                        image = image_name(task, config.docker.image_template)
                        try:
                            client.images.get(image)
                        except docker.errors.ImageNotFound:
                            missing.append(image)
                    if missing and not config.docker.pull_missing:
                        raise ValueError(
                            "Missing task images; run build-images --execute for local images "
                            "or prepare-images for registry images: " + ", ".join(missing)
                        )
                    return "Docker reachable; missing images: " + str(len(missing))
                finally:
                    client.close()

            check("docker", docker_check)

        def llm_check():
            url = config.llm.base_url.rstrip("/") + "/models"
            request = urllib.request.Request(url)
            request.add_header(
                "Authorization", "Bearer " + os.environ.get(config.llm.api_key_env, "dummy")
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.load(response)
            names = [m["id"] for m in data.get("data", [])]
            model = config.llm.model.removeprefix("openai/")
            if model not in names:
                raise ValueError(
                    f"Model {model!r} is not advertised by /models; check served-model-name"
                )
            return {
                "model": model,
                "note": "Endpoint reachability only; tool calling must pass a pilot.",
            }

        check("llm_endpoint", llm_check)
    return dict(ok=all(c["ok"] for c in checks), runtime_checked=not offline, checks=checks)


def latest_attempt(run_dir, task_id):
    root = Path(run_dir) / "tasks" / task_key(task_id)
    paths = sorted(p for p in root.glob("attempt-*") if p.is_dir())
    return paths[-1] if paths else None


def export_run(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    kind = manifest["config"]["dataset"]["kind"]
    records = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        attempt = latest_attempt(run_dir, task_id)
        source = attempt / "submission.json" if attempt else None
        if source is not None and source.exists():
            records.append(json.loads(source.read_text()))
        elif kind == "swe":
            records.append(
                dict(
                    instance_id=task_id,
                    model_name_or_path=manifest["config"]["llm"]["model"],
                    model_patch="",
                )
            )
        elif kind == "hotpot":
            records.append(dict(task_id=task_id, answer="", sp=[]))
        else:
            records.append(
                dict(
                    query_id=task_id,
                    tool_call_counts={},
                    status="failed",
                    retrieved_docids=[],
                    result=[],
                )
            )
    out = run_dir / "submissions"
    out.mkdir(exist_ok=True)
    if kind == "swe":
        path = out / "predictions.jsonl"
        temp = path.with_suffix(".partial")
        temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
        temp.replace(path)
    elif kind == "hotpot":
        path = out / "predictions.json"
        write_json(
            path,
            {
                "answer": {r["task_id"]: r["answer"] for r in records},
                "sp": {r["task_id"]: r["sp"] for r in records},
            },
        )
    else:
        path = out / "browsecomp"
        path.mkdir(exist_ok=True)
        for record in records:
            write_json(path / (task_key(record["query_id"]) + ".json"), record)
    return path


def run_selected(config, tasks, run_id, resume=False):
    from .runner import run_task

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", run_id):
        raise ValueError("Invalid run ID")
    root = Path(config.runtime.runs_dir) / run_id
    manifest_path = root / "manifest.json"
    dataset_hash = checked_dataset_digest(config)
    source_hash = adapter_source_digest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError("Run already exists; use --resume or a new --run-id")
        if previous.get("adapter_source_sha256") != source_hash:
            raise ValueError("Resume requires identical adapter source; use a new run ID")
        if (
            previous["config_hash"] != config.fingerprint
            or previous["tasks"] != [t.to_dict() for t in tasks]
            or previous["dataset_sha256"] != dataset_hash
        ):
            raise ValueError("Resume requires identical config and task selection")
    elif resume:
        raise ValueError("Cannot resume a run without manifest.json")
    else:
        root.mkdir(parents=True)
        write_json(
            manifest_path,
            dict(
                schema_version=1,
                run_id=run_id,
                config=config.to_dict(),
                config_hash=config.fingerprint,
                dataset_sha256=dataset_hash,
                adapter_source_sha256=source_hash,
                tasks=[t.to_dict() for t in tasks],
                submission_attempt_policy="latest_attempt",
                created_at=datetime.now(UTC).isoformat(),
            ),
        )
    results = []
    for task in tasks:
        last = latest_attempt(root, task.task_id)
        if resume and last is not None and (last / "result.json").exists():
            previous = json.loads((last / "result.json").read_text())
            if previous["execution_status"] == "completed" and previous["artifact_status"] in {
                "valid",
                "empty",
            }:
                results.append(previous)
                continue
        attempt = (
            "attempt-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6]
        )
        result = run_task(
            config, task, root / "tasks" / task_key(task.task_id) / attempt, run_id=run_id
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "execution_status": result["execution_status"],
                    "artifact_status": result["artifact_status"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if result["execution_status"] == "cancelled":
            break
    export_run(root)
    write_json(
        root / "summary.json",
        dict(
            selected=len(tasks),
            attempted=len(results),
            completed=sum(r["execution_status"] == "completed" for r in results),
            valid_artifacts=sum(r["artifact_status"] == "valid" for r in results),
            evaluation_status="pending",
        ),
    )
    return root, results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pinned OpenHands benchmark adapters")
    parser.add_argument(
        "command",
        choices=[
            "list",
            "doctor",
            "prepare-images",
            "build-images",
            "run",
            "export",
            "evaluate",
            "build-index",
        ],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Default: one task, or all explicit IDs. 0 selects all.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute official evaluation/image build; otherwise write a plan",
    )
    parser.add_argument(
        "--corpus", type=Path, help="JSONL corpus for build-index; format in README"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            if args.run_dir is None:
                raise ValueError("--run-dir is required")
            print(export_run(args.run_dir))
            return 0
        if args.config is None:
            raise ValueError("--config is required")
        config = load_config(args.config)
        if args.command == "build-index":
            from .retrieval import build_index

            if args.corpus is None:
                raise ValueError("--corpus is required")
            build_index(args.corpus, config.retrieval.index_path, config.retrieval.corpus_revision)
            return 0
        if args.command == "evaluate":
            if args.run_dir is None:
                raise ValueError("--run-dir is required")
            manifest = json.loads((args.run_dir / "manifest.json").read_text())

            def actor_config(c):
                return {k: v for k, v in c.items() if k != "evaluation"}

            if actor_config(manifest["config"]) != actor_config(config.to_dict()):
                raise ValueError("Actor configuration must match the run manifest")
            if checked_dataset_digest(config) != manifest["dataset_sha256"]:
                raise ValueError("Dataset contents changed since the actor run")
            prediction = export_run(args.run_dir)
            ids = [t["task_id"] for t in manifest["tasks"]]
            if config.dataset.kind == "swe":
                from .evaluation import execute_evaluation, prepare_swe_evaluation

                plan = prepare_swe_evaluation(
                    config, args.run_dir, ids, prediction, manifest["run_id"]
                )
                status = execute_evaluation(plan) if args.execute else plan
            else:
                from .retrieval import evaluate_retrieval

                status = evaluate_retrieval(config, args.run_dir, ids, prediction, args.execute)
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 1 if status["evaluation_status"] in {"evaluator_error", "partial_error"} else 0
        limit = args.limit if args.limit is not None else (0 if args.ids else 1)
        tasks = tasks_for(config, args.ids, limit)
        if not tasks:
            raise ValueError("No tasks selected")
        if args.command == "list":
            for task in tasks:
                print(
                    json.dumps(
                        {"task_id": task.task_id, "metadata": task.public_metadata},
                        ensure_ascii=False,
                    )
                )
            return 0
        if args.command == "doctor":
            report = doctor(config, offline=args.offline, tasks=tasks)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ok"] else 2
        if args.command == "build-images":
            from .images import execute_image_build, prepare_image_build

            plan = prepare_image_build(
                config, tasks, Path(config.runtime.runs_dir) / "image-builds"
            )
            status = execute_image_build(plan) if args.execute else plan
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 1 if status["build_status"] == "build_error" else 0
        if args.command == "prepare-images":
            if config.dataset.kind != "swe":
                raise ValueError("prepare-images is only for SWE")
            if config.evaluation.namespace == "none":
                raise ValueError("Use build-images --execute for local SWE images")
            import docker

            client = docker.from_env(timeout=600)
            try:
                client.ping()
                for task in tasks:
                    name = image_name(task, config.docker.image_template)
                    try:
                        client.images.get(name)
                        print("EXISTS " + name, flush=True)
                    except docker.errors.ImageNotFound:
                        print("PULL " + name, flush=True)
                        client.images.pull(name)
            finally:
                client.close()
            return 0
        report = doctor(config, tasks=tasks)
        if not report["ok"]:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
        run_id = (
            args.run_id or datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
        )
        root, results = run_selected(config, tasks, run_id, args.resume)
        print("RUN_DIRECTORY " + str(root))
        return (
            0
            if all(
                r["execution_status"] == "completed" and r["artifact_status"] in {"valid", "empty"}
                for r in results
            )
            else 1
        )
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
