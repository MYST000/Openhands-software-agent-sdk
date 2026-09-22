"""Prepare, validate, run and evaluate one frozen mixed task queue.

Task metadata selects an adapter. Every adapter may occupy all four slots; no
slot is reserved for a benchmark. Official scoring runs after actor collection.
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from .adapter_registry import ADAPTERS, load_adapter_tasks, resolve_adapter
from .cli import adapter_source_digest, task_key
from .code_campaign import freeze_selection, runtime_fingerprint, sha, validate
from .code_evaluation import evaluate_code
from .code_tasks import CodeBundle, materialize_workspace
from .config import Config, load_config
from .contracts import Task
from .controller_paths import private_controller_directory
from .local_environment import LocalPilotEnvironment
from .parallel_runtime import run_parallel
from .prediction_export import build_prediction_rows, write_prediction_dataset
from .research_splits import apply_research_split
from .runner import run_task
from .sdk_provenance import check_sdk, sdk_source_state
from .tracing import write_json

RESEARCH_SPLITS = {"historical_dev", "fit", "tune", "calibration", "test"}


def worker_uids(base, slot):
    if type(base) is not int or type(slot) is not int or slot < 0:
        raise ValueError("UID base and worker slot must be integers")
    actor, evaluator = base + 2 * slot, base + 2 * slot + 1
    if not 60000 <= actor < evaluator < 65000:
        raise ValueError("Each worker needs two distinct dedicated UIDs in 60000..64999")
    return actor, evaluator


def _private_output(path):
    path = Path(path).resolve()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o077:
        raise ValueError("Campaign output must be owned by controller with mode 0700")
    return path


def _config(raw):
    defaults = Config()
    return Config(**{name: type(getattr(defaults, name))(**value) for name, value in raw.items()})


def _payload(job):
    path = Path(job["input_path"])
    if sha(path) != job["input_sha256"]:
        raise ValueError("Frozen task input changed")
    return json.loads(path.read_text())


def _bundle(payload):
    row = payload["bundle"]
    return CodeBundle(
        kind=row["kind"],
        task=Task(**row["task"]),
        public_files=row["public_files"],
        solution_path=row["solution_path"],
        private=row["private"],
    )


def _research_tasks(pairs, entry, config, base):
    split = entry.get("research_split", "historical_dev")
    if split not in RESEARCH_SPLITS:
        raise ValueError("Unknown research_split")
    if config.dataset.kind == "quixbugs" and split != "historical_dev":
        raise ValueError("QuixBugs40 remains historical development/auxiliary data")
    manifest = entry.get("split_manifest")
    expected = entry.get("split_manifest_sha256")
    if bool(manifest) != bool(expected):
        raise ValueError("A research manifest requires its exact SHA256")
    groups = {}
    if manifest and config.dataset.kind == "livecodebench":
        assigned, _ = apply_research_split(
            [bundle for _, bundle in pairs], base / manifest, expected, split
        )
        return [(bundle.task, bundle) for bundle in assigned]
    if manifest:
        content = (base / manifest).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Research manifest checksum mismatch")
        rows, group_splits, content_splits = {}, {}, {}
        for line in content.decode().splitlines():
            row = json.loads(line)
            key = (row["dataset_id"], row["dataset_revision"], row["task_id"])
            if key in rows or row["research_split"] not in RESEARCH_SPLITS:
                raise ValueError("Duplicate identity or invalid split in research manifest")
            if type(row["historically_exposed"]) is not bool:
                raise ValueError("historically_exposed must be a boolean")
            if row["historically_exposed"] and row["research_split"] != "historical_dev":
                raise ValueError("Exposed task cannot be relabeled as an unseen task")
            for mapping, identity in (
                (group_splits, row["task_group_id"]),
                (content_splits, row["instruction_sha256"]),
            ):
                if identity in mapping and mapping[identity] != row["research_split"]:
                    raise ValueError("Task group or identical task text crosses research splits")
                mapping[identity] = row["research_split"]
            rows[key] = row
        for task, _ in pairs:
            row = rows[(task.dataset_id, task.revision, task.task_id)]
            if (
                row["research_split"] != split
                or row["instruction_sha256"]
                != hashlib.sha256(task.instruction.encode()).hexdigest()
            ):
                raise ValueError("Research split/content differs from selected task")
            groups[task.task_id] = row["task_group_id"]
    elif split != "historical_dev":
        raise ValueError("Formal research splits require a frozen, hash-checked manifest")
    assigned = []
    for task, bundle in pairs:
        task = replace(
            task,
            split=split,
            public_metadata={
                **task.public_metadata,
                "research_split": split,
                "task_group_id": groups.get(task.task_id, f"{task.dataset_id}:{task.task_id}"),
            },
        )
        assigned.append((task, replace(bundle, task=task) if bundle else None))
    return assigned


def prepare_campaign(campaign_path):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    allowed = {
        "version",
        "run_id",
        "runs_dir",
        "concurrency",
        "adapter_concurrency",
        "seed",
        "queue_policy",
        "partition",
        "episode_id",
        "replica_id",
        "uid_base",
        "entries",
    }
    if campaign.keys() - allowed or campaign.get("version") != 1:
        raise ValueError("Unsupported campaign fields/version")
    run_id = campaign["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id) or ".." in run_id:
        raise ValueError("Invalid run_id")
    concurrency = campaign.get("concurrency", 4)
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be in 1..64")
    caps = campaign.get("adapter_concurrency", {kind: 4 for kind in ADAPTERS})
    if set(caps) != set(ADAPTERS) or any(
        type(v) is not int or not 1 <= v <= 64 for v in caps.values()
    ):
        raise ValueError("Declare positive concurrency caps for all four adapters")
    partition = campaign.get("partition", "historical_dev")
    accepted = {"historical_dev", "fit"} if partition == "fit_plus_aux" else {partition}
    if not accepted <= RESEARCH_SPLITS:
        raise ValueError("Unknown campaign partition")
    uid_base = campaign.get("uid_base", 63200)
    worker_uids(uid_base, concurrency - 1)
    base = campaign_path.parent
    output = _private_output(base / campaign["runs_dir"])
    root = output / run_id
    root.mkdir(mode=0o700, exist_ok=False)
    (root / "inputs").mkdir(mode=0o700)
    jobs, seen, job_ids = [], set(), set()
    for entry_index, entry in enumerate(campaign["entries"]):
        if entry.keys() - {
            "config",
            "task_ids",
            "checker",
            "python_bin",
            "research_split",
            "split_manifest",
            "split_manifest_sha256",
        }:
            raise ValueError("Unknown campaign entry field")
        config = load_config(base / entry["config"])
        registration = resolve_adapter(config)
        sdk_source_state(config.runtime.sdk_path, config.runtime.sdk_commit)
        if not entry["task_ids"] or len(set(entry["task_ids"])) != len(entry["task_ids"]):
            raise ValueError("Explicit unique task_ids are required")
        checker = base / entry["checker"] if entry.get("checker") else None
        pairs, provenance = load_adapter_tasks(config, entry["task_ids"], checker_path=checker)
        pairs = _research_tasks(pairs, entry, config, base)
        python_bin = (
            str((base / entry["python_bin"]).absolute()) if entry.get("python_bin") else None
        )
        if registration.code_workspace and python_bin is None:
            raise ValueError("Code entries require an explicit task python_bin")
        for task, bundle in pairs:
            identity = (task.dataset_id, task.revision, task.task_id)
            if identity in seen or task.split not in accepted:
                raise ValueError("Repeated task or incompatible split in campaign")
            seen.add(identity)
            job_id = registration.kind + "--" + task_key(task.task_id)
            if job_id in job_ids:
                raise ValueError("One campaign cannot mix revisions of the same task identity")
            job_ids.add(job_id)
            payload = {
                "config": config.to_dict(),
                "task": task.to_dict(),
                "bundle": asdict(bundle) if bundle else None,
                "provenance": provenance,
                "entry_index": entry_index,
                "python_bin": python_bin,
                "registration": asdict(registration),
            }
            path = root / "inputs" / (job_id + ".json")
            write_json(path, payload)
            path.chmod(0o600)
            jobs.append(
                {
                    "job_id": job_id,
                    "adapter": registration.kind,
                    "input_path": str(path),
                    "input_sha256": sha(path),
                    "attempt_dir": str(root / "tasks" / job_id / "attempt-001"),
                    "uid_base": uid_base,
                    "controller_timeout_s": config.runtime.task_timeout + 120,
                    "trace_context": {
                        "episode_id": campaign.get("episode_id", run_id),
                        "replica_id": campaign.get("replica_id", "replica-0"),
                        "research_split": task.split,
                        "task_group_id": task.public_metadata["task_group_id"],
                        "campaign_partition": partition,
                    },
                    "run_id": run_id,
                }
            )
    if not jobs:
        raise ValueError("Campaign queue must be nonempty")
    policy = campaign.get("queue_policy", "seeded_shuffle")
    if policy == "seeded_shuffle":
        random.Random(campaign.get("seed", 20260921)).shuffle(jobs)
    elif policy != "ordered":
        raise ValueError("queue_policy must be ordered or seeded_shuffle")
    for position, job in enumerate(jobs):
        job["trace_context"]["queue_position"] = position
    manifest = {
        "schema_version": 1,
        "campaign": campaign,
        "jobs": jobs,
        "concurrency": concurrency,
        "adapter_concurrency": caps,
        "adapter_source_sha256": adapter_source_digest(),
        "queue_policy": policy + "; first eligible job backfills a free slot",
        "intra_task_tool_concurrency": 1,
        "single_attempt": True,
        "evaluators_run_after_collection": True,
    }
    write_json(root / "campaign.json", manifest)
    return root


def _checked_manifest(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "campaign.json").read_text())
    if manifest["adapter_source_sha256"] != adapter_source_digest():
        raise ValueError("Adapter changed since preparation; prepare a new campaign")
    for job in manifest["jobs"]:
        _payload(job)
    return root, manifest


def _isolation_preflight(root, manifest):
    code = [job for job in manifest["jobs"] if ADAPTERS[job["adapter"]].code_workspace]
    if not code:
        return
    if os.geteuid() != 0:
        raise ValueError(
            "Code collection requires root + dedicated UID isolation. Same-UID shell fallback is intentionally unavailable; configure the isolated backend before running code tasks."
        )
    uids = [
        uid
        for slot in range(manifest["concurrency"])
        for uid in worker_uids(code[0]["uid_base"], slot)
    ]
    uids.extend([63111, 63112])
    private_controller_directory(root, task_uids=uids)
    sources = {Path(_payload(job)["config"]["dataset"]["path"]) for job in manifest["jobs"]}
    for source in sources:
        private_controller_directory(source if source.is_dir() else source.parent, task_uids=uids)


def validate_campaign(root):
    root, manifest = _checked_manifest(root)
    _isolation_preflight(root, manifest)
    destination = root / "validation.json"
    if destination.exists():
        raise ValueError("Validation already exists; do not overwrite")
    checks, groups = [], {}
    for job in manifest["jobs"]:
        payload = _payload(job)
        groups.setdefault(payload["entry_index"], []).append(payload)
    for index, payloads in groups.items():
        payload = payloads[0]
        config = _config(payload["config"])
        check_sdk(config.runtime)
        if payload["bundle"] is not None:
            ops = root / "validation" / str(index)
            ops.mkdir(parents=True)
            bundles = [_bundle(row) for row in payloads]
            selection = freeze_selection(
                ops, config.dataset.kind, bundles, payload["provenance"], config
            )
            report = validate(bundles, config, selection, payload["python_bin"])
            checks.append(
                {
                    "entry": index,
                    "ok": report["all_checks_passed"],
                    "runtime_fingerprint": runtime_fingerprint(payload["python_bin"]),
                    "python_bin": payload["python_bin"],
                }
            )
        else:
            from .native_browsecomp import create_retrieval_environment

            env = create_retrieval_environment(config)
            try:
                metadata = env.prepare()
                checks.append({"entry": index, "ok": True, "metadata": metadata})
            finally:
                env.close()
    report = {
        "all_checks_passed": bool(checks) and all(row["ok"] for row in checks),
        "campaign_sha256": sha(root / "campaign.json"),
        "checks": checks,
    }
    write_json(destination, report)
    return report


def collect_job(job, context):
    payload = _payload(job)
    config, task = _config(payload["config"]), Task(**payload["task"])
    attempt = Path(job["attempt_dir"])
    trace_context = {**job["trace_context"], "worker_slot": context.slot}
    environment, task_root = None, None
    try:
        if payload["bundle"] is not None:
            bundle = _bundle(payload)
            task_root = Path(tempfile.mkdtemp(prefix="flowpilot-code-local-actor-", dir="/tmp"))
            task = materialize_workspace(bundle, task_root / "repo")
            actor_uid, _ = worker_uids(job["uid_base"], context.slot)
            environment = LocalPilotEnvironment(
                config,
                task,
                attempt / "artifacts",
                task_root=task_root,
                python_bin=payload["python_bin"],
                uid=actor_uid,
            )
        else:
            from .native_browsecomp import create_retrieval_environment

            environment = create_retrieval_environment(
                config, verified_index=job.get("verified_retrieval_metadata")
            )
        context.set_phase("actor")
        result = run_task(
            config,
            task,
            attempt,
            environment=environment,
            run_id=job["run_id"],
            trace_context=trace_context,
            load_monitor=context,
        )
        rows = build_prediction_rows(attempt)
        (attempt / "prediction_samples.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        write_json(
            attempt / "trace_audit.json",
            {"audit_passed": True, "request_count": len(rows), "schema_version": 3},
        )
        return result
    finally:
        if environment is not None:
            environment.close()
        if task_root is not None:
            shutil.rmtree(task_root)


def run_campaign(root):
    root, manifest = _checked_manifest(root)
    _isolation_preflight(root, manifest)
    validation = json.loads((root / "validation.json").read_text())
    if not validation["all_checks_passed"] or validation["campaign_sha256"] != sha(
        root / "campaign.json"
    ):
        raise ValueError("Campaign validation gate failed")
    for check in validation["checks"]:
        if (
            "runtime_fingerprint" in check
            and runtime_fingerprint(check["python_bin"]) != check["runtime_fingerprint"]
        ):
            raise ValueError("Task Python environment changed since validation")
    for job in manifest["jobs"]:
        check_sdk(_config(_payload(job)["config"]).runtime)
    marker = root / "collection_started.json"
    with marker.open("x") as f:
        json.dump({"campaign_sha256": sha(root / "campaign.json")}, f)

    def received(record):
        write_json(root / "receipts" / (record["job_id"] + ".json"), record)

    runtime_metadata = {
        check["entry"]: check["metadata"] for check in validation["checks"] if "metadata" in check
    }
    jobs = [
        {
            **job,
            "verified_retrieval_metadata": runtime_metadata.get(_payload(job)["entry_index"]),
        }
        for job in manifest["jobs"]
    ]
    try:
        report = run_parallel(
            jobs,
            collect_job,
            concurrency=manifest["concurrency"],
            adapter_limits=manifest["adapter_concurrency"],
            on_result=received,
        )
        report["status"] = "collected"
        write_json(root / "collection_summary.json", report)
        write_prediction_dataset(
            [job["attempt_dir"] for job in manifest["jobs"]], root / "prediction_dataset"
        )
        return report
    except BaseException as exc:
        write_json(
            root / "collection_summary.json",
            {
                "status": "interrupted_or_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "receipts_retained": True,
                "residual_process_audit_required": True,
                "possible_task_uids": sorted(
                    {
                        uid
                        for job in manifest["jobs"]
                        if ADAPTERS[job["adapter"]].code_workspace
                        for slot in range(manifest["concurrency"])
                        for uid in worker_uids(job["uid_base"], slot)
                    }
                ),
            },
        )
        raise


def evaluate_campaign(root):
    root, manifest = _checked_manifest(root)
    if json.loads((root / "collection_summary.json").read_text())["status"] != "collected":
        raise ValueError("Evaluate only after the collection window has finished")
    _isolation_preflight(root, manifest)
    results = []
    for job in manifest["jobs"]:
        payload = _payload(job)
        config = _config(payload["config"])
        attempt = Path(job["attempt_dir"])
        output = attempt / "evaluation"
        if output.exists():
            raise ValueError("Evaluation already exists; never overwrite")
        if payload["bundle"] is not None:
            code = attempt / "artifacts/solution.py"
            if not code.is_file():
                report = {"status": "no_submission", "passed": False}
                write_json(output / "evaluation.json", report)
            else:
                _, uid = worker_uids(job["uid_base"], 0)
                report = evaluate_code(
                    _bundle(payload),
                    code.read_text(),
                    config,
                    output,
                    python_bin=payload["python_bin"],
                    uid=uid,
                )
        else:
            from .retrieval import evaluate_retrieval

            task = Task(**payload["task"])
            submitted = attempt / "submission.json"
            value = json.loads(submitted.read_text()) if submitted.exists() else None
            output.mkdir(parents=True)
            if job["adapter"] == "hotpot":
                predictions = output / "predictions.json"
                write_json(
                    predictions,
                    {
                        "answer": {task.task_id: value["answer"] if value else ""},
                        "sp": {task.task_id: value["sp"] if value else []},
                    },
                )
            else:
                predictions = output / "predictions"
                predictions.mkdir()
                write_json(
                    predictions / (task_key(task.task_id) + ".json"),
                    value
                    or {
                        "query_id": task.task_id,
                        "status": "failed",
                        "retrieved_docids": [],
                        "result": [],
                    },
                )
            report = evaluate_retrieval(
                config, output, [task.task_id], predictions, execute=job["adapter"] == "hotpot"
            )
        results.append({"job_id": job["job_id"], "evaluation": report})
    write_json(root / "evaluation_summary.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "validate", "run", "evaluate"])
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.campaign is None:
            parser.error("prepare requires --campaign")
        print(json.dumps({"prepared": str(prepare_campaign(args.campaign))}))
    else:
        if args.run_dir is None:
            parser.error("action requires --run-dir")
        result = {
            "validate": validate_campaign,
            "run": run_campaign,
            "evaluate": evaluate_campaign,
        }[args.action](args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
