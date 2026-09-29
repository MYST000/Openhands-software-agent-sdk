"""Small wrapper around the pinned official SWE image builder."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .evaluation import HARNESS_COMMIT, _freeze, verify_harness
from .swe import checked_dataset_digest, image_name, load_records, select_records
from .tracing import write_json


def prepare_image_build(config, tasks, output_dir):
    if config.dataset.kind != "swe" or config.evaluation.namespace != "none":
        raise ValueError("build-images requires SWE with evaluation.namespace = 'none'")
    ids = [task.task_id for task in tasks]
    if not ids:
        raise ValueError("No tasks selected for image build")
    expected = [image_name(task, config.docker.image_template) for task in tasks]
    official = [
        f"sweb.eval.x86_64.{task_id.lower()}:{config.evaluation.image_tag}" for task_id in ids
    ]
    if expected != official:
        raise ValueError("docker.image_template must match the official local x86_64 image names")
    dataset_hash = checked_dataset_digest(config)
    rows = select_records(load_records(config.dataset.path), "instance_id", ids)
    gold_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    identity = dict(
        dataset_sha256=dataset_hash,
        gold_sha256=hashlib.sha256(gold_bytes).hexdigest(),
        image_names=expected,
        harness_commit=HARNESS_COMMIT,
    )
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    root = Path(output_dir).resolve() / ("images_" + digest)
    root.mkdir(parents=True, exist_ok=True)
    gold = root / "gold.jsonl"
    _freeze(gold, gold_bytes)
    command = [
        config.evaluation.python or sys.executable,
        "-m",
        "swebench.harness.prepare_images",
        "--dataset_name",
        str(gold),
        "--split",
        config.dataset.split,
        "--max_workers",
        str(config.evaluation.workers),
        "--namespace",
        "none",
        "--tag",
        config.evaluation.image_tag,
        "--env_image_tag",
        "latest",
    ]
    plan = dict(
        command=command,
        cwd=str(root),
        task_ids=ids,
        expected_images=expected,
        identity=identity,
        harness_path=config.evaluation.harness_path,
        build_status="pending",
        logs="builder.log",
    )
    write_json(root / "build_plan.json", plan)
    return plan


def execute_image_build(plan):
    import docker

    root = Path(plan["cwd"])
    status = {**plan, "build_status": "running"}
    write_json(root / "build_status.json", status)
    client = None
    try:
        status["harness_verified"] = verify_harness(plan)
        if (
            hashlib.sha256((root / "gold.jsonl").read_bytes()).hexdigest()
            != plan["identity"]["gold_sha256"]
        ):
            raise ValueError("Frozen image-build input checksum mismatch")
        client = docker.from_env(timeout=30)
        client.ping()
        with (root / "builder.log").open("w") as log:
            proc = subprocess.run(plan["command"], cwd=root, stdout=log, stderr=subprocess.STDOUT)
        status["returncode"] = proc.returncode
        missing, images = [], {}
        for name in plan["expected_images"]:
            try:
                found = client.images.get(name)
                images[name] = dict(id=found.id, repo_digests=found.attrs.get("RepoDigests", []))
            except docker.errors.ImageNotFound:
                missing.append(name)
        status.update(missing_images=missing, images=images)
        status["build_status"] = "ready" if proc.returncode == 0 and not missing else "build_error"
    except (Exception, KeyboardInterrupt) as exc:
        status.update(build_status="build_error", error_type=type(exc).__name__, error=str(exc))
    finally:
        if client is not None:
            client.close()
    write_json(root / "build_status.json", status)
    return status
