"""Register experiment deadlines before the SDK registers its conversation."""

import hashlib
import json
import math
import os
from datetime import timedelta
from pathlib import Path

import httpx


def register_workflow_slo(config, task, adapter, *, conversation_id, started_at):
    profile_value = os.environ.get("FLOWPILOT_EXPERIMENT_PROFILE")
    if not profile_value:
        return None
    profile_path = Path(profile_value)
    workload = json.loads(profile_path.read_text())["workload"]
    baseline_value = workload.get("baseline_latency_path")
    if not baseline_value:
        return None
    baseline_path = profile_path.parent / baseline_value
    content = baseline_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != workload["baseline_latency_sha256"]:
        raise ValueError("Workflow SLO baseline checksum mismatch")
    baseline = json.loads(content)
    if baseline["schema_version"] != 1 or baseline["unit"] != "seconds":
        raise ValueError("Unsupported workflow SLO baseline format")
    matches = [
        row
        for row in baseline["tasks"]
        if (row["dataset_id"], row["dataset_revision"], row["task_id"])
        == (config.dataset.id, config.dataset.revision, task.task_id)
    ]
    if len(matches) != 1:
        raise ValueError("Workflow SLO requires exactly one matching historical task")
    row = matches[0]
    if row["instruction_sha256"] != hashlib.sha256(task.instruction.encode()).hexdigest():
        raise ValueError("Workflow SLO task instruction differs from historical baseline")
    seconds = row["baseline_seconds"]
    multiplier = workload["slo_multiplier"]
    if not all(math.isfinite(value) and value > 0 for value in (seconds, multiplier)):
        raise ValueError("Workflow SLO duration and multiplier must be finite and positive")
    budget = seconds * multiplier
    payload = {
        "job_id": adapter.job_id or f"job-{conversation_id}",
        "root_conversation_id": adapter.root_conversation_id or str(conversation_id),
        "deployment_id": adapter.deployment_id,
        "namespace_id": adapter.namespace_id,
        "workflow_started_at": started_at.isoformat(),
        "deadline": (started_at + timedelta(seconds=budget)).isoformat(),
    }
    with httpx.Client(timeout=adapter.timeout, trust_env=False) as client:
        response = client.post(
            adapter.control_base_url + "/flowpilot/v1/jobs",
            headers={"x-flowpilot-api-key": adapter.api_key},
            json=payload,
        )
        response.raise_for_status()
    if response.json()["job_id"] != payload["job_id"]:
        raise ValueError("Workflow SLO registration returned a different job")
    return {
        **payload,
        "conversation_id": str(conversation_id),
        "baseline_seconds": seconds,
        "slo_multiplier": multiplier,
        "budget_seconds": budget,
        "baseline_sha256": digest,
        "historical_result_path": row["result_path"],
        "historical_result_sha256": row["result_sha256"],
    }
