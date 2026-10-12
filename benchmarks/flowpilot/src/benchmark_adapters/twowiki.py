"""2WikiMultiHopQA identities, public prompts and pinned official evaluation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .contracts import Task
from .retrieval import HotpotAdapter, parse_hotpot_answer
from .swe import checked_dataset_digest, load_records, select_records
from .tracing import write_json

SCORER_COMMIT = "13800e5be57df1b4040b9b1588c6c811779e69e9"
SCORER_SHA256 = "854b03c487608286d6381379eb32e46d9b845493527d1fe1951d0bf4ec923a40"


class TwoWikiAdapter(HotpotAdapter):
    @staticmethod
    def from_record(row, *, dataset_id, revision, split):
        task = HotpotAdapter.from_record(row, dataset_id=dataset_id, revision=revision, split=split)
        return Task(
            dataset_id,
            revision,
            split,
            task.task_id,
            task.instruction + '\nAlso include "evidences": a list of [subject, relation, object] '
            "string triples supported by the retrieved documents.",
        )


def export_answer(task, text, environment):
    parsed = parse_hotpot_answer(text, environment)
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    evidence = json.loads(text).get("evidences")
    if not isinstance(evidence, list) or any(
        not isinstance(triple, list)
        or len(triple) != 3
        or any(not isinstance(value, str) for value in triple)
        for triple in evidence
    ):
        raise ValueError("2Wiki finish requires evidences as string triples")
    return {"task_id": task.task_id, **parsed, "evidence": evidence}


def evaluate_twowiki(config, run_dir, ids, prediction, execute):
    checked_dataset_digest(config)
    script = (
        Path(config.evaluation.twowiki_script)
        if config.evaluation.twowiki_script
        else Path(__file__).resolve().parents[2] / "vendor/twowiki/2wikimultihop_evaluate.py"
    )
    if hashlib.sha256(script.read_bytes()).hexdigest() != SCORER_SHA256:
        raise ValueError("2Wiki evaluator differs from the pinned official source")
    rows = select_records(load_records(config.dataset.path), "_id", ids)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    gold = root / "gold.json"
    write_json(gold, rows)
    plan = {
        "evaluator": "twowiki",
        "evaluator_identity": {
            "commit": SCORER_COMMIT,
            "script_sha256": SCORER_SHA256,
            "variant": "original evidence-triples scorer; no answer aliases",
        },
        "command": [
            config.evaluation.python or sys.executable,
            str(script),
            str(prediction),
            str(gold),
        ],
        "task_ids": ids,
        "metric_scale": "0..100",
        "evaluation_status": "pending",
        "supporting_fact_scope": (
            "Page-local live sentence IDs differ from the released corpus; "
            "answer EM/F1 is primary and live supporting-fact scores are diagnostic."
            if config.retrieval.backend == "wikipedia_live"
            else "Original fixed-corpus sentence IDs"
        ),
    }
    write_json(root / "evaluation_plan.json", plan)
    if execute:
        proc = subprocess.run(
            plan["command"],
            text=True,
            capture_output=True,
            timeout=config.evaluation.timeout,
            check=False,
        )
        (root / "evaluator.log").write_text(proc.stdout + proc.stderr)
        proc.check_returncode()
        metrics = json.loads(proc.stdout[proc.stdout.rfind("{") :])
        plan.update(evaluation_status="scored", metrics=metrics)
        write_json(root / "evaluation_status.json", plan)
    return plan
