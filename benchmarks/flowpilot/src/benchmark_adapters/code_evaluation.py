"""Post-actor evaluation in a fresh low-privilege process/workspace.

Private cases never return to an active actor. This is local evaluation with upstream
checks, not a namespace sandbox or the original single-generation leaderboard protocol.
"""

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .code_tasks import evaluation_files, materialize_workspace
from .local_environment import LocalPilotEnvironment
from .tracing import write_json


def evaluate_code(bundle, code, config, output_dir, *, python_bin, uid=63112, timeout=None):
    if timeout is None:
        timeout = (
            7 * len(json.loads(bundle.private["sample"]["input_output"])["inputs"]) + 5
            if bundle.kind == "livecodebench"
            else 180
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    files, command = evaluation_files(bundle, code)
    root = Path(tempfile.mkdtemp(prefix="flowpilot-code-local-eval-", dir="/tmp"))
    environment = None
    try:
        task = materialize_workspace(bundle, root / "repo", files=files)
        environment = LocalPilotEnvironment(
            config, task, output_dir / "environment", task_root=root, python_bin=python_bin, uid=uid
        )
        environment.prepare()
        result = environment.execute(command, timeout=timeout, max_bytes=200000)
        status = "evaluated"
        details = {}
        if result.timed_out:
            passed = False
            details = {"failure_kind": "submission_timeout", "duration_right_censored": True}
        elif bundle.kind in {"livecodebench", "classeval"}:
            lines = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("FLOWPILOT_EVAL_JSON=")
            ]
            if result.exit_code or len(lines) != 1:
                passed = False
                status = "evaluation_error"
            else:
                details = json.loads(lines[0].split("=", 1)[1])
                passed = details["passed"] is True
        else:
            passed = result.exit_code == 0
            text = result.stdout + "\n" + result.stderr
            if bundle.kind == "quixbugs":
                # pytest collection errors are submission errors for malformed actor code;
                # reference validation must pass before actor tasks begin.
                if result.exit_code not in {0, 1, 2}:
                    status = "evaluation_error"
                for label in ("passed", "failed", "error", "skipped"):
                    matches = re.findall(r"(\d+) " + label, text)
                    if matches:
                        details[label + "_count"] = int(matches[-1])
            else:
                match = re.search(r"Ran (\d+) tests?", text)
                details["test_count"] = int(match[1]) if match else 0
                if passed and not details["test_count"]:
                    passed, status = False, "evaluation_error"
        report = dict(
            task_id=bundle.task.task_id,
            benchmark=bundle.kind,
            status=status,
            passed=passed,
            protocol=bundle.task.public_metadata["protocol"],
            code_sha256=hashlib.sha256(code.encode()).hexdigest(),
            evaluation_inputs_sha256=hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()
            ).hexdigest(),
            command=command,
            command_result=asdict(result),
            details=details,
            actor_feedback=False,
            backend="local-process-pilot",
        )
    except Exception as exc:
        report = dict(
            task_id=bundle.task.task_id,
            benchmark=bundle.kind,
            status="evaluation_error",
            passed=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        if environment is not None:
            environment.close()
        shutil.rmtree(root)
    write_json(output_dir / "evaluation.json", report)
    return report
