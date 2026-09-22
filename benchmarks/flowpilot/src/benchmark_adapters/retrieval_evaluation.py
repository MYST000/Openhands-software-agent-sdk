"""Pinned native evaluation sources and machine-readable Hotpot results."""

import ast
import hashlib
import subprocess
from pathlib import Path

HOTPOT_COMMIT = "3635853403a8735609ee997664e1528f4480762a"
HOTPOT_SCORER_SHA256 = "d35fc91a6db21d791dbdda11daf3856e9359f5701d54e3eefba20d88fecc02c0"
BROWSECOMP_COMMIT = "046949032b0328319cc9a02663a759ec601d9402"


def evaluator_source(config):
    if config.dataset.kind == "hotpot":
        script = (
            Path(config.evaluation.hotpot_script).resolve()
            if config.evaluation.hotpot_script
            else Path(__file__).resolve().parents[2] / "vendor/hotpot/hotpot_evaluate_v1.py"
        )
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        if digest != HOTPOT_SCORER_SHA256:
            raise ValueError("Hotpot evaluator differs from the pinned official source")
        return script, {"commit": HOTPOT_COMMIT, "script_sha256": digest}
    repo = Path(config.evaluation.browsecomp_repo).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != BROWSECOMP_COMMIT:
        raise ValueError("BrowseComp evaluator repository must use the pinned official commit")
    identity = {"commit": commit}
    for relative in ("scripts_evaluation/evaluate_run.py", "topics-qrels/qrel_evidence.txt"):
        data = (repo / relative).read_bytes()
        official = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{relative}"])
        if data != official:
            raise ValueError(f"Official evaluation source was modified: {relative}")
        identity[relative] = hashlib.sha256(data).hexdigest()
    return repo / "scripts_evaluation/evaluate_run.py", identity


def hotpot_metrics(log):
    for line in reversed(Path(log).read_text().splitlines()):
        if line.startswith("{"):
            result = ast.literal_eval(line)
            expected = {
                "em",
                "f1",
                "prec",
                "recall",
                "sp_em",
                "sp_f1",
                "sp_prec",
                "sp_recall",
                "joint_em",
                "joint_f1",
                "joint_prec",
                "joint_recall",
            }
            if not isinstance(result, dict) or set(result) != expected:
                break
            if not all(
                isinstance(value, (int, float)) and 0 <= value <= 1 for value in result.values()
            ):
                break
            return result
    raise ValueError("Official Hotpot evaluator did not produce its expected metrics")
