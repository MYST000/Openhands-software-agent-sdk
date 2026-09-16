"""Collect explicitly selected QuixBugs/LiveCodeBench tasks using OpenHands."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from .code_campaign import freeze_selection, run_stage, sha, validate
from .code_tasks import LiveCodeBenchAdapter, QuixBugsAdapter
from .config import load_config
from .controller_paths import private_controller_directory
from .sdk_provenance import check_sdk


def checked_ids(task_ids):
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Provide nonempty, unique task IDs")
    return list(task_ids)


def select_lcb_records(paths, task_ids, *, revision, checker_source):
    task_ids = checked_ids(task_ids)
    requested = set(task_ids)
    found = {}
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                task_id = str(row["question_id"])
                if task_id not in requested:
                    continue
                if task_id in found:
                    raise ValueError(f"Duplicate dataset task ID: {task_id}")
                if row["difficulty"] not in {"easy", "medium"}:
                    raise ValueError("This profile supports easy/medium tasks only")
                found[task_id] = LiveCodeBenchAdapter.from_record(
                    row, revision=revision, checker_source=checker_source
                )
    if missing := requested - found.keys():
        raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    return [found[task_id] for task_id in task_ids]


def load_selected(config, task_ids, checker_path=None):
    task_ids = checked_ids(task_ids)
    source = private_controller_directory(config.dataset.path)
    if config.dataset.kind == "quixbugs":
        actual = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        if actual != config.dataset.revision or dirty:
            raise ValueError("QuixBugs must be clean at the declared dataset revision")
        available = {
            p.stem.removeprefix("test_") for p in (source / "python_testcases").glob("test_*.py")
        }
        if missing := set(task_ids) - available:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
        bundles = [QuixBugsAdapter.load_one(source, name, revision=actual) for name in task_ids]
        provenance = {"revision": actual, "source": str(source.resolve())}
    elif config.dataset.kind == "livecodebench":
        lock = json.loads((Path(__file__).parent / "data/lcb_release_v6.lock.json").read_text())
        if config.dataset.revision != lock["revision"]:
            raise ValueError("This profile requires the pinned LiveCodeBench release_v6")
        if checker_path is None:
            raise ValueError(
                "LiveCodeBench requires --checker pointing to the pinned official checker"
            )
        checker_path = Path(checker_path)
        if sha(checker_path) != lock["checker_sha256"]:
            raise ValueError("LiveCodeBench checker SHA256 mismatch")
        paths = [source / name for name in lock["files"]]
        for path in paths:
            if sha(path) != lock["files"][path.name]:
                raise ValueError(f"LiveCodeBench source SHA256 mismatch: {path.name}")
        bundles = select_lcb_records(
            paths,
            task_ids,
            revision=config.dataset.revision,
            checker_source=checker_path.read_text(),
        )
        provenance = {
            **lock,
            "source": str(source.resolve()),
            "checker_path": str(checker_path.resolve()),
        }
    else:
        raise ValueError("code_collection currently supports quixbugs and livecodebench")
    if config.dataset.split != "dev":
        raise ValueError(
            "This collection profile is development-only; freeze a new split policy before formal training/test"
        )
    return bundles, {
        **provenance,
        "selection_rule": "explicit ordered task IDs; no score-based selection",
    }


def prepare_collection(config, task_ids, python_bin, run_id, *, checker_path=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", run_id) or ".." in run_id:
        raise ValueError("Invalid run_id; use one plain directory name")
    bundles, provenance = load_selected(config, task_ids, checker_path)
    provenance["python_bin"] = str(Path(python_bin).absolute())
    output_root = private_controller_directory(config.runtime.runs_dir, create=True)
    ops = output_root / run_id
    private_controller_directory(ops, create=True)
    root = freeze_selection(ops, config.dataset.kind, bundles, provenance, config)
    return bundles, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--python-bin",
        type=Path,
        required=True,
        help="Task venv bin directory; preserve venv symlinks",
    )
    parser.add_argument("--checker", type=Path, help="Pinned LiveCodeBench testing_util.py")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Freeze task selection without executing task code or calling a model",
    )
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Run independent reference/checker conformance checks",
    )
    mode.add_argument(
        "--run", action="store_true", help="Run selected tasks after successful validation"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    sdk = check_sdk(config.runtime)
    bundles, root = prepare_collection(
        config, args.ids, args.python_bin, args.run_id, checker_path=args.checker
    )
    if args.prepare:
        result = {
            "prepared": True,
            "root": str(root),
            "task_ids": [b.task.task_id for b in bundles],
            "sdk": sdk,
        }
    elif args.validate:
        result = validate(bundles, config, root, args.python_bin)
    else:
        result = run_stage(
            bundles, config, root, args.python_bin, require_previous_stage=False, run_id=args.run_id
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.validate and not result["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
