"""Public task adapters for small Python benchmarks, independent of the SWE contract.

CodeBundle.private is controller-only. Never serialize the bundle into actor artifacts.
"""

import base64
import copy
import hashlib
import io
import json
import pickle
import shlex
import subprocess
import zlib
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from .contracts import Task

CODE_KINDS = {"quixbugs", "livecodebench", "classeval"}


@dataclass
class CodeBundle:
    kind: str
    task: Task
    public_files: dict[str, str]
    solution_path: str
    private: dict


def safe_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in {".", "..", ".git"} for x in p.parts):
        raise ValueError(f"Unsafe workspace path: {name}")
    return p


def write_files(repo, files):
    repo = Path(repo)
    for name, content in files.items():
        safe_relative(name)
        dest = repo / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # All callers create a fresh controller-owned root; reject pre-existing symlinks anyway.
        if dest.is_symlink() or not dest.resolve().is_relative_to(repo.resolve()):
            raise ValueError("Unsafe workspace destination")
        dest.write_text(content)


def materialize_workspace(bundle, repo, *, files=None):
    repo = Path(repo)
    repo.mkdir(parents=True, exist_ok=False)
    write_files(repo, bundle.public_files if files is None else files)
    (repo / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n*.pyc\n")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "FlowPilot experiment")
    git("config", "user.email", "experiment@localhost.invalid")
    git("add", ".")
    git("commit", "-qm", "Frozen public task baseline")
    return replace(
        bundle.task,
        public_metadata={
            **bundle.task.public_metadata,
            "base_commit": git("rev-parse", "HEAD"),
            "solution_path": bundle.solution_path,
        },
    )


def export_code(environment, solution_path):
    safe_relative(solution_path)
    script = """import pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
p=root/sys.argv[2]
if not p.resolve().is_relative_to(root): raise ValueError('Export escapes repository')
for part in [p,*p.parents]:
    if part==root: break
    if part.is_symlink(): raise ValueError('Symlink submission rejected')
if not p.is_file() or p.stat().st_size>500000: raise ValueError('Missing/oversized submission')
sys.stdout.write(p.read_text())
"""
    command = (
        "python3 -I -c "
        + shlex.quote(script)
        + " "
        + shlex.quote(environment.repo_dir)
        + " "
        + shlex.quote(solution_path)
    )
    result = environment.execute(command, timeout=30, max_bytes=600000)
    if result.exit_code or result.timed_out or result.truncated:
        raise RuntimeError("Code export failed: " + result.stderr[-500:])
    return result.stdout


class QuixBugsAdapter:
    @staticmethod
    def load_one(repo, name, *, revision):
        safe_relative(name)
        if "/" in name:
            raise ValueError("QuixBugs task must be a program name")
        repo = Path(repo)
        solution_path = f"python_programs/{name}.py"
        test_path = f"python_testcases/test_{name}.py"
        paths = [solution_path, test_path, "conftest.py", "python_testcases/load_testdata.py"]
        for optional in (
            "python_programs/node.py",
            "python_testcases/node.py",
            f"json_testcases/{name}.json",
        ):
            if (repo / optional).is_file():
                paths.append(optional)
        files = {p: (repo / p).read_text() for p in paths}
        command = f"python -m pytest -q {shlex.quote(test_path)}"
        instruction = (
            f"Repair the buggy Python function in {solution_path}. Inspect the source and supplied "
            f"developer tests to understand the intended behavior. Run tests with: {command}. "
            "You may create your own additional checks. Modify only the target implementation for "
            "the submitted solution; changes to supplied tests/helpers are not submitted. "
            "Run relevant checks after your change and finish when done. The supplied tests are "
            "public development tests, not a complete proof of correctness."
        )
        task = Task(
            "jkoppel/QuixBugs",
            revision,
            "dev",
            name,
            instruction,
            {
                "protocol": "quixbugs-public-tests-repair-v1",
                "solution_path": solution_path,
                "test_visibility": "upstream supplied tests public; no unseen-test claim",
            },
        )
        return CodeBundle(
            "quixbugs",
            task,
            files,
            solution_path,
            {
                "evaluation_files": copy.deepcopy(files),
                "command": command,
                "reference_code": (repo / f"correct_python_programs/{name}.py").read_text(),
            },
        )


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Executable objects not allowed in encoded test data")


def decode_lcb_cases(value):
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        data = zlib.decompress(base64.b64decode(value))
        return json.loads(_DataOnlyUnpickler(io.BytesIO(data)).load())


LCB_DRIVER = """import json,sys
if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(50000)
from pathlib import Path
from testing_util import run_test
sample=json.loads(Path('cases.json').read_text())
code=Path('solution.py').read_text()
# The upstream checker mutates process globals. Invoke only in this disposable subprocess.
results,metadata=run_test(sample,test=code,timeout=6)
passed=bool(results) and len(results)==len(json.loads(sample['input_output'])['inputs']) and all(x is True or x==1 for x in results)
print('FLOWPILOT_EVAL_JSON='+json.dumps({'passed':passed,'results':results,'metadata':metadata},default=str))
"""


class LiveCodeBenchAdapter:
    @staticmethod
    def from_record(row, *, revision, checker_source):
        public = decode_lcb_cases(row["public_test_cases"])
        private = decode_lcb_cases(row["private_test_cases"])
        metadata = (
            json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        )
        if not public or not private:
            raise ValueError("LCB pilot requires nonempty public and private tests")
        # This initial profile deliberately supports stdin Python tasks only.
        if metadata.get("func_name") or any(t["testtype"] != "stdin" for t in public + private):
            raise ValueError("This LCB profile supports stdin tasks only")

        def sample(cases):
            return {
                "input_output": json.dumps(
                    {
                        "inputs": [t["input"] for t in cases],
                        "outputs": [t["output"] for t in cases],
                        "fn_name": None,
                    }
                )
            }

        public_sample = sample(public)
        files = {
            "solution.py": "# Implement the solution: read stdin and write stdout.\n",
            "public_cases.json": json.dumps(public, indent=2),
            "cases.json": json.dumps(public_sample),
            "testing_util.py": checker_source,
            "check_public.py": LCB_DRIVER,
        }
        instruction = (
            f"{row['question_title']}\n\n{row['question_content']}\n\n"
            "Implement a standalone Python stdin/stdout solution in solution.py. "
            "The public examples are in public_cases.json. Check them with "
            "`python check_public.py`; you may add your own tests. Hidden tests are "
            "unavailable and their feedback will not be given during this task. "
            "Only solution.py is submitted. Run useful checks and then call finish."
        )
        task = Task(
            "livecodebench/code_generation_lite",
            revision,
            "dev",
            str(row["question_id"]),
            instruction,
            {
                "protocol": "lcb-release-v6-stdin-python-agentic-v1",
                "upstream_split": "test",
                "pilot_split": "development-only",
                "difficulty": row["difficulty"],
                "platform": row["platform"],
                "contest_date": row["contest_date"],
                "statement_sha256": hashlib.sha256(row["question_content"].encode()).hexdigest(),
                "solution_path": "solution.py",
                "test_visibility": "public examples only",
            },
        )
        return CodeBundle(
            "livecodebench",
            task,
            files,
            "solution.py",
            {
                "sample": sample(public + private),
                "public_count": len(public),
                "private_count": len(private),
                "checker_source": checker_source,
            },
        )


class ClassEvalAdapter:
    @staticmethod
    def from_record(row, *, revision, harness_files=None):
        files = {"solution.py": row["skeleton"] + "\n"}
        instruction = (
            f"Implement the Python class {row['class_name']} specified in solution.py. "
            "The file contains the public skeleton, docstrings and examples. Implement all methods, "
            "including state changes shared across methods. Preserve the required interface. "
            "Create and run your own tests from the specification and examples; the benchmark evaluation "
            "tests and reference implementation are unavailable. Only solution.py is submitted. "
            "When finished, call finish with a concise summary.\n\n"
            + row.get("class_description", "")
        )
        task = Task(
            "FudanSELab/ClassEval",
            revision,
            "dev",
            row["task_id"],
            instruction,
            {
                "protocol": "classeval-holistic-agentic-v1",
                "solution_path": "solution.py",
                "class_name": row["class_name"],
                "test_visibility": "skeleton examples; final tests withheld",
            },
        )
        return CodeBundle(
            "classeval",
            task,
            files,
            "solution.py",
            {
                "test": row["test"],
                "reference_code": row["solution_code"],
                "imports": "\n".join(row["import_statement"]),
                "test_classes": row["test_classes"],
                "harness_files": dict(harness_files or {}),
            },
        )


def evaluation_files(bundle, code):
    if bundle.kind == "quixbugs":
        files = copy.deepcopy(bundle.private["evaluation_files"])
        files[bundle.solution_path] = code
        return files, bundle.private["command"]
    if bundle.kind == "livecodebench":
        return {
            "solution.py": code,
            "cases.json": json.dumps(bundle.private["sample"]),
            "testing_util.py": bundle.private["checker_source"],
            "evaluate.py": LCB_DRIVER,
        }, "python evaluate.py"
    if bundle.kind == "classeval":
        # Upstream ClassEval concatenates imports, generated class code and unchanged test code.
        source = bundle.private["imports"] + "\n\n" + code + "\n\n" + bundle.private["test"]
        harness = bundle.private["harness_files"]
        if not harness:
            raise ValueError("ClassEval evaluation requires pinned upstream harness files")
        driver = """import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('upstream').resolve()))
from test_pipeline import AutoTest
engine=AutoTest.__new__(AutoTest)
meta=json.loads(Path('evaluation_metadata.json').read_text())
code=engine.add_static_statement(Path('solution.py').read_text())
code=meta['imports']+'\\n'+code
engine.gen_py_file('submission',[code],Path('official_tests.txt').read_text())
results=engine.test(1,'submission',meta['test_classes'],'pilot')
classes=results['submission_0']
passed=bool(classes) and all(r['testsRun']>0 and r['errors']==0 and r['failures']==0 for r in classes.values())
print('FLOWPILOT_EVAL_JSON='+json.dumps({'passed':passed,'test_classes':classes,'test_count':sum(r['testsRun'] for r in classes.values()),'normalisation':'upstream add_static_statement'}))
log=Path('log/pilot_log_data.log')
if log.exists(): print(log.read_text())
"""
        return {
            "test_submission.py": source,
            "solution.py": code,
            "official_tests.txt": bundle.private["test"],
            "evaluation_metadata.json": json.dumps(
                {
                    "imports": bundle.private["imports"],
                    "test_classes": bundle.private["test_classes"],
                }
            ),
            "evaluate_unittest.py": driver,
            **{"upstream/" + name: text for name, text in harness.items()},
        }, "python evaluate_unittest.py"
    raise ValueError("Unsupported code adapter")
