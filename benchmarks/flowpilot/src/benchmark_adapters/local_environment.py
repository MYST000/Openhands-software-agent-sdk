"""Explicit local pilot backend; this is not Docker or namespace isolation."""

import fcntl
import json
import os
import pwd
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .contracts import CommandResult
from .environment import command_argv
from .swe import export_patch_command
from .tracing import write_json


class LocalPilotEnvironment:
    def __init__(self, config, task, artifact_dir, *, task_root, python_bin, uid=63101):
        self.config, self.task = config, task
        self.artifact_dir = Path(artifact_dir)
        self.task_root = Path(task_root).resolve()
        if self.task_root.parent != Path("/tmp") or not self.task_root.name.startswith(
            ("flowpilot-swe-local-", "flowpilot-code-local-", "fp-local-test-")
        ):
            raise ValueError("Local pilot requires its own dedicated /tmp task directory")
        self.repo_dir = str(self.task_root / "repo")
        self.python_bin = str(Path(python_bin).absolute())
        self.uid = uid
        self._started = self._closed = self._ready = self._uid_owned = False
        self._uid_lock = None
        self._process = None
        self.metadata = dict(
            environment_id=uuid.uuid4().hex,
            backend="local-process-pilot",
            isolation="dedicated-directory-and-unprivileged-uid; shared-kernel-and-network",
            task_id=task.task_id,
            repo_dir=self.repo_dir,
            task_root=str(self.task_root),
            executor_uid=uid,
            python_bin=self.python_bin,
            official_docker_evaluation=False,
        )

    def prepare(self):
        if self._started or self._closed:
            raise RuntimeError("LocalPilotEnvironment is single-use")
        self._started = True
        if os.geteuid() != 0 or not 60000 <= self.uid < 65000:
            raise ValueError(
                "Local pilot requires a root controller and an unused UID in 60000..64999"
            )
        lock_dir = Path("/root/.cache/flowpilot-local-uid-locks")
        lock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._uid_lock = (lock_dir / str(self.uid)).open("a")
        try:
            fcntl.flock(self._uid_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._uid_lock.close()
            self._uid_lock = None
            raise ValueError("Pilot UID is already reserved by another environment") from None
        try:
            return self._prepare()
        except BaseException:
            self._release_uid()
            raise

    def _prepare(self):
        try:
            pwd.getpwuid(self.uid)
        except KeyError:
            pass
        else:
            raise ValueError("Pilot UID already belongs to an existing account")
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                text = status.read_text()
            except (OSError, ProcessLookupError):
                continue
            for line in text.splitlines():
                if line.startswith("Uid:") and str(self.uid) in line.split()[1:]:
                    raise ValueError("Pilot UID is already in use by a process")
        self._uid_owned = True
        base = subprocess.check_output(
            ["git", "-C", self.repo_dir, "rev-parse", "HEAD"], text=True
        ).strip()
        if base != self.task.public_metadata["base_commit"]:
            raise ValueError("Local repository is not at the exact SWE base commit")
        if subprocess.check_output(
            ["git", "-C", self.repo_dir, "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("Local repository must be clean before the actor")
        self.baseline_commit = base
        for name in ("user-home", "cache", "config", "data", "tmp"):
            (self.task_root / name).mkdir(exist_ok=True)
        self.task_root.chmod(0o700)
        for current, dirs, files in os.walk(self.task_root, followlinks=False):
            os.chown(current, self.uid, self.uid, follow_symlinks=False)
            for name in dirs + files:
                os.chown(Path(current) / name, self.uid, self.uid, follow_symlinks=False)
        self.metadata.update(
            prepared_head=base, patch_baseline="base_commit", lifecycle_status="ready"
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "image_setup.patch").write_text("")
        write_json(self.artifact_dir / "environment.json", self.metadata)
        self._ready = True
        return self.metadata.copy()

    def execute(self, command, timeout=None, max_bytes=None, **kwargs):
        if self._closed:
            raise RuntimeError("Local pilot environment is closed")
        if not self._ready:
            raise RuntimeError("Local pilot environment is not prepared")
        timeout = timeout or self.config.runtime.tool_timeout
        max_bytes = max_bytes or self.config.runtime.max_output_chars
        argv = [
            "/usr/bin/setpriv",
            f"--reuid={self.uid}",
            f"--regid={self.uid}",
            "--clear-groups",
            "--no-new-privs",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--bounding-set=-all",
            *command_argv(command, self.repo_dir, timeout, max_bytes),
        ]
        process_env = dict(
            PATH=self.python_bin + ":/usr/bin:/bin",
            LANG="C.UTF-8",
            LC_ALL="C.UTF-8",
            HOME=str(self.task_root / "user-home"),
            XDG_CACHE_HOME=str(self.task_root / "cache"),
            XDG_CONFIG_HOME=str(self.task_root / "config"),
            XDG_DATA_HOME=str(self.task_root / "data"),
            TMPDIR=str(self.task_root / "tmp"),
            USER="flowpilot-pilot",
            LOGNAME="flowpilot-pilot",
            CUDA_VISIBLE_DEVICES="",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=self.repo_dir + "/src",
            PIP_DISABLE_PIP_VERSION_CHECK="1",
            PIP_CACHE_DIR=str(self.task_root / "pip-cache"),
        )
        self._process = subprocess.Popen(
            argv,
            cwd=self.repo_dir,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = self._process.communicate(timeout=timeout + 15)
            if self._process.returncode:
                raise RuntimeError("Local executor failed: " + err.decode(errors="replace")[-1500:])
            data = json.loads(out)
            data["clock_domain"] = "local-process"
            return CommandResult(**data)
        finally:
            self.quiesce()

    def _live_uid_pids(self):
        pids = []
        for status in Path("/proc").glob("[0-9]*/status"):
            try:
                fields = dict(
                    line.split(":", 1) for line in status.read_text().splitlines() if ":" in line
                )
                if str(self.uid) in fields["Uid"].split() and fields["State"].strip()[0] != "Z":
                    pids.append(int(status.parent.name))
            except (OSError, KeyError):
                continue
        return pids

    def quiesce(self):
        # Actor commands start their own sessions. A root controller must also clean
        # them up when the same-UID supervisor is killed or exits unexpectedly.
        if self._uid_owned:
            deadline = time.monotonic() + 10
            while pids := self._live_uid_pids():
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("Could not quiesce local pilot UID processes")
                time.sleep(0.01)
        if self._process is not None:
            self._process.communicate(timeout=5)
            self._process = None

    def _release_uid(self):
        self._uid_owned = False
        if self._uid_lock is not None:
            self._uid_lock.close()
            self._uid_lock = None

    def export_patch(self):
        result = self.execute(export_patch_command(self.baseline_commit), max_bytes=20000000)
        if result.exit_code or result.timed_out or result.truncated:
            raise RuntimeError("Local patch export failed")
        return result.stdout

    def close(self):
        self.quiesce()
        self._closed = True
        self._ready = False
        self._release_uid()
        self.metadata["lifecycle_status"] = "closed-workspace-retained"
        write_json(self.artifact_dir / "environment.json", self.metadata)
        return self.metadata.copy()
