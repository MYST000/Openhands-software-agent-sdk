import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.geteuid() != 0, reason="Local pilot drops root to a dedicated unprivileged UID"
)
def test_local_pilot_executes_real_edits_as_unprivileged_user_and_exports_patch(tmp_path):
    from benchmark_adapters.config import Config
    from benchmark_adapters.contracts import Task
    from benchmark_adapters.local_environment import LocalPilotEnvironment

    root = Path(tempfile.mkdtemp(prefix="fp-local-test-", dir="/tmp"))
    secret_fd, secret_path = tempfile.mkstemp(prefix="fp-protected-", dir="/tmp")
    os.close(secret_fd)
    secret = Path(secret_path)
    secret.write_text("protected")
    secret.chmod(0o600)
    try:
        repo = root / "repo"
        repo.mkdir()

        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

        git("init", "-q")
        git("config", "user.name", "Test")
        git("config", "user.email", "test@example.invalid")
        (repo / "a.py").write_text("value=0\n")
        git("add", ".")
        git("commit", "-qm", "base")
        base = git("rev-parse", "HEAD")
        invalid_task = Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": "0" * 40})
        invalid = LocalPilotEnvironment(
            Config(),
            invalid_task,
            tmp_path / "invalid",
            task_root=root,
            python_bin="/usr/bin",
            uid=63103,
        )
        with pytest.raises(ValueError, match="exact SWE base commit"):
            invalid.prepare()
        with pytest.raises(RuntimeError, match="not prepared"):
            invalid.execute("true")
        task = Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": base})
        env = LocalPilotEnvironment(
            Config(), task, tmp_path / "artifacts", task_root=root, python_bin="/usr/bin", uid=63102
        )
        env.prepare()
        try:
            assert env.execute("id -u").stdout.strip() == "63102"
            home = env.execute("python3 -c 'from pathlib import Path; print(Path.home())'")
            assert home.exit_code == 0
            assert home.stdout.strip() == str(root / "user-home")
            assert env.execute(f"cat {secret}").exit_code != 0
            result = env.execute("printf 'value=1\\n' > a.py")
            assert result.exit_code == 0
            assert result.clock_domain == "local-process"
            assert (repo / "a.py").read_text() == "value=1\n"
            assert "+value=1" in env.export_patch()
            with pytest.raises(RuntimeError, match="Local executor failed"):
                env.execute('kill -KILL "$PPID"; sleep 0.3; echo late > late.txt')
            import time

            time.sleep(0.6)
            assert not (repo / "late.txt").exists()
            with pytest.raises(RuntimeError, match="single-use"):
                env.prepare()
        finally:
            env.close()
        with pytest.raises(RuntimeError, match="closed"):
            env.execute("true")
    finally:
        shutil.rmtree(root)
        secret.unlink()
