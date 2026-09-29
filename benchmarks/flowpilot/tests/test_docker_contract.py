import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_docker_prepare_preserves_setup_baseline_and_removes_only_owned_container(
    tmp_path, monkeypatch
):
    import docker

    from benchmark_adapters.config import Config, DockerConfig
    from benchmark_adapters.environment import DockerEnvironment
    from benchmark_adapters.swe import SWEAdapter

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "source.py").write_text("x=0\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (repo / "setup.txt").write_text("environment installation\n")
    git("add", ".")
    git("commit", "-qm", "SWE-bench")
    prepared = git("rev-parse", "HEAD")
    (repo / "untracked-install-file").write_text("keep\n")

    class Container:
        id = "owned-container"
        removed = False

        def start(self):
            pass

        def remove(self, force):
            self.removed = True

    container = Container()
    created_options = {}

    class API:
        def exec_create(self, container_id, command):
            assert container_id == container.id
            self.command = command
            return {"Id": "exec"}

        def exec_start(self, exec_id):
            return subprocess.check_output(self.command)

        def exec_inspect(self, exec_id):
            return {"Running": False, "ExitCode": 0}

    def create(image, **kwargs):
        created_options.update(kwargs)
        assert image == "sha256:fixture"
        return container

    client = SimpleNamespace(
        ping=lambda: True,
        close=lambda: None,
        api=API(),
        images=SimpleNamespace(
            get=lambda _: SimpleNamespace(
                id="sha256:fixture", attrs={"RepoDigests": ["fixture@sha256:1"]}
            )
        ),
        containers=SimpleNamespace(create=create),
    )
    monkeypatch.setattr(docker, "from_env", lambda **kwargs: client)
    task = SWEAdapter.from_record(
        dict(instance_id="org__repo-1", repo="org/repo", base_commit=base, problem_statement="fix"),
        dataset_id="princeton-nlp/SWE-bench",
        revision="v",
        split="dev",
    )
    config = Config(docker=replace(DockerConfig(), repo_dir=str(repo), conda_env=""))
    env = DockerEnvironment(config, task, tmp_path / "artifacts")
    try:
        metadata = env.prepare()
        assert metadata["prepared_head"] == prepared
        assert git("rev-parse", "HEAD") == prepared
        (repo / "source.py").write_text("x=1\n")
        patch = env.export_patch()
        assert "+x=1" in patch
        assert "setup.txt" not in patch and "untracked-install-file" not in patch
        assert "setup.txt" in (tmp_path / "artifacts/image_setup.patch").read_text()
    finally:
        env.close()
    assert container.removed
    assert "volumes" not in created_options and created_options["network_disabled"] is True


def test_container_start_failure_still_leaves_handle_for_cleanup(tmp_path, monkeypatch):
    import docker

    from benchmark_adapters.config import Config
    from benchmark_adapters.contracts import Task
    from benchmark_adapters.environment import DockerEnvironment

    class Container:
        id = "failed-start"
        removed = False

        def start(self):
            raise RuntimeError("start failed")

        def remove(self, force):
            self.removed = True

    container = Container()
    client = SimpleNamespace(
        ping=lambda: True,
        close=lambda: None,
        images=SimpleNamespace(get=lambda _: SimpleNamespace(id="image", attrs={})),
        containers=SimpleNamespace(create=lambda *args, **kwargs: container),
    )
    monkeypatch.setattr(docker, "from_env", lambda **kwargs: client)
    env = DockerEnvironment(
        Config(),
        Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": "a" * 40}),
        tmp_path,
    )
    with pytest.raises(RuntimeError, match="start failed"):
        env.prepare()
    env.close()
    assert container.removed


def test_repeated_attempts_get_fresh_labeled_containers_and_close_is_idempotent(
    tmp_path, monkeypatch
):
    import json

    import docker

    from benchmark_adapters.config import Config
    from benchmark_adapters.contracts import Task
    from benchmark_adapters.environment import DockerEnvironment

    created = []

    class Container:
        def __init__(self, options):
            self.id = f"container-{len(created)}"
            self.options = options
            self.removals = 0
            self.stops = 0

        def start(self):
            pass

        def remove(self, force):
            self.removals += 1

        def stop(self, timeout):
            self.stops += 1

    def create(image, **options):
        container = Container(options)
        created.append(container)
        return container

    monkeypatch.setattr(
        docker,
        "from_env",
        lambda **kw: SimpleNamespace(
            ping=lambda: True,
            close=lambda: None,
            images=SimpleNamespace(get=lambda name: SimpleNamespace(id="image", attrs={})),
            containers=SimpleNamespace(create=create),
        ),
    )
    # Lifecycle test only: Git baseline behavior is covered by the real-Git test above.
    monkeypatch.setattr(
        DockerEnvironment,
        "checked",
        lambda self, command, **kw: "a" * 40 + "\n" if command == "git rev-parse HEAD" else "",
    )
    task = Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": "a" * 40})
    envs = [
        DockerEnvironment(
            Config(),
            task,
            tmp_path / f"attempt-{i}/artifacts",
            identity={"run_id": "run-1", "attempt_id": f"attempt-{i}"},
        )
        for i in range(2)
    ]
    try:
        metadata = [env.prepare() for env in envs]
        assert metadata[0]["container_id"] != metadata[1]["container_id"]
        assert metadata[0]["environment_id"] != metadata[1]["environment_id"]
        for i, container in enumerate(created):
            assert container.options["labels"]["flowpilot.attempt_id"] == f"attempt-{i}"
            assert container.options["labels"]["flowpilot.run_id"] == "run-1"
            assert container.options["auto_remove"] is True
            assert container.options["command"][0] == "sleep"
            assert int(container.options["command"][1]) > Config().runtime.task_timeout
            assert not container.options.get("volumes")
            with pytest.raises(RuntimeError, match="single-use"):
                envs[i].prepare()
    finally:
        for env in envs:
            env.close()
            env.close()
    assert [c.removals for c in created] == [1, 1]
    for i, env in enumerate(envs):
        with pytest.raises(RuntimeError, match="closed"):
            env.execute("true")
        data = json.loads((tmp_path / f"attempt-{i}/artifacts/environment.json").read_text())
        assert data["lifecycle_status"] == "removed"


def test_kept_container_is_stopped_and_never_reused(tmp_path, monkeypatch):
    import docker

    from benchmark_adapters.config import Config, DockerConfig
    from benchmark_adapters.contracts import Task
    from benchmark_adapters.environment import DockerEnvironment

    calls = []
    container = SimpleNamespace(
        id="kept",
        start=lambda: None,
        stop=lambda **kw: calls.append("stop"),
        remove=lambda **kw: calls.append("remove"),
    )
    monkeypatch.setattr(
        docker,
        "from_env",
        lambda **kw: SimpleNamespace(
            ping=lambda: True,
            close=lambda: None,
            images=SimpleNamespace(get=lambda name: SimpleNamespace(id="image", attrs={})),
            containers=SimpleNamespace(create=lambda *a, **kw: container),
        ),
    )
    monkeypatch.setattr(
        DockerEnvironment,
        "checked",
        lambda self, command, **kw: "a" * 40 + "\n" if command == "git rev-parse HEAD" else "",
    )
    env = DockerEnvironment(
        Config(docker=DockerConfig(keep_container=True)),
        Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": "a" * 40}),
        tmp_path,
    )
    env.prepare()
    env.close()
    assert calls == ["stop"]
    with pytest.raises(RuntimeError, match="single-use"):
        env.prepare()


@pytest.mark.parametrize("keep_container", [False, True])
def test_cleanup_failure_can_be_retried_without_reopening_actor(tmp_path, keep_container):
    import docker

    from benchmark_adapters.config import Config, DockerConfig
    from benchmark_adapters.contracts import Task
    from benchmark_adapters.environment import DockerEnvironment

    attempts, fail = [], [True]

    def remove_or_stop(**kwargs):
        attempts.append(kwargs)
        if fail[0]:
            raise docker.errors.DockerException("temporary connection failure")

    env = DockerEnvironment(
        Config(docker=DockerConfig(keep_container=keep_container)),
        Task("swe", "v", "dev", "org__repo-1", "issue", {"base_commit": "a" * 40}),
        tmp_path,
    )
    env.container = SimpleNamespace(id="owned", remove=remove_or_stop, stop=remove_or_stop)
    env.client = SimpleNamespace(close=lambda: None)
    with pytest.raises(docker.errors.DockerException):
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.execute("true")
    fail[0] = False
    status = env.close()
    assert status["lifecycle_status"] == ("retained_stopped" if keep_container else "removed")
    assert len(attempts) >= 2
    total = len(attempts)
    env.close()
    assert len(attempts) == total
