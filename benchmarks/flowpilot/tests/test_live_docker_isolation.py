"""Opt-in acceptance against two real SWE containers; never pulls an image."""

import os
import shlex
import uuid
from contextlib import ExitStack
from dataclasses import replace

import pytest


@pytest.mark.skipif(
    not os.environ.get("BENCHMARK_ADAPTERS_DOCKER_TEST_CONFIG"),
    reason="Set BENCHMARK_ADAPTERS_DOCKER_TEST_CONFIG to a prepared SWE config for real Docker acceptance",
)
def test_real_swe_containers_do_not_share_changes(tmp_path):
    import docker

    from benchmark_adapters.cli import tasks_for
    from benchmark_adapters.config import load_config
    from benchmark_adapters.environment import DockerEnvironment
    from benchmark_adapters.swe import checked_dataset_digest

    config = load_config(os.environ["BENCHMARK_ADAPTERS_DOCKER_TEST_CONFIG"])
    assert config.dataset.kind == "swe"
    config = replace(
        config, docker=replace(config.docker, keep_container=False, pull_missing=False)
    )
    checked_dataset_digest(config)
    task = tasks_for(
        config, ids=os.environ.get("BENCHMARK_ADAPTERS_DOCKER_TEST_IDS", "").split() or None
    )[0]
    marker = shlex.quote("adapter_isolation_" + uuid.uuid4().hex + ".py")
    with ExitStack() as stack:
        environments = []
        for i in range(2):
            env = DockerEnvironment(
                config,
                task,
                tmp_path / f"attempt-{i}/artifacts",
                identity={"run_id": "docker-isolation-test", "attempt_id": f"attempt-{i}"},
            )
            stack.callback(env.close)
            env.prepare()
            environments.append(env)
        first, second = environments
        ids = [env.metadata["container_id"] for env in environments]
        assert ids[0] != ids[1]
        assert first.execute(f'printf "value=1\\n" > {marker}').exit_code == 0
        assert "value=1" in first.execute(f"cat {marker}").stdout
        assert second.execute(f"test ! -e {marker}").exit_code == 0
        # Test patches come from actual repository changes, not model-generated patch text.
        assert "+value=1" in first.export_patch()
        assert second.export_patch() == ""
    client = docker.from_env(timeout=10)
    try:
        for container_id in ids:
            with pytest.raises(docker.errors.NotFound):
                client.containers.get(container_id)
    finally:
        client.close()
