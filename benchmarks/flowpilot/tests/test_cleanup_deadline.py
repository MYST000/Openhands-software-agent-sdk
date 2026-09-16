"""Kernel exit/reaping may outlast one second after a command timeout."""

import ast
import signal
from types import SimpleNamespace

import pytest

from benchmark_adapters.environment import _COMMAND_RUNNER


def run_cleanup_with_delayed_reaping(ready_at):
    # Simulate delayed kernel teardown without allocating gigabytes in a test.
    clock = SimpleNamespace(now=0.0, reaped=False)

    def sleep(seconds):
        clock.now += seconds

    def waitpid(pid, flags):
        if clock.reaped:
            raise ChildProcessError
        if clock.now >= ready_at:
            clock.reaped = True
            return (42, 9)
        return (0, 0)

    children = SimpleNamespace(read_text=lambda: "" if clock.reaped else "42")
    scope = {
        "os": SimpleNamespace(
            getpid=lambda: 1,
            killpg=lambda *args: None,
            kill=lambda *args: None,
            waitpid=waitpid,
            WNOHANG=1,
        ),
        "pathlib": SimpleNamespace(Path=lambda path: children),
        "time": SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep),
        "signal": signal,
    }
    cleanup_node = next(
        n
        for n in ast.parse(_COMMAND_RUNNER).body
        if isinstance(n, ast.FunctionDef) and n.name == "cleanup"
    )
    exec(compile(ast.Module(body=[cleanup_node], type_ignores=[]), "supervisor", "exec"), scope)
    return lambda: scope["cleanup"](SimpleNamespace(pid=42)), clock


def test_cleanup_allows_delayed_kernel_reaping():
    cleanup, clock = run_cleanup_with_delayed_reaping(1.5)
    cleanup()
    assert clock.reaped
    assert 1.5 <= clock.now < 15


def test_unreapable_descendants_still_fail_within_outer_grace():
    cleanup, clock = run_cleanup_with_delayed_reaping(float("inf"))
    with pytest.raises(RuntimeError, match="Could not quiesce"):
        cleanup()
    assert not clock.reaped
    assert 0 < clock.now < 15
