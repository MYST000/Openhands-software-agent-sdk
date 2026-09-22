import json
import os
import threading
import time
from pathlib import Path

import pytest

from benchmark_adapters.parallel_runtime import WorkerContext, run_parallel
from benchmark_adapters.tracing import TraceRecorder


def fixture_worker(job, context):
    """Spawn-importable worker exercising scheduling without executing actor code."""
    if job.get("marker"):
        Path(job["marker"]).write_text(str(os.getpid()))
    if job.get("crash"):
        os._exit(9)
    if job.get("raise"):
        raise ValueError("fixture failure")
    uids = None
    if "uid_base" in job:
        from benchmark_adapters.parallel_collection import worker_uids

        uids = worker_uids(job["uid_base"], context.slot)
    context.set_phase("actor")
    started = time.monotonic_ns()
    if job.get("barrier"):
        with context.lock:
            context.state["test:started"] = context.state.get("test:started", 0) + 1
        deadline = time.monotonic() + 10
        while context.state.get("test:started", 0) < job["barrier"]:
            if time.monotonic() > deadline:
                raise RuntimeError("Fixture barrier did not reach actual worker concurrency")
            time.sleep(0.005)
    with context.activity(job.get("phase", "llm")):
        snapshot = context.snapshot()
        time.sleep(job.get("delay", 0.03))
    return {
        "start": started,
        "end": time.monotonic_ns(),
        "pid": os.getpid(),
        "uids": uids,
        "slot": context.slot,
        "snapshot": snapshot,
        "task_status": job.get("task_status", "completed"),
    }


def peak_overlap(results):
    boundaries = [(r["start"], 1) for r in results] + [(r["end"], -1) for r in results]
    active = peak = 0
    for _, change in sorted(boundaries):
        active += change
        peak = max(peak, active)
    return peak


def test_four_same_adapter_tasks_overlap_and_fifth_reuses_finished_slot():
    # Same adapter throughout: capacity must not reserve one slot per benchmark.
    jobs = [
        {
            "job_id": str(i),
            "adapter": "livecodebench",
            "barrier": 4,
            "uid_base": 63200,
            "delay": 0.05 if i == 0 else 0.35,
        }
        for i in range(4)
    ]
    jobs.append({"job_id": "4", "adapter": "livecodebench", "delay": 0.02, "uid_base": 63200})
    delivered = []
    report = run_parallel(
        jobs,
        fixture_worker,
        concurrency=4,
        adapter_limits={"livecodebench": 4},
        on_result=delivered.append,
    )
    assert report["peak_active_workers"] == 4
    assert report["records"] == delivered
    assert len(delivered) == 5
    assert all(row["status"] == "returned" for row in delivered)
    records = {row["job_id"]: row for row in delivered}
    results = [row["result"] for row in delivered]
    assert peak_overlap(results) == 4
    assert len({records[str(i)]["worker_slot"] for i in range(4)}) == 4
    assert len({records[str(i)]["worker_pid"] for i in range(4)}) == 4
    all_uids = [uid for i in range(4) for uid in records[str(i)]["result"]["uids"]]
    assert len(set(all_uids)) == 8  # Actual production mapping, no root operations.
    assert records["4"]["result"]["uids"] == records["0"]["result"]["uids"]
    assert records["4"]["worker_slot"] == records["0"]["worker_slot"]
    assert records["4"]["result"]["start"] >= records["0"]["finished_monotonic_ns"]
    assert all(result["snapshot"]["active_sessions"] <= 4 for result in results)
    assert all(result["snapshot"]["scope"] == "campaign" for result in results)


def test_adapter_limit_skips_blocked_head_without_blocking_other_adapters():
    jobs = [
        {"job_id": "a0", "adapter": "a", "delay": 0.3},
        {"job_id": "a1", "adapter": "a", "delay": 0.05},
        {"job_id": "b0", "adapter": "b", "delay": 0.05},
        {"job_id": "b1", "adapter": "b", "delay": 0.05},
    ]
    report = run_parallel(jobs, fixture_worker, concurrency=4, adapter_limits={"a": 1, "b": 4})
    rows = {r["job_id"]: r["result"] for r in report["records"]}
    assert rows["a1"]["start"] >= rows["a0"]["end"]
    assert rows["b0"]["start"] < rows["a0"]["end"]
    assert rows["b1"]["start"] < rows["a0"]["end"]
    assert peak_overlap(list(rows.values())) <= 4


def test_returned_actor_failure_is_kept_and_next_job_runs():
    report = run_parallel(
        [{"job_id": "failed-task", "task_status": "tool_error"}, {"job_id": "next"}],
        fixture_worker,
        concurrency=1,
    )
    assert [r["job_id"] for r in report["records"]] == ["failed-task", "next"]
    assert report["records"][0]["result"]["task_status"] == "tool_error"


def test_worker_exception_delivers_receipt_and_aborts_unknown_cleanup_slot(tmp_path):
    forbidden = tmp_path / "never-admitted"
    receipts = []
    with pytest.raises(RuntimeError):
        run_parallel(
            [{"job_id": "bad", "raise": True}, {"job_id": "next", "marker": str(forbidden)}],
            fixture_worker,
            concurrency=1,
            on_result=receipts.append,
        )
    assert not forbidden.exists()
    assert len(receipts) == 1
    assert receipts[0]["status"] == "worker_error"
    assert receipts[0]["error_type"] == "ValueError"


def test_controller_timeout_interrupts_worker_and_does_not_reuse_slot(tmp_path):
    forbidden = tmp_path / "never-admitted"
    receipts = []
    began = time.monotonic()
    with pytest.raises(RuntimeError):
        run_parallel(
            [
                {"job_id": "slow", "delay": 30, "controller_timeout_s": 2},
                {"job_id": "next", "marker": str(forbidden)},
            ],
            fixture_worker,
            concurrency=1,
            on_result=receipts.append,
        )
    assert time.monotonic() - began < 12
    assert not forbidden.exists()
    assert len(receipts) == 1
    assert receipts[0]["status"] == "worker_error"
    assert receipts[0]["error_type"] == "KeyboardInterrupt"
    assert receipts[0]["controller_timeout_requested"] is True


def test_abrupt_worker_exit_aborts_without_admitting_pending_job(tmp_path):
    forbidden = tmp_path / "never-admitted"
    with pytest.raises(RuntimeError, match="without a receipt"):
        run_parallel(
            [{"job_id": "crash", "crash": True}, {"job_id": "next", "marker": str(forbidden)}],
            fixture_worker,
            concurrency=1,
        )
    assert not forbidden.exists()


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5, 65])
def test_invalid_global_capacity_rejected_before_spawning(concurrency):
    with pytest.raises(ValueError):
        run_parallel([{"job_id": "one"}], fixture_worker, concurrency=concurrency)


@pytest.mark.parametrize("jobs", [[], [{"job_id": "one"}, {"job_id": "one"}]])
def test_empty_or_duplicate_jobs_rejected_before_spawning(jobs):
    with pytest.raises(ValueError):
        run_parallel(jobs, fixture_worker)


def test_activity_restores_phase_and_snapshots_do_not_mutate():
    state = {"slot:0": "actor", "slot:1": "llm", "queued": 3}
    context = WorkerContext(0, 4, state, threading.RLock())
    before = context.snapshot()
    with context.activity("tool"):
        assert context.snapshot()["tool_inflight"] == 1
        with pytest.raises(ValueError), context.activity("llm"):
            assert context.snapshot()["llm_inflight"] == 2
            raise ValueError("fixture")
        assert context.snapshot()["tool_inflight"] == 1
    state["slot:1"] = "tool"
    state["queued"] = 0
    after = context.snapshot()
    assert before["llm_inflight"] == 1
    assert before["queued_sessions"] == 3
    assert after["llm_inflight"] == 0
    assert after["tool_inflight"] == 1
    assert state["slot:0"] == "actor"


def test_request_freezes_global_load_before_future_state_changes(tmp_path):
    from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input

    state = {"slot:0": "actor", "slot:1": "llm", "queued": 3}
    context = WorkerContext(0, 4, state, threading.RLock())
    recorder = TraceRecorder(tmp_path, {"run_id": "fixture", "attempt_id": "a"}, context)
    recorder.request("r1", {"messages": [], "tools": []})
    state["slot:1"] = "tool"
    state["queued"] = 0
    state["future_response_usage"] = {"completion_tokens": 99}
    recorder.request("r2", {"messages": [], "tools": []})
    recorder.close()
    first, second = build_prediction_rows(tmp_path)
    x1, x2 = load_t0_input(tmp_path, first), load_t0_input(tmp_path, second)
    load1, load2 = x1["t0_features"]["client_load"], x2["t0_features"]["client_load"]
    assert load1["llm_inflight"] == 1 and load1["tool_inflight"] == 0
    assert load2["llm_inflight"] == 0 and load2["tool_inflight"] == 1
    assert load1["queued_sessions"] == 3 and load2["queued_sessions"] == 0
    assert load1["sampled_monotonic_ns"] <= first["features"]["monotonic_ns"]
    assert "future_response_usage" not in json.dumps(x1)
    assert "future_response_usage" not in json.dumps(x2)


@pytest.mark.parametrize(
    "base,slot", [(True, 0), (63200, True), (63200, -1), (59999, 0), (64999, 0), (64998, 1)]
)
def test_production_uid_mapping_rejects_overlap_with_reserved_ranges(base, slot):
    from benchmark_adapters.parallel_collection import worker_uids

    with pytest.raises(ValueError):
        worker_uids(base, slot)


@pytest.mark.parametrize("limits", [{"a": 0}, {"a": -1}, {"a": True}, {"a": 1.5}, {"b": 4}])
def test_invalid_adapter_capacity_rejected_before_spawning(limits):
    with pytest.raises(ValueError):
        run_parallel([{"job_id": "one", "adapter": "a"}], fixture_worker, adapter_limits=limits)


def test_cleanup_continues_when_workers_exit_between_liveness_check_and_signal(monkeypatch):
    from unittest.mock import MagicMock

    context, manager = MagicMock(), MagicMock()
    context.Manager.return_value.__enter__.return_value = manager
    manager.dict.return_value = {}
    manager.RLock.return_value = threading.RLock()
    first, second = MagicMock(pid=900001), MagicMock(pid=900002)
    first.is_alive.return_value = second.is_alive.return_value = True
    first.kill.side_effect = ProcessLookupError
    context.Process.side_effect = [first, second]
    channel = context.Queue.return_value
    channel.get.side_effect = RuntimeError("fixture queue failure")
    monkeypatch.setattr(
        "benchmark_adapters.parallel_runtime.multiprocessing.get_context", lambda _method: context
    )
    signaled = []

    def signal_process(pid, _signal):
        signaled.append(pid)
        if pid == first.pid:
            raise ProcessLookupError

    monkeypatch.setattr("benchmark_adapters.parallel_runtime.os.kill", signal_process)
    with pytest.raises(RuntimeError, match="fixture queue failure"):
        run_parallel([{"job_id": "first"}, {"job_id": "second"}], fixture_worker, concurrency=2)
    assert signaled == [first.pid, second.pid]
    assert first.join.call_count == second.join.call_count == 2
    first.kill.assert_called_once()
    second.kill.assert_called_once()
    channel.close.assert_called_once()


def test_manager_state_lock_timeout_is_bounded_and_release_recovers():
    import multiprocessing

    from benchmark_adapters.parallel_runtime import _state_lock

    with multiprocessing.get_context("spawn").Manager() as manager:
        lock = manager.RLock()
        lock.acquire()
        outcomes = []

        def contender():
            try:
                with _state_lock(lock, timeout=0.05):
                    outcomes.append("acquired")
            except RuntimeError:
                outcomes.append("timed_out")

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=2)
        lock.release()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert outcomes == ["timed_out"]
        with _state_lock(lock, timeout=0.05):
            assert lock.acquire(blocking=False)
            lock.release()
