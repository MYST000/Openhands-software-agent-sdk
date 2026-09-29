"""Bounded task processes with isolated worker slots and causal load snapshots."""

import contextlib
import multiprocessing
import os
import queue
import signal
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


@contextlib.contextmanager
def _state_lock(lock, timeout=5.0):
    if not lock.acquire(timeout=timeout):
        raise RuntimeError("Shared state lock timed out; admission must stop")
    try:
        yield
    finally:
        lock.release()


@dataclass
class WorkerContext:
    slot: int
    max_sessions: int
    state: Any
    lock: Any

    def set_phase(self, phase):
        with _state_lock(self.lock):
            self.state[f"slot:{self.slot}"] = phase

    @contextlib.contextmanager
    def activity(self, phase):
        key = f"slot:{self.slot}"
        with _state_lock(self.lock):
            previous = self.state.get(key, "actor")
            self.state[key] = phase
        try:
            yield
        finally:
            with _state_lock(self.lock):
                self.state[key] = previous

    def snapshot(self):
        with _state_lock(self.lock):
            phases = Counter(value for key, value in self.state.items() if key.startswith("slot:"))
            return {
                "sampled_monotonic_ns": time.monotonic_ns(),
                "active_sessions": sum(phases.values()),
                "llm_inflight": phases["llm"],
                "tool_inflight": phases["tool"],
                "setup_active": phases["setup"],
                "queued_sessions": self.state.get("queued", 0),
                "max_sessions": self.max_sessions,
                "scope": "campaign",
            }


def _execute(worker, job, context, channel):
    os.setsid()
    started = time.monotonic_ns()
    try:
        context.set_phase("setup")
        result = worker(job, context)
        record = {"status": "returned", "result": result}
    except BaseException as exc:
        record = {"status": "worker_error", "error_type": type(exc).__name__, "error": str(exc)}
    finally:
        try:
            with _state_lock(context.lock):
                context.state.pop(f"slot:{context.slot}", None)
        except Exception as exc:
            record = {"status": "worker_error", "error_type": type(exc).__name__, "error": str(exc)}
    channel.put(
        {
            "job_id": job["job_id"],
            "worker_slot": context.slot,
            "worker_pid": os.getpid(),
            "started_monotonic_ns": started,
            "finished_monotonic_ns": time.monotonic_ns(),
            **record,
        }
    )


def run_parallel(jobs, worker, *, concurrency=4, adapter_limits=None, on_result=None):
    """Consume a frozen ordered queue; concurrency is global to this invocation.

    ``worker`` must be importable for spawn. It owns task cleanup. A killed or
    lost worker aborts admission: a possibly unclean code slot is never reused.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be an integer in 1..64")
    if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("Expected nonempty jobs with unique job_id")
    limits = adapter_limits or {job.get("adapter", "default"): 4 for job in jobs}
    if any(type(limit) is not int or limit < 1 for limit in limits.values()):
        raise ValueError("Each adapter concurrency limit must be positive")
    if any(job.get("adapter", "default") not in limits for job in jobs):
        raise ValueError("Every adapter needs an explicit concurrency limit")
    ctx = multiprocessing.get_context("spawn")
    pending = deque(jobs)
    records, active = [], {}
    peak_active = 0
    with ctx.Manager() as manager:
        state, lock = manager.dict(), manager.RLock()
        state["queued"] = len(pending)
        channel = ctx.Queue()
        try:
            while pending or active:
                for slot in range(concurrency):
                    if not pending or slot in active:
                        continue
                    counts = Counter(
                        item["job"].get("adapter", "default") for item in active.values()
                    )
                    job = next(
                        (
                            candidate
                            for candidate in pending
                            if counts[candidate.get("adapter", "default")]
                            < limits[candidate.get("adapter", "default")]
                        ),
                        None,
                    )
                    if job is None:
                        continue
                    pending.remove(job)
                    with _state_lock(lock):
                        state["queued"] = len(pending)
                        state[f"slot:{slot}"] = "setup"
                    context = WorkerContext(slot, concurrency, state, lock)
                    process = ctx.Process(target=_execute, args=(worker, job, context, channel))
                    dispatched = time.monotonic_ns()
                    process.start()
                    active[slot] = {
                        "process": process,
                        "job": job,
                        "dispatched": dispatched,
                        "interrupted_at": None,
                        "dead_since": None,
                    }
                    peak_active = max(peak_active, len(active))
                try:
                    record = channel.get(timeout=0.1)
                except queue.Empty:
                    record = None
                if record is not None:
                    item = active.pop(record["worker_slot"])
                    if item["job"]["job_id"] != record["job_id"]:
                        raise RuntimeError("Worker result identity mismatch")
                    item["process"].join(timeout=10)
                    if item["process"].is_alive():
                        active[record["worker_slot"]] = item
                        raise RuntimeError("Worker returned but did not exit; refusing slot reuse")
                    record["dispatched_monotonic_ns"] = item["dispatched"]
                    record["controller_timeout_requested"] = item["interrupted_at"] is not None
                    records.append(record)
                    if on_result:
                        on_result(record)
                    if record["status"] != "returned":
                        raise RuntimeError(
                            "Worker failed outside the actor result contract; admission stopped"
                        )
                now = time.monotonic()
                for item in active.values():
                    process, job = item["process"], item["job"]
                    timeout = job.get("controller_timeout_s", 0)
                    elapsed = (time.monotonic_ns() - item["dispatched"]) / 1e9
                    if timeout and elapsed > timeout and item["interrupted_at"] is None:
                        if process.is_alive():
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(process.pid, signal.SIGINT)
                        item["interrupted_at"] = now
                    if item["interrupted_at"] is not None and now - item["interrupted_at"] > 30:
                        raise RuntimeError("Worker deadline exceeded; cleanup must be inspected")
                    if not process.is_alive():
                        if item["dead_since"] is None:
                            item["dead_since"] = now
                        elif now - item["dead_since"] > 2:
                            raise RuntimeError(
                                "Worker exited without a receipt; refusing slot reuse"
                            )
        finally:
            for item in active.values():
                if item["process"].is_alive():
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(item["process"].pid, signal.SIGINT)
            deadline = time.monotonic() + 15
            for item in active.values():
                process = item["process"]
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
                    process.join(timeout=5)
            channel.close()
    return {
        "concurrency": concurrency,
        "adapter_limits": limits,
        "peak_active_workers": peak_active,
        "records": records,
    }
