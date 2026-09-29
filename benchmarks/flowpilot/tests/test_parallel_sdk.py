"""CPU-only SDK integration fixtures; no benchmark/private data or real model."""

import hashlib
import json
import os
import re
import subprocess
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from benchmark_adapters.config import (
    Config,
    DatasetConfig,
    LLMConfig,
    RetrievalConfig,
    RuntimeConfig,
)
from benchmark_adapters.contracts import CommandResult, Task
from benchmark_adapters.environment import command_argv
from benchmark_adapters.parallel_runtime import run_parallel
from benchmark_adapters.prediction_export import build_prediction_rows, load_t0_input
from benchmark_adapters.retrieval import build_index
from benchmark_adapters.runner import run_task

CODE_KINDS = {"quixbugs", "livecodebench"}
SAMPLING_PARAMETERS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 20,
    "seed": 17,
    "presence_penalty": 0.2,
    "min_p": 0.05,
    "repetition_penalty": 1.1,
    "chat_template_kwargs": {"enable_thinking": False},
}
TOOL_MASKS = {
    "quixbugs": {"code_terminal", "code_file_editor", "finish", "think"},
    "livecodebench": {"code_terminal", "code_file_editor", "finish", "think"},
    "hotpot": {"search", "read_document", "finish", "think"},
    "browsecomp": {"search", "get_document", "finish", "think"},
}


class ControlledCodeFixture:
    """Only used with the literal responses below; not a production sandbox."""

    def __init__(self, directory, task_id):
        self.repo_dir = str(directory)
        self.task_id = task_id
        self.closed = False

    def prepare(self):
        directory = Path(self.repo_dir)
        directory.mkdir()
        (directory / "solution.py").write_text(f"fixture_identity = {self.task_id!r}\nvalue = 0\n")
        return {
            "backend": "controlled-public-fixture-not-a-sandbox",
            "environment_id": f"environment-{self.task_id}",
        }

    def execute(self, command, timeout=30, max_bytes=16000):
        output = subprocess.check_output(
            command_argv(command, self.repo_dir, timeout, max_bytes), text=True
        )
        return CommandResult(**json.loads(output))

    def quiesce(self):
        pass

    def close(self):
        self.closed = True
        return {"closed": True, "fixture_only": True}


def sdk_fixture_worker(job, context):
    """Importable spawn target executing the real SDK and adapter runner."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    kind, task_id = job["adapter"], job["job_id"]
    directory = Path(job["root"]) / task_id
    directory.mkdir()
    config = Config(
        dataset=DatasetConfig(kind=kind, id=f"fixture/{kind}", revision="fixture-v1"),
        llm=LLMConfig(
            model="openai/controlled-fixture",
            base_url=job["base_url"],
            api_key_env="FLOWPILOT_FIXTURE_UNUSED_API_KEY",
            timeout=90,
            num_retries=0,
            max_output_tokens=256,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
            seed=17,
            presence_penalty=0.2,
            min_p=0.05,
            repetition_penalty=1.1,
            enable_thinking=False,
        ),
        runtime=RuntimeConfig(
            max_iterations=4,
            max_tool_calls=4,
            max_llm_requests=4,
            task_timeout=150,
            tool_timeout=10,
            runs_dir=job["root"],
        ),
        retrieval=RetrievalConfig(
            backend="sqlite", index_path=job["index"], corpus_revision="public-fixture-v1"
        ),
    )
    task = Task(
        f"fixture/{kind}",
        "fixture-v1",
        "dev",
        task_id,
        f"MICROCHECK_TASK={task_id}\nUse the supplied tools to finish the fixture.",
        {"solution_path": "solution.py"} if kind in CODE_KINDS else {},
    )
    environment = (
        ControlledCodeFixture(directory / "code_workspace", task_id) if kind in CODE_KINDS else None
    )
    context.set_phase("actor")
    result = run_task(
        config,
        task,
        directory / "attempt",
        environment=environment,
        run_id=job["scenario"],
        trace_context={
            "episode_id": task_id,
            "replica_id": "cpu-controlled-fixture",
            "worker_slot": context.slot,
            "research_split": "fixture-only",
            "task_group_id": task_id,
            "queue_position": job["queue_position"],
        },
        load_monitor=context,
    )
    return {
        "actor": result,
        "attempt": str(directory / "attempt"),
        "code_environment_closed": environment.closed if environment else None,
    }


def fixture_action(kind, stage):
    if stage == 0:
        if kind in CODE_KINDS:
            return "code_file_editor", {
                "command": "str_replace",
                "path": "solution.py",
                "old_str": "value = 0",
                "new_str": "value = 1",
            }
        return "search", {"query": "Alpha", "top_k": 2}
    if stage == 1:
        if kind in CODE_KINDS:
            return "code_terminal", {"command": "cat solution.py", "timeout": 5}
        if kind == "hotpot":
            return "read_document", {
                "doc_id": "alpha",
                "start_sentence": 0,
                "max_sentences": 2,
            }
        return "get_document", {"docid": "alpha", "offset": 0}
    message = (
        json.dumps({"answer": "Alpha", "supporting_facts": [["Alpha", 0]]})
        if kind == "hotpot"
        else "Fixture complete. Evidence document: alpha."
    )
    return "finish", {"message": message}


def response_payload(task_id, stage, kind):
    tool, arguments = fixture_action(kind, stage)
    return {
        "id": f"chatcmpl-{task_id}-{stage}",
        "object": "chat.completion",
        "created": 1,
        "model": "controlled-fixture",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{task_id}-{stage}",
                            "type": "function",
                            "function": {
                                "name": tool,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def fixture_handler(jobs, state):
    by_id = {job["job_id"]: job for job in jobs}
    barrier = threading.Barrier(4, timeout=60)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            match = re.search(r"MICROCHECK_TASK=([\w-]+)", json.dumps(body["messages"]))
            assert match is not None
            task_id = match[1]
            job = by_id[task_id]
            with lock:
                stage = state["counts"][task_id]
                state["counts"][task_id] += 1
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["requests"].append({"task_id": task_id, "stage": stage, "body": body})
            try:
                if stage == 0 and job["queue_position"] < 4:
                    barrier.wait()
                assert stage < 3
                for key, value in SAMPLING_PARAMETERS.items():
                    assert body.get(key) == value, f"Sampling parameter changed: {key}"
                assert {tool["function"]["name"] for tool in body["tools"]} == TOOL_MASKS[
                    job["adapter"]
                ]
                payload = response_payload(task_id, stage, job["adapter"])
                status = 200
            except Exception as exc:
                with lock:
                    state["errors"].append(f"{task_id}: {type(exc).__name__}: {exc}")
                payload, status = {"error": {"message": "Controlled fixture failed"}}, 500
            finally:
                with lock:
                    state["active"] -= 1
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format, *args):
            pass

    return Handler


def checked_trace(job, record, http_requests):
    result = record["result"]
    actor = result["actor"]
    assert actor["execution_status"] == "completed", actor
    assert actor["artifact_status"] == "valid", actor
    assert not actor["errors"]
    assert actor["llm_requests"] == 3 and actor["tool_calls"] == 2
    attempt = Path(result["attempt"])
    assert (attempt / "conversation_workspace").is_dir()
    assert (attempt / "sdk_state").is_dir()
    events = [json.loads(line) for line in (attempt / "events.jsonl").read_text().splitlines()]
    requests = [event for event in events if event["event"] == "llm_request_prepared"]
    decisions = [event for event in events if event["event"] == "response_action_summary"]
    starts = [event for event in events if event["event"] == "tool_start"]
    ends = [event for event in events if event["event"] == "tool_end"]
    assert len(requests) == len(decisions) == 3
    assert len(starts) == len(ends) == 2
    samples = build_prediction_rows(attempt)
    assert len(samples) == 3
    expected_output_chars = 16000 if job["adapter"] in CODE_KINDS else None
    for stage, sample in enumerate(samples):
        assert sample["request_id"] == requests[stage]["request_id"]
        assert sample["masks"]["next_action"]
        assert sample["masks"]["first_executor_duration"] is (stage < 2)
        assert sample["labels"]["next_tool_name"] == fixture_action(job["adapter"], stage)[0]
        actor_input = load_t0_input(attempt, sample)
        assert "labels" not in actor_input and "response_usage" not in actor_input
        assert actor_input["t0_features"]["client_load"]["max_sessions"] == 4
        assert actor_input["tool_execution_profile"]["max_output_chars"] == expected_output_chars
    assert not [event for event in events if event["event"] == "tool_error"]
    for stage, (request, decision, actual) in enumerate(
        zip(requests, decisions, http_requests, strict=True)
    ):
        raw = (attempt / request["request"]["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == request["request"]["sha256"]
        saved = json.loads(raw)
        assert saved["messages"] == actual["body"]["messages"]
        assert saved["tools"] == actual["body"]["tools"]
        assert {tool["function"]["name"] for tool in saved["tools"]} == TOOL_MASKS[job["adapter"]]
        load = request["t0_features"]["client_load"]
        assert load["max_sessions"] == 4 and 1 <= load["active_sessions"] <= 4
        assert 1 <= load["llm_inflight"] <= 4
        assert load["sampled_monotonic_ns"] <= request["monotonic_ns"]
        assert request["worker_slot"] == record["worker_slot"]
        assert request["tool_execution_profile"]["intra_task_tool_concurrency"] == 1
        assert request["tool_execution_profile"]["max_output_chars"] == expected_output_chars
        assert decision["request_id"] == request["request_id"]
        assert decision["llm_response_id"] == f"chatcmpl-{job['job_id']}-{stage}"
        assert decision["label_stage"] == "after_response"
        assert decision["next_tool_name"] == fixture_action(job["adapter"], stage)[0]
        assert decision["has_environment_tool_call"] is (stage < 2)
        assert decision["seq"] > request["seq"]
    for stage, (start, end) in enumerate(zip(starts, ends, strict=True)):
        assert start["request_id"] == end["request_id"] == requests[stage]["request_id"]
        assert start["tool_call_id"] == end["tool_call_id"] == f"call-{job['job_id']}-{stage}"
        assert start["tool_name"] == end["tool_name"] == fixture_action(job["adapter"], stage)[0]
        for key, value in fixture_action(job["adapter"], stage)[1].items():
            assert start["arguments"][key] == value
        assert end["round_trip_ms"] >= 0 and end["executor_duration_ms"] >= 0
        assert start["seq"] < end["seq"]
    if job["adapter"] in CODE_KINDS:
        assert result["code_environment_closed"] is True
        solution = (attempt / "artifacts/solution.py").read_text()
        assert f"fixture_identity = {job['job_id']!r}" in solution
        assert "value = 1" in solution
    return {
        "task_id": job["job_id"],
        "adapter": job["adapter"],
        "attempt": str(attempt),
        "conversation_id": actor["conversation_id"],
        "worker_slot": record["worker_slot"],
        "request_ids": [row["request_id"] for row in requests],
        "tool_mask": sorted(TOOL_MASKS[job["adapter"]]),
        "max_output_chars": expected_output_chars,
        "t0_peak_llm_inflight": max(
            row["t0_features"]["client_load"]["llm_inflight"] for row in requests
        ),
    }


def test_real_sdk_four_same_profile_and_four_mixed_fixtures(tmp_path, monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    corpus = tmp_path / "public-fixture.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"docid": name.lower(), "title": name, "sentences": [f"{name} evidence."]})
            + "\n"
            for name in ["Alpha", "Beta", "Gamma"]
        )
    )
    index = tmp_path / "public-fixture.sqlite3"
    build_index(corpus, index, "public-fixture-v1")
    summaries = []
    for scenario, kinds in [
        ("same", ["quixbugs"] * 5),
        ("mixed", ["quixbugs", "livecodebench", "hotpot", "browsecomp"]),
    ]:
        root = tmp_path / scenario
        root.mkdir()
        jobs = [
            {
                "job_id": f"{scenario}-{position}",
                "adapter": kind,
                "scenario": scenario,
                "root": str(root),
                "index": str(index),
                "queue_position": position,
                "controller_timeout_s": 180,
            }
            for position, kind in enumerate(kinds)
        ]
        state = {"counts": Counter(), "requests": [], "errors": [], "active": 0, "max_active": 0}
        server = ThreadingHTTPServer(("127.0.0.1", 0), fixture_handler(jobs, state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for job in jobs:
            job["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        try:
            report = run_parallel(jobs, sdk_fixture_worker, concurrency=4)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        (root / "http_requests.json").write_text(json.dumps(state["requests"], indent=2))
        (root / "scheduler_report.json").write_text(json.dumps(report, indent=2))
        assert not state["errors"], state["errors"]
        assert state["max_active"] == report["peak_active_workers"] == 4
        assert len(report["records"]) == len(jobs)
        by_id = {row["job_id"]: row for row in report["records"]}
        rows = [
            checked_trace(
                job,
                by_id[job["job_id"]],
                [row for row in state["requests"] if row["task_id"] == job["job_id"]],
            )
            for job in jobs
        ]
        assert len({row["conversation_id"] for row in rows}) == len(jobs)
        assert (
            len({request_id for row in rows for request_id in row["request_ids"]}) == len(jobs) * 3
        )
        assert max(row["t0_peak_llm_inflight"] for row in rows) == 4
        if scenario == "same":
            fifth = by_id["same-4"]
            previous = next(
                row
                for row in report["records"]
                if row["worker_slot"] == fifth["worker_slot"] and row["job_id"] != "same-4"
            )
            assert fifth["dispatched_monotonic_ns"] > previous["finished_monotonic_ns"]
        summaries.append(
            {
                "scenario": scenario,
                "fixture_tasks": len(jobs),
                "physical_http_requests": len(state["requests"]),
                "max_simultaneous_http_requests": state["max_active"],
                "peak_active_workers": report["peak_active_workers"],
                "tasks": rows,
            }
        )
    summary = {
        "status": "passed",
        "real_openhands_sdk": True,
        "qwen_used": False,
        "gpu_service_started": False,
        "benchmark_tasks_run": 0,
        "fixture_tasks": 9,
        "physical_http_requests": 27,
        "retrieval_corpus_documents": 3,
        "sampling_parameters_verified_over_http": SAMPLING_PARAMETERS,
        "code_environment": "controlled public fixture, not production sandbox",
        "checks": [
            "same-profile C4 and slot refill",
            "four-family C4",
            "isolated workspaces and traces",
            "per-adapter tool masks",
            "causal T0 client load",
            "request blob hashes and raw HTTP links",
            "response labels",
            "tool arguments and durations",
            "only code tools advertise the applied max_output_chars limit",
        ],
        "scenarios": summaries,
    }
    (tmp_path / "microcheck_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
