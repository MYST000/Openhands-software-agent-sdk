import json
import os
import subprocess
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"


@pytest.mark.parametrize("kind", ["swe", "quixbugs"])
@pytest.mark.parametrize("malformed_first", [False, True])
def test_real_sdk_model_transport_tools_patch_and_trace(tmp_path, malformed_first, kind):
    from benchmark_adapters.config import Config, DatasetConfig, LLMConfig, RuntimeConfig
    from benchmark_adapters.contracts import CommandResult
    from benchmark_adapters.environment import command_argv
    from benchmark_adapters.runner import run_task
    from benchmark_adapters.swe import SWEAdapter, export_patch_command

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "answer.py").write_text("value = 0\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            if malformed_first and len(received) == 1:
                tool, args = (
                    ("swe_file_editor" if kind == "swe" else "code_file_editor"),
                    dict(command="NOT_VALID", summary="Malformed call"),
                )
            elif len(received) == 1 + int(malformed_first):
                tool, args = (
                    "swe_file_editor" if kind == "swe" else "code_file_editor",
                    dict(
                        command="str_replace",
                        path="answer.py",
                        old_str="value = 0",
                        new_str="value = 1",
                        summary="Fix value",
                    ),
                )
            else:
                tool, args = "finish", dict(message="Updated value.", summary="Finish")
            payload = dict(
                id=f"chatcmpl-{len(received)}",
                object="chat.completion",
                created=1,
                model="test-model",
                choices=[
                    dict(
                        index=0,
                        finish_reason="tool_calls",
                        message=dict(
                            role="assistant",
                            content=None,
                            tool_calls=[
                                dict(
                                    id=f"call-{len(received)}",
                                    type="function",
                                    function=dict(name=tool, arguments=json.dumps(args)),
                                )
                            ],
                        ),
                    )
                ],
                usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            )
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class TestEnvironment:
        repo_dir = str(repo)
        baseline_commit = base
        closed = False

        def prepare(self):
            return {
                "backend": "test-fixture",
                "prepared_head": base,
                "environment_id": "environment-fixture",
                "container_id": "container-fixture",
            }

        def execute(self, command, timeout=30, max_bytes=16000, **kwargs):
            data = json.loads(
                subprocess.check_output(
                    command_argv(command, str(repo), timeout, max_bytes), text=True
                )
            )
            return CommandResult(**data)

        def export_patch(self):
            return subprocess.check_output(
                ["bash", "-c", export_patch_command(base)], cwd=repo, text=True
            )

        def close(self):
            self.closed = True

        def quiesce(self):
            pass

    env = TestEnvironment()
    config = Config(
        dataset=DatasetConfig(kind=kind, path="unused", revision="fixture"),
        llm=LLMConfig(
            model="openai/test-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            num_retries=0,
        ),
        runtime=replace(RuntimeConfig(), max_iterations=4, runs_dir=str(tmp_path / "runs")),
    )
    task = SWEAdapter.from_record(
        dict(
            instance_id="org__repo-1",
            repo="org/repo",
            base_commit=base,
            problem_statement="Set value to one.",
            patch="GOLD_DO_NOT_LEAK",
        ),
        dataset_id="princeton-nlp/SWE-bench",
        revision="fixture",
        split="dev",
    )
    if kind != "swe":
        task = replace(task, public_metadata={**task.public_metadata, "solution_path": "answer.py"})
    try:
        result = run_task(config, task, tmp_path / "attempt", environment=env)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result["execution_status"] == "completed", result
    assert env.closed
    assert len(received) == 2 + int(malformed_first)
    submission = json.loads((tmp_path / "attempt" / "submission.json").read_text())
    if kind == "swe":
        assert "+value = 1" in submission["model_patch"]
    else:
        assert "value = 1" in submission["code"]
    events = [
        json.loads(line)
        for line in (tmp_path / "attempt" / "events.jsonl").read_text().splitlines()
    ]
    assert not [e for e in events if e["event"] == "tool_error"]
    observations = [m["content"] for m in received[-1]["messages"] if m["role"] == "tool"]
    assert any("exit_code=0" in str(content) for content in observations)
    starts = [e for e in events if e["event"] == "tool_start"]
    assert starts[0]["tool_call_id"] == f"call-{1 + int(malformed_first)}"
    assert starts[0]["request_id"]
    assert starts[0]["container_id"] == "container-fixture"
    assert starts[0]["environment_id"] == "environment-fixture"
    assert result["container_id"] == "container-fixture"
    decisions = [e for e in events if e["event"] == "response_action_summary"]
    assert decisions[-1]["next_action_kind"] == "finish"
    assert decisions[-1]["has_environment_tool_call"] is False
    assert decisions[-2]["has_environment_tool_call"] is True
    cleanup = [e for e in events if e["event"] == "environment_cleanup"]
    assert len(cleanup) == 1
    assert cleanup[0]["container_id"] == "container-fixture"
    requests = [e for e in events if e["event"] == "llm_request_prepared"]
    for request, actual in zip(requests, received, strict=True):
        saved = json.loads((tmp_path / "attempt" / request["request"]["path"]).read_text())
        assert saved["messages"] == actual["messages"]
        assert saved["tools"] == actual["tools"]
    assert not any(
        "GOLD_DO_NOT_LEAK" in p.read_text() for p in (tmp_path / "attempt").rglob("*.json")
    )


def test_recorded_llm_disables_unobserved_http_retries(tmp_path):
    from openhands.sdk import Message, TextContent

    from benchmark_adapters.sdk_bridge import RecordedLLM
    from benchmark_adapters.tracing import Budget, TraceRecorder

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            payload = json.dumps(
                {"error": {"message": "fixture failure", "type": "server_error"}}
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    recorder = TraceRecorder(tmp_path, {"task_id": "fixture"})
    llm = RecordedLLM(
        model="openai/test-model",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        api_key="dummy",
        num_retries=0,
        timeout=3,
    )
    llm.attach(recorder, Budget(5, 5, 20))
    try:
        with pytest.raises(Exception):
            llm.completion(messages=[Message(role="user", content=[TextContent(text="hi")])])
    finally:
        recorder.close()
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(calls) == 1


def test_real_sdk_timeout_wrapper_is_censored_by_audit(tmp_path):
    import time

    from benchmark_adapters.code_audit import audit_attempt
    from benchmark_adapters.config import Config, DatasetConfig, LLMConfig, RuntimeConfig
    from benchmark_adapters.contracts import CommandResult, Task
    from benchmark_adapters.environment import command_argv
    from benchmark_adapters.runner import run_task

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solution.py").write_text("pass\n")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            time.sleep(2)
            self.send_response(500)
            self.end_headers()

        def log_message(self, *args):
            pass

    class Env:
        repo_dir = str(repo)

        def prepare(self):
            return {"environment_id": "fixture-environment"}

        def execute(self, command, timeout=30, max_bytes=16000):
            return CommandResult(
                **json.loads(
                    subprocess.check_output(
                        command_argv(command, str(repo), timeout, max_bytes), text=True
                    )
                )
            )

        def quiesce(self):
            pass

        def close(self):
            return {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = Config(
        dataset=DatasetConfig(kind="quixbugs"),
        llm=LLMConfig(
            model="openai/test-model",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            num_retries=0,
            timeout=1,
        ),
        runtime=replace(RuntimeConfig(), task_timeout=30, max_iterations=2),
    )
    task = Task(
        "fixture", "v", "dev", "timeout-fixture", "Say done", {"solution_path": "solution.py"}
    )
    try:
        result = run_task(config, task, tmp_path / "attempt", environment=Env())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert len(calls) == 1
    assert "LLMTimeoutError" in result["errors"]
    report = audit_attempt(tmp_path / "attempt")
    assert report["audit_passed"], report
    assert len(report["budget_censored_request_ids"]) == 1
