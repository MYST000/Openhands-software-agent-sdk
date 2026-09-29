import json
import subprocess
import time

import pytest


def test_command_timeout_and_bounded_output(tmp_path):
    from benchmark_adapters.environment import command_argv

    result = subprocess.run(
        command_argv("printf hello; sleep 3", str(tmp_path), 0.1, 100),
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    data = json.loads(result.stdout)
    assert data["timed_out"] is True
    assert data["stdout"] == "hello"
    assert data["duration_ms"] >= 90
    data = json.loads(
        subprocess.check_output(command_argv("printf 123456789", str(tmp_path), 1, 4), text=True)
    )
    assert data["stdout"] == "1234" and data["truncated"] is True


def test_file_edit_is_literal_and_cannot_follow_symlink_outside_repo(tmp_path):
    from benchmark_adapters.environment import file_operation_command

    repo = tmp_path / "repo"
    repo.mkdir()
    text = "literal $(touch BAD) `touch BAD2`\n"
    command = file_operation_command(
        str(repo), dict(command="create", path="a.txt", file_text=text)
    )
    subprocess.run(["bash", "-c", command], cwd=repo, check=True)
    assert (repo / "a.txt").read_text() == text
    assert not (repo / "BAD").exists()
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    (repo / "link").symlink_to(outside)
    command = file_operation_command(
        str(repo), dict(command="str_replace", path="link", old_str="preserve", new_str="oops")
    )
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert result.returncode != 0
    assert outside.read_text() == "preserve"


def test_trace_preserves_prompt_keys_but_omits_transport_credentials(tmp_path):
    from benchmark_adapters.tracing import TraceRecorder

    recorder = TraceRecorder(tmp_path, {"task_id": "t"})
    recorder.request(
        "r",
        {
            "messages": [{"role": "user", "content": "api_key is a variable in my code"}],
            "api_key": "SECRET",
            "extra_headers": {"Authorization": "SECRET"},
            "tools": [
                {
                    "function": {
                        "name": "f",
                        "parameters": {"properties": {"api_key": {"type": "string"}}},
                    }
                }
            ],
        },
    )
    recorder.close()
    text = "".join(p.read_text() for p in tmp_path.rglob("*.json*"))
    assert "SECRET" not in text
    assert "api_key is a variable" in text
    assert '"api_key": {"type": "string"}' in text


def test_budget_rejects_call_after_deadline():
    from benchmark_adapters.tracing import Budget, BudgetExceeded

    budget = Budget(max_tools=1, max_requests=1, timeout=10)
    budget.tool()
    with pytest.raises(BudgetExceeded):
        budget.tool()
    assert budget.reason == "max_tool_calls"


def test_foreground_only_command_does_not_leave_background_writers(tmp_path):
    from benchmark_adapters.environment import command_argv

    subprocess.run(
        command_argv("(sleep 0.3; echo later > late.txt) &", str(tmp_path), 2, 100),
        capture_output=True,
        check=True,
    )
    time.sleep(0.5)
    assert not (tmp_path / "late.txt").exists()


@pytest.mark.parametrize(
    "names,expected,has_environment",
    [
        ([], "no_tool", False),
        (["finish"], "finish", False),
        (["think"], "think", False),
        (["swe_terminal"], "environment_tool", True),
        (["think", "swe_file_editor"], "think", True),
        (["invented_tool"], "unknown_tool", None),
    ],
)
def test_response_decision_distinguishes_control_no_tool_and_environment(
    tmp_path, names, expected, has_environment
):
    from benchmark_adapters.tracing import TraceRecorder

    recorder = TraceRecorder(tmp_path, {"task_id": "t"})
    recorder.environment_tool_names = {"swe_terminal", "swe_file_editor"}
    calls = [
        dict(id=f"call-{i}", type="function", function=dict(name=name, arguments="{}"))
        for i, name in enumerate(names)
    ]
    recorder.response_decision(
        "request-1",
        {
            "id": "response-1",
            "choices": [
                {
                    "finish_reason": "tool_calls" if names else "stop",
                    "message": {"tool_calls": calls},
                }
            ],
        },
    )
    recorder.close()
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["event"] == "response_action_summary"
    assert event["next_action_kind"] == expected
    assert event["has_environment_tool_call"] is has_environment
    assert event["has_tool_calls"] == bool(names)
    assert event["request_id"] == "request-1"
    assert event["tool_calls"] == calls


def test_incomplete_model_response_does_not_create_no_tool_negative(tmp_path):
    from benchmark_adapters.tracing import TraceRecorder

    recorder = TraceRecorder(tmp_path, {})
    recorder.response_decision("request-1", {"id": "r", "choices": []})
    recorder.close()
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["has_environment_tool_call"] is None
    assert event["next_action_kind"] == "unavailable"
