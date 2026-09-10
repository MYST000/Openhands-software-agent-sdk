from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from litellm.types.utils import ModelResponse

from openhands.sdk import Agent, LocalConversation, LocalWorkspace
from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.flowpilot import (
    FlowPilotConfig,
    FlowPilotRuntime,
    context_digest,
)
from openhands.sdk.io import InMemoryFileStore
from openhands.sdk.llm import (
    LLM,
    LLMCallContext,
    LLMResponse,
    Message,
    MessageToolCall,
    TextContent,
)
from openhands.sdk.llm.options.chat_options import select_chat_options
from openhands.sdk.llm.options.responses_options import select_responses_options
from openhands.sdk.llm.utils.metrics import MetricsSnapshot
from openhands.sdk.tool import ToolAnnotations, ToolDefinition
from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.schema import Action, Observation
from openhands.sdk.tool.spec import Tool
from openhands.sdk.tool.tool import ToolExecutor


class ReusableSearchAction(Action):
    query: str


class ReusableSearchObservation(Observation):
    items: list[dict[str, str]]


class _SearchExecutor(ToolExecutor[ReusableSearchAction, ReusableSearchObservation]):
    def __init__(self, on_call: Callable[[], None]):
        self._on_call = on_call

    def __call__(
        self,
        action: ReusableSearchAction,
        conversation: LocalConversation | None = None,
    ) -> ReusableSearchObservation:
        self._on_call()
        return ReusableSearchObservation(items=[{"title": f"local:{action.query}"}])


class _SearchTool(ToolDefinition[ReusableSearchAction, ReusableSearchObservation]):
    name: ClassVar[str] = "flowpilot_test_search"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence[_SearchTool]:
        raise AssertionError("tests register a fixed Tool instance")


class LocalReadAction(Action):
    query: str


class LocalReadObservation(Observation):
    result: str


class _LocalReadExecutor(ToolExecutor[LocalReadAction, LocalReadObservation]):
    def __init__(self, on_call: Callable[[], None]):
        self._on_call = on_call

    def __call__(
        self,
        action: LocalReadAction,
        conversation: LocalConversation | None = None,
    ) -> LocalReadObservation:
        self._on_call()
        return LocalReadObservation(result=f"local:{action.query}")


class _LocalReadTool(ToolDefinition[LocalReadAction, LocalReadObservation]):
    name: ClassVar[str] = "flowpilot_test_local_read"

    @classmethod
    def create(cls, *args, **kwargs) -> Sequence[_LocalReadTool]:
        raise AssertionError("tests register a fixed Tool instance")


def _llm_response(message: Message, response_id: str) -> LLMResponse:
    return LLMResponse(
        message=message,
        metrics=MetricsSnapshot(model_name="gpt-4o"),
        raw_response=ModelResponse(
            id=response_id,
            choices=[],
            model="gpt-4o",
            object="chat.completion",
        ),
    )


def _tool_response(call_id: str, query: str) -> LLMResponse:
    return _llm_response(
        Message(
            role="assistant",
            tool_calls=[
                MessageToolCall(
                    id=call_id,
                    name=_SearchTool.name,
                    arguments=json.dumps({"query": query}),
                    origin="completion",
                )
            ],
        ),
        f"response-{call_id}",
    )


def _local_tool_response(call_id: str, query: str) -> LLMResponse:
    return _llm_response(
        Message(
            role="assistant",
            tool_calls=[
                MessageToolCall(
                    id=call_id,
                    name=_LocalReadTool.name,
                    arguments=json.dumps({"query": query}),
                    origin="completion",
                )
            ],
        ),
        f"response-{call_id}",
    )


def _content_response(text: str, response_id: str = "response-final") -> LLMResponse:
    return _llm_response(
        Message(role="assistant", content=[TextContent(text=text)]), response_id
    )


class _DCSControl:
    def __init__(
        self,
        conversation: LocalConversation,
        decisions: list[dict],
    ):
        self.conversation = conversation
        self.decisions = iter(decisions)
        self.request_snapshot: dict | None = None
        self.pending_messages: list[dict] = []
        self.ack_payloads: list[dict] = []
        self.sync_payloads: list[dict] = []
        self.sync_visible_event_types: list[list[str]] = []
        self.acked = False
        self.released = False

    def request_json(self, path: str, payload: dict | None, *, method: str) -> dict:
        assert method in {"GET", "POST"}
        if path.endswith("/dcs/delegations"):
            assert payload is not None
            self.request_snapshot = deepcopy(payload["request_snapshot"])
            return {"delta_digest": payload["base_context_digest"]}
        if path.endswith("/dcs/reuse/resolve"):
            return next(self.decisions)
        if path.endswith("/dcs/delegations/release"):
            self.released = True
            return {"state": "aborted", "pending_message_count": 0}
        if path.endswith("/dcs/deltas/append"):
            assert payload is not None
            self.pending_messages.extend(deepcopy(payload["messages"]))
            return {
                "delta_digest": "b" * 64,
                "last_seq": len(self.pending_messages),
                "state": "open",
                "barrier_reason": None,
            }
        if path.endswith("/dcs/continuations"):
            assert self.request_snapshot is not None
            body = deepcopy(self.request_snapshot)
            body["messages"].extend(deepcopy(self.pending_messages))
            return {"body": body}
        if path.endswith("/dcs/sync"):
            assert payload is not None
            self.sync_payloads.append(deepcopy(payload))
            events = list(self.conversation.state.active_branch())
            self.sync_visible_event_types.append(
                [event.__class__.__name__ for event in events]
            )
            messages = [
                *deepcopy(self.pending_messages),
                *deepcopy(payload["barrier_messages"]),
            ]
            return {
                "first_seq": 1,
                "last_seq": len(messages),
                "delta_digest": "c" * 64,
                "wal_delta_digest": "c" * 64,
                "messages": messages,
            }
        if path.endswith("/dcs/sync/ack"):
            assert payload is not None
            self.ack_payloads.append(deepcopy(payload))
            self.acked = True
            return {"pending_message_count": 0, "state": "acked"}
        if path.endswith("/flowpilot/v1/reuse/resolve"):
            return {
                "decision": "sync_and_execute_as_leader",
                "binding_id": "binding-local",
            }
        if "/reuse/bindings/binding-local/" in path:
            return {}
        if path.endswith("/dcs/reconcile"):
            assert payload is not None and self.ack_payloads
            assert (
                payload["context_cursor"] == self.ack_payloads[-1]["new_context_cursor"]
            )
            assert (
                payload["context_digest"] == self.ack_payloads[-1]["new_context_digest"]
            )
            return {"status": "in_sync", "sync_required": False}
        raise AssertionError(path)


def _dcs_conversation(
    tmp_path: Path,
    *,
    on_execute: Callable[[], None],
) -> LocalConversation:
    register_tool(
        _SearchTool.name,
        _SearchTool(
            description="Read-only test search",
            action_type=ReusableSearchAction,
            observation_type=ReusableSearchObservation,
            annotations=ToolAnnotations(readOnlyHint=True),
            executor=_SearchExecutor(on_execute),
        ),
    )
    register_tool(
        _LocalReadTool.name,
        _LocalReadTool(
            description="Read-only local test tool",
            action_type=LocalReadAction,
            observation_type=LocalReadObservation,
            annotations=ToolAnnotations(readOnlyHint=True),
            executor=_LocalReadExecutor(on_execute),
        ),
    )
    agent = Agent(
        llm=LLM(model="gpt-4o", caching_prompt=False),
        tools=[Tool(name=_SearchTool.name), Tool(name=_LocalReadTool.name)],
        include_default_tools=[],
        tool_concurrency_limit=1,
    )
    with patch.object(FlowPilotRuntime, "register"):
        conversation = LocalConversation(
            agent=agent,
            workspace=LocalWorkspace(working_dir=tmp_path),
            flowpilot=_config(
                exact_reuse_enabled=True,
                deferred_context_enabled=True,
                reusable_web_tools=(_SearchTool.name,),
            ),
            visualizer=None,
        )
    conversation.send_message("Find FlowPilot")
    return conversation


def _config(**updates) -> FlowPilotConfig:
    values = {
        "enabled": True,
        "gateway_url": "http://flowpilot:9000",
        "api_key": "secret",
        "job_id": "job-1",
        "line_id": "line-1",
    }
    values.update(updates)
    return FlowPilotConfig(**values)


def test_disabled_configuration_preserves_existing_behavior() -> None:
    FlowPilotConfig().validate(tool_concurrency_limit=8)
    llm = LLM(model="gpt-4o", base_url="http://provider")
    options = select_chat_options(llm, {}, has_tools=True)
    assert not any(
        key.startswith("x-flowpilot-") for key in options.get("extra_headers", {})
    )


def test_identity_attempts_keep_logical_request_and_rotate_transport_call() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    identity = runtime.begin_request(
        context_sequence=0, base_context_cursor="root", context_digest="a" * 64
    )
    first = runtime.prepare_attempt()
    second = runtime.prepare_attempt()
    assert first["x-flowpilot-request-id"] == second["x-flowpilot-request-id"]
    assert first["x-flowpilot-llm-call-id"] != second["x-flowpilot-llm-call-id"]
    assert first["x-flowpilot-request-attempt"] == "1"
    assert second["x-flowpilot-request-attempt"] == "2"
    assert identity.request_id == first["x-flowpilot-request-id"]


def test_child_config_preserves_root_job_and_records_parent_line() -> None:
    config = _config()
    child = config.child(
        line_id="line-child",
        parent_conversation_id="conversation-root",
        spawn_id="spawn-1",
    )
    assert child.job_id == config.job_id
    assert child.parent_line_id == config.line_id
    assert child.parent_conversation_id == "conversation-root"


def test_enabled_configuration_requires_serial_tool_execution() -> None:
    _config().validate(tool_concurrency_limit=1)
    with pytest.raises(ValueError, match="tool_concurrency_limit == 1"):
        _config().validate(tool_concurrency_limit=2)

    with pytest.raises(ValueError, match="requires exact reuse"):
        _config(deferred_context_enabled=True).validate(tool_concurrency_limit=1)

    with pytest.raises(ValueError, match="semantic reuse requires exact reuse"):
        _config(semantic_reuse_enabled=True).validate(tool_concurrency_limit=1)


@pytest.mark.parametrize("selector", ["chat", "responses"])
def test_dynamic_headers_are_used_only_for_the_flowpilot_gateway(selector: str) -> None:
    headers = {"x-flowpilot-llm-call-id": "call-1"}
    context = LLMCallContext(
        flowpilot_headers=headers,
        flowpilot_gateway_url="http://flowpilot:9000",
    )
    llm = LLM(model="gpt-4o", base_url="http://flowpilot:9000")
    if selector == "chat":
        options = select_chat_options(llm, {}, has_tools=True, call_context=context)
    else:
        options = select_responses_options(
            llm, {}, include=None, store=None, call_context=context
        )
    assert options["extra_headers"]["x-flowpilot-llm-call-id"] == "call-1"

    direct = llm.model_copy(update={"base_url": "http://provider"})
    if selector == "chat":
        options = select_chat_options(direct, {}, has_tools=True, call_context=context)
    else:
        options = select_responses_options(
            direct, {}, include=None, store=None, call_context=context
        )
    assert "x-flowpilot-llm-call-id" not in options.get("extra_headers", {})


def test_runtime_registers_job_and_line_without_prompt_content() -> None:
    requests: list[tuple[str, dict]] = []
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    with (
        patch.object(
            FlowPilotRuntime,
            "_post_control",
            side_effect=lambda path, payload: requests.append((path, payload)),
        ),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
    ):
        runtime.register(
            context_sequence=3,
            base_context_cursor="event-3",
            context_digest="a" * 64,
        )
    assert [url.rsplit("/", 1)[-1] for url, _ in requests] == ["jobs", "lines"]
    assert requests[-1][1]["context_sequence"] == 3
    assert requests[-1][1]["base_context_cursor"] == "event-3"
    serialized = json.dumps(requests)
    assert "prompt" not in serialized
    assert "secret" not in serialized


def test_runtime_registers_explicit_parent_child_metadata_and_dependencies() -> None:
    requests: list[tuple[str, dict]] = []
    dependency_requests: list[tuple[str, dict]] = []
    runtime = FlowPilotRuntime(
        _config(
            parent_conversation_id="conversation-root",
            task_id="task-7",
            agent_id="agent-7",
            parent_action_id="action-parent-7",
        ),
        "conversation-child",
    )
    with (
        patch.object(
            FlowPilotRuntime,
            "_post_control",
            side_effect=lambda path, payload: requests.append((path, payload)),
        ),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
        patch.object(
            FlowPilotRuntime,
            "_request_json",
            side_effect=lambda path, payload, **_kwargs: (
                dependency_requests.append((path, payload)) or {}
            ),
        ),
    ):
        runtime.register()
        runtime.report_dependency("line-parent", actual_wait=False)
        runtime.report_dependency("line-parent", actual_wait=True)

    line_payload = next(
        payload for path, payload in requests if path.endswith("/lines")
    )
    assert line_payload["conversation_id"] == "conversation-child"
    assert line_payload["parent_conversation_id"] == "conversation-root"
    assert line_payload["task_id"] == "task-7"
    assert line_payload["agent_id"] == "agent-7"
    assert line_payload["parent_action_id"] == "action-parent-7"
    dependency_payloads = [
        payload
        for path, payload in dependency_requests
        if path.endswith("/dependencies")
    ]
    assert dependency_payloads[0]["prerequisite_line_ids"] == ()
    assert dependency_payloads[1]["prerequisite_line_ids"] == ("line-parent",)


def test_runtime_register_recovers_authoritative_tail_version() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    with (
        patch.object(FlowPilotRuntime, "_post_control") as post_control,
        patch.object(
            FlowPilotRuntime,
            "_find_authoritative_tail",
            return_value={"phase": "READY", "version": 7},
        ),
    ):
        runtime.register()
    assert runtime.tail_version == 7
    assert [call.args[0] for call in post_control.call_args_list] == [
        "/flowpilot/v1/jobs"
    ]


def test_context_digest_binds_event_content() -> None:
    first = MessageEvent(
        id="event-1",
        timestamp="2026-08-17T00:00:00",
        parent_id=None,
        source="user",
        llm_message=Message(role="user", content=[TextContent(text="first")]),
    )
    second = first.model_copy(
        update={
            "llm_message": Message(role="user", content=[TextContent(text="different")])
        }
    )
    assert context_digest([first]) != context_digest([second])


def test_local_conversation_enables_only_with_serial_tools(tmp_path) -> None:
    with patch.object(FlowPilotRuntime, "register"):
        conversation = LocalConversation(
            agent=Agent(
                llm=LLM(model="gpt-4o", base_url="http://provider"),
                tools=[],
                tool_concurrency_limit=1,
            ),
            workspace=LocalWorkspace(working_dir=tmp_path),
            flowpilot=_config(),
        )
    assert conversation.agent.llm.base_url == "http://flowpilot:9000/v1"
    conversation.close()

    with pytest.raises(ValueError, match="tool_concurrency_limit == 1"):
        LocalConversation(
            agent=Agent(llm=LLM(model="gpt-4o"), tools=[], tool_concurrency_limit=2),
            workspace=LocalWorkspace(working_dir=tmp_path),
            flowpilot=_config(),
        )


def test_tool_telemetry_is_best_effort_and_keeps_identity() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    assert identity.headers("key")["x-flowpilot-context-sequence"] == "1"
    action = ActionEvent(
        thought=[],
        action=None,
        tool_name="terminal",
        tool_call_id="tool-call-1",
        tool_call=MessageToolCall(
            id="tool-call-1",
            name="terminal",
            arguments="{}",
            origin="completion",
        ),
        llm_response_id="response-1",
    )
    captured: list[dict] = []

    def capture(payload):
        captured.append(payload)

    with patch.object(FlowPilotRuntime, "_post_tool_event", side_effect=capture):
        token = runtime.tool_start(action)
        runtime.tool_terminal(action, token=token, started=time.monotonic(), events=[])
        runtime.tool_terminal(action, token=token, started=time.monotonic(), events=[])
    assert [item["event_kind"] for item in captured] == ["start", "finish"]
    assert {item["tool_call_id"] for item in captured} == {"tool-call-1"}
    assert set(captured[0]) == {
        "protocol_version",
        "job_id",
        "line_id",
        "context_epoch",
        "tail_request_id",
        "request_id",
        "llm_call_id",
        "attempt",
        "conversation_id",
        "action_id",
        "tool_call_id",
        "tool_name",
        "tool_class",
        "execution_attempt",
        "event_id",
        "sequence",
        "event_kind",
    }

    with patch("urllib.request.urlopen", side_effect=OSError("offline")):
        runtime._post_tool_event({"event_kind": "start"})


def test_runtime_uses_authoritative_tail_version() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    authoritative = {
        "tail_request_id": identity.tail_request_id,
        "llm_call_id": identity.llm_call_id,
        "phase": "BLOCKED",
        "version": 4,
    }
    with patch.object(
        FlowPilotRuntime, "_fetch_authoritative_tail", return_value=authoritative
    ):
        runtime.commit_request(identity)
    assert runtime.tail_version == 4
    assert runtime.active_identity is None
    assert runtime._tool_identity == identity


def test_runtime_rejects_non_authoritative_llm_completion() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    with (
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            return_value={
                "tail_request_id": "other-tail",
                "llm_call_id": identity.llm_call_id,
                "state": "NEXT_READY",
                "version": 1,
            },
        ),
        pytest.raises(RuntimeError, match="authoritative"),
    ):
        runtime.commit_request(identity)
    assert runtime.tail_version == 0


def test_runtime_abort_reconciles_rolled_back_version() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1", tail_version=3)
    identity = runtime.begin_request(
        context_sequence=2,
        base_context_cursor="cursor-2",
        context_digest="b" * 64,
    )
    with patch.object(
        FlowPilotRuntime,
        "_fetch_authoritative_tail",
        return_value={"version": 2},
    ):
        runtime.abort_request(identity)
    assert runtime.tail_version == 2
    assert runtime.active_identity is None


def test_blocked_tool_telemetry_does_not_emit_start() -> None:
    runtime = FlowPilotRuntime(_config(), "conversation-1")
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    action = ActionEvent(
        thought=[],
        action=None,
        tool_name="terminal",
        tool_call_id="tool-call-1",
        tool_call=MessageToolCall(
            id="tool-call-1",
            name="terminal",
            arguments="{}",
            origin="completion",
        ),
        llm_response_id="response-1",
    )
    captured: list[dict] = []
    with patch.object(
        FlowPilotRuntime, "_post_tool_event", side_effect=captured.append
    ):
        runtime.tool_blocked(action)
    assert [event["event_kind"] for event in captured] == ["blocked"]
    assert captured[0]["error_class"] == "HookBlocked"
    assert "rejection_reason" not in captured[0]


def _reuse_action() -> ActionEvent:
    return ActionEvent(
        thought=[],
        action=ReusableSearchAction(query="flowpilot"),
        tool_name="web_search",
        tool_call_id="tool-call-1",
        tool_call=MessageToolCall(
            id="tool-call-1",
            name="web_search",
            arguments='{"query":"flowpilot"}',
            origin="completion",
        ),
        llm_response_id="response-1",
    )


def _reuse_runtime() -> FlowPilotRuntime:
    runtime = FlowPilotRuntime(
        _config(exact_reuse_enabled=True, reusable_web_tools=("web_search",)),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    return runtime


def test_exact_hit_returns_validated_observation_with_safe_provenance() -> None:
    runtime = _reuse_runtime()
    response = {
        "decision": "sync_with_reused_result",
        "result": {
            "content": [{"text": "result", "type": "text"}],
            "is_error": False,
            "items": [{"title": "FlowPilot"}],
        },
        "provenance": {
            "reuse_type": "historical",
            "observed_at": "2026-08-13T00:00:00Z",
            "result_schema_version": "1",
            "source_query_digest": "a" * 64,
            "binding_id": "must-not-leak",
            "similarity_score": 1.0,
        },
    }
    with patch.object(
        FlowPilotRuntime, "_request_json", return_value=response
    ) as request_json:
        observation = runtime.resolve_reuse(_reuse_action(), ReusableSearchObservation)
    assert isinstance(observation, ReusableSearchObservation)
    assert observation.items == [{"title": "FlowPilot"}]
    text = observation.text
    assert "historical" in text
    assert "must-not-leak" not in text
    assert "similarity_score" not in text
    assert request_json.call_args.args[1]["arguments"] == {"query": "flowpilot"}


def test_semantic_opt_in_uses_phase3_and_progress_is_best_effort() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            semantic_reuse_enabled=True,
            reusable_web_tools=("web_search",),
        ),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    action = _reuse_action()
    requests: list[tuple[str, dict | None, str]] = []

    def request_json(path, payload, *, method):
        requests.append((path, payload, method))
        if path.endswith("/resolve"):
            return {
                "decision": "sync_and_execute_as_leader",
                "binding_id": "binding-1",
            }
        if path.endswith("/progress"):
            return {"status": "accepted"}
        raise AssertionError(path)

    with (
        patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
    ):
        assert runtime.resolve_reuse(action, ReusableSearchObservation) is None
        runtime.tool_start(action)

    assert requests[0][1] is not None
    assert requests[0][1]["protocol_version"] == "flowpilot-phase3-reuse-v2"
    assert requests[1][0].endswith("/bindings/binding-1/progress")
    assert requests[1][1] is not None
    assert requests[1][1]["protocol_version"] == "flowpilot-phase3-reuse-v2"


def test_follower_cancellation_releases_waiting_binding() -> None:
    runtime = _reuse_runtime()
    action = _reuse_action()
    tool_identity = runtime._tool_identity
    assert tool_identity is not None
    identity = runtime._reuse_identity(tool_identity, action)
    cancel_event = threading.Event()
    runtime._reuse_cancellations[action.id] = cancel_event
    runtime._waiting_reuses[action.id] = ("binding-1", identity)
    requests: list[tuple[str, dict | None, str]] = []

    def request_json(path, payload, *, method):
        requests.append((path, payload, method))
        return {}

    with patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json):
        runtime.tool_cancel(action)

    assert cancel_event.is_set()
    assert action.id not in runtime._waiting_reuses
    assert requests == [
        (
            "/flowpilot/v1/reuse/bindings/binding-1/cancel",
            {
                "protocol_version": "flowpilot-phase1-reuse-v2",
                "binding_id": "binding-1",
                "identity": identity,
            },
            "POST",
        )
    ]


def test_re_elected_follower_wait_is_bounded_and_cancelled() -> None:
    runtime = _reuse_runtime()
    action = _reuse_action()
    decisions = iter(
        [
            {
                "decision": "wait_and_sync_reused_result",
                "binding_id": "binding-1",
            },
            {
                "decision": "wait_and_sync_reused_result",
                "binding_id": "binding-2",
            },
        ]
    )
    cancelled: list[str] = []

    def request_json(path, payload, *, method):
        if path.endswith("/resolve"):
            return next(decisions)
        raise AssertionError(path)

    def wait_for_reuse(binding_id, identity, *, cancel_event):
        cancelled.append(binding_id)
        return {"decision": "execute_locally"}

    with (
        patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json),
        patch.object(FlowPilotRuntime, "_wait_for_reuse", side_effect=wait_for_reuse),
    ):
        observation = runtime.resolve_reuse(action, ReusableSearchObservation)

    assert observation is None
    assert cancelled == ["binding-1", "binding-2"]
    assert action.id not in runtime._waiting_reuses


def test_reuse_error_cancels_registered_follower_before_local_fallback() -> None:
    runtime = _reuse_runtime()
    action = _reuse_action()
    cancelled: list[str] = []

    with (
        patch.object(
            FlowPilotRuntime,
            "_request_json",
            return_value={
                "decision": "wait_and_sync_reused_result",
                "binding_id": "binding-1",
            },
        ),
        patch.object(
            FlowPilotRuntime, "_wait_for_reuse", side_effect=OSError("offline")
        ),
        patch.object(
            FlowPilotRuntime,
            "_cancel_follower",
            side_effect=lambda binding_id, _identity: cancelled.append(binding_id),
        ),
    ):
        observation = runtime.resolve_reuse(action, ReusableSearchObservation)

    assert observation is None
    assert cancelled == ["binding-1"]
    assert action.id not in runtime._waiting_reuses


def test_malformed_reuse_result_falls_back_to_local_execution() -> None:
    runtime = _reuse_runtime()
    response = {
        "decision": "sync_with_reused_result",
        "result": {"content": [], "is_error": False},
        "provenance": {"reuse_type": "historical"},
    }
    with patch.object(FlowPilotRuntime, "_request_json", return_value=response):
        observation = runtime.resolve_reuse(_reuse_action(), ReusableSearchObservation)
    assert observation is None


def test_leader_result_publication_is_best_effort() -> None:
    runtime = _reuse_runtime()
    action = _reuse_action()
    requests: list[tuple[str, dict | None, str]] = []

    def request_json(path, payload, *, method):
        requests.append((path, payload, method))
        if path.endswith("/resolve"):
            return {
                "decision": "sync_and_execute_as_leader",
                "binding_id": "binding-1",
            }
        raise OSError("control plane unavailable after local success")

    with patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json):
        assert runtime.resolve_reuse(action, ReusableSearchObservation) is None
        token = runtime.tool_start(action)
        event = ObservationEvent(
            observation=ReusableSearchObservation(
                items=[{"title": "FlowPilot"}], content=[]
            ),
            action_id=action.id,
            tool_name=action.tool_name,
            tool_call_id=action.tool_call_id,
        )
        with patch.object(FlowPilotRuntime, "_post_tool_event"):
            runtime.tool_terminal(
                action,
                token=token,
                started=time.monotonic(),
                events=[event],
            )
    assert requests[0][0].endswith("/resolve")
    assert requests[1][0].endswith("/bindings/binding-1/result")
    assert isinstance(event.observation, ReusableSearchObservation)
    assert event.observation.items[0]["title"] == "FlowPilot"


def test_phase2_adapter_grants_appends_continues_syncs_and_reconciles() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    requests: list[tuple[str, dict | None, str]] = []

    def request_json(path, payload, *, method):
        requests.append((path, payload, method))
        if path.endswith("/delegations"):
            return {"delta_digest": "a" * 64}
        if path.endswith("/deltas/append"):
            return {
                "delta_digest": "b" * 64,
                "last_seq": 2,
                "state": "open",
                "barrier_reason": None,
            }
        if path.endswith("/continuations"):
            return {
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "user", "content": "question"},
                        {"role": "tool", "tool_call_id": "tool-call-1"},
                    ],
                }
            }
        if path.endswith("/dcs/sync"):
            return {
                "first_seq": 1,
                "last_seq": 2,
                "delta_digest": "b" * 64,
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "tool-call-1"}]},
                    {"role": "tool", "tool_call_id": "tool-call-1"},
                ],
            }
        if path.endswith("/sync/ack"):
            return {"pending_message_count": 0, "state": "acked"}
        if path.endswith("/reconcile"):
            return {"status": "in_sync", "sync_required": False}
        raise AssertionError(path)

    with patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json):
        reference = runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "question"}],
                "stream": False,
            },
        )
        assert reference.delta_digest == "a" * 64
        updated = runtime.append_deferred_context(
            expected_last_seq=0,
            parent_llm_call_id=identity.llm_call_id,
            messages=[
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "tool-call-1"}],
                },
                {"role": "tool", "tool_call_id": "tool-call-1"},
            ],
            tool_call_ids=["tool-call-1"],
            resolution_receipts=["receipt-1"],
            result_digests=["c" * 64],
        )
        assert updated.delta_digest == "b" * 64
        body = runtime.prepare_internal_continuation(identity.llm_call_id)
        assert body["messages"][-1]["tool_call_id"] == "tool-call-1"
        applied: list[tuple[dict, ...]] = []
        runtime.synchronize_deferred_context(
            barrier_reason="terminal_response",
            apply_atomically=lambda messages: (
                applied.append(messages) or "cursor-3",
                "d" * 64,
            ),
        )
        assert [item["role"] for item in applied[0]] == ["assistant", "tool"]
        reconciled = runtime.reconcile_deferred_context(
            context_cursor="cursor-3", context_digest="b" * 64
        )
        assert reconciled["status"] == "in_sync"
    assert [path.rsplit("/", 1)[-1] for path, _, _ in requests] == [
        "delegations",
        "append",
        "continuations",
        "sync",
        "ack",
        "reconcile",
    ]
    append_payload = requests[1][1]
    assert append_payload is not None
    assert append_payload["resolution_receipts"] == ["receipt-1"]
    assert append_payload["result_digests"] == ["c" * 64]
    ack_payload = requests[4][1]
    assert ack_payload is not None
    assert ack_payload["new_context_digest"] == "d" * 64


def test_phase2_empty_delegation_is_released_before_agent_fallback() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    requests: list[str] = []

    def request_json(path, payload, *, method):
        requests.append(path)
        if path.endswith("/delegations"):
            return {"delta_digest": "a" * 64}
        if path.endswith("/delegations/release"):
            return {"state": "aborted", "pending_message_count": 0}
        raise AssertionError(path)

    with patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json):
        runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={"messages": [], "stream": False},
        )
        runtime.release_delegation()

    assert requests[-1].endswith("/delegations/release")
    assert runtime._dcs_reference is None


def test_phase2_sync_recovers_existing_server_barrier() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    requests: list[str] = []
    lease_ids: list[str] = []

    def request_json(path, payload, *, method):
        requests.append(path)
        if path.endswith("/delegations"):
            return {"delta_digest": "a" * 64}
        if path.endswith("/dcs/sync"):
            raise RuntimeError("barrier already active")
        if path.endswith("/dcs/reconcile"):
            return {
                "status": "sync_required",
                "sync_required": True,
                "state": "syncing",
                "lease_id": lease_ids[0],
                "base_context_cursor": "cursor-1",
                "delta_digest": "b" * 64,
                "last_seq": 2,
                "barrier_reason": "ttl",
            }
        if path.endswith("/sync/next"):
            return {
                "first_seq": 1,
                "last_seq": 2,
                "delta_digest": "b" * 64,
                "wal_delta_digest": "b" * 64,
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                    {"role": "tool", "tool_call_id": "call-1"},
                ],
            }
        if path.endswith("/sync/ack"):
            return {"state": "acked", "pending_message_count": 0}
        raise AssertionError(path)

    applied: list[tuple[dict, ...]] = []
    with patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json):
        reference = runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={"messages": [], "stream": False},
        )
        lease_ids.append(reference.lease_id)
        runtime.synchronize_deferred_context(
            barrier_reason="failure",
            apply_atomically=lambda messages: (
                applied.append(messages) or "cursor-2",
                "c" * 64,
            ),
        )

    assert [item["role"] for item in applied[0]] == ["assistant", "tool"]
    assert any(path.endswith("/sync/next") for path in requests)
    assert runtime._dcs_reference is None


def test_phase2_deferred_reuse_never_injects_an_observation() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            semantic_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
            reuse_poll_interval=0.001,
        ),
        "conversation-1",
    )
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="cursor-1",
        context_digest="a" * 64,
    )
    runtime._tool_identity = identity
    responses = iter(
        [
            {"delta_digest": "a" * 64},
            {"decision": "defer_wait_for_inflight", "binding_id": "binding-1"},
            {
                "decision": "defer_with_cached_result",
                "result": {"items": [{"title": "cached"}]},
            },
        ]
    )
    with patch.object(
        FlowPilotRuntime, "_request_json", side_effect=responses
    ) as request_json:
        runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={"messages": [], "stream": False},
        )
        decision = runtime.resolve_deferred_reuse(_reuse_action())
    assert decision["decision"] == "defer_with_cached_result"
    assert decision["result"]["items"][0]["title"] == "cached"
    assert request_json.call_args_list[1].args[1]["reuse"]["arguments"] == {
        "query": "flowpilot"
    }
    assert (
        request_json.call_args_list[1].args[1]["reuse"]["protocol_version"]
        == "flowpilot-phase1-reuse-v2"
    )


def test_agent_loop_defers_exact_hit_until_terminal_sync(tmp_path: Path) -> None:
    executions: list[str] = []
    conversation = _dcs_conversation(
        tmp_path,
        on_execute=lambda: executions.append("executed"),
    )
    control = _DCSControl(
        conversation,
        [
            {
                "decision": "defer_with_cached_result",
                "result": {"items": [{"title": "cached"}]},
                "provider_content": (
                    '{"items":[{"title":"cached"}]}\n'
                    "[FlowPilot reuse provenance: "
                    '{"observed_at":"2026-08-16T00:00:00Z",'
                    '"result_schema_version":"1",'
                    '"reuse_type":"historical"}]'
                ),
                "resolution_receipt": "receipt-1",
                "result_digest": "d" * 64,
            }
        ],
    )
    runtime = conversation._flowpilot_runtime
    assert runtime is not None

    def authoritative_tail() -> dict:
        identity = runtime.active_identity
        assert identity is not None
        return {
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "state": "NEXT_READY",
            "version": identity.tail_version + 1,
        }

    responses = iter(
        [_tool_response("call-cached", "flowpilot"), _content_response("done")]
    )
    with (
        patch.object(
            FlowPilotRuntime, "_request_json", side_effect=control.request_json
        ),
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            side_effect=authoritative_tail,
        ),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
        patch(
            "openhands.sdk.agent.agent.make_llm_completion",
            side_effect=lambda *args, **kwargs: next(responses),
        ),
    ):
        conversation.run()
        events = list(conversation.state.active_branch())
        cursor = events[-1].id
        digest = context_digest(events)
        reconciled = runtime.reconcile_deferred_context(
            context_cursor=cursor,
            context_digest=digest,
        )

    assert executions == []
    assert reconciled == {"status": "in_sync", "sync_required": False}
    assert control.sync_visible_event_types
    assert "ActionEvent" not in control.sync_visible_event_types[0]
    assert "ObservationEvent" not in control.sync_visible_event_types[0]
    authoritative = [
        event
        for event in conversation.state.active_branch()
        if isinstance(event, (ActionEvent, ObservationEvent))
        or (isinstance(event, MessageEvent) and event.source == "agent")
    ]
    assert [event.__class__.__name__ for event in authoritative] == [
        "ActionEvent",
        "ObservationEvent",
        "MessageEvent",
    ]
    appended_tool = next(
        message for message in control.pending_messages if message.get("role") == "tool"
    )
    provider_content = json.dumps(appended_tool, sort_keys=True)
    assert "cached" in provider_content
    assert "reuse_type" in provider_content
    assert "binding_id" not in provider_content
    assert control.ack_payloads[-1]["new_context_digest"] == context_digest(
        list(conversation.state.active_branch())
    )
    conversation.close()


def test_agent_loop_releases_empty_delegation_before_local_fallback(
    tmp_path: Path,
) -> None:
    executions: list[str] = []
    conversation = _dcs_conversation(
        tmp_path,
        on_execute=lambda: executions.append("executed"),
    )
    control = _DCSControl(conversation, [{"decision": "execute_locally"}])
    runtime = conversation._flowpilot_runtime
    assert runtime is not None

    def authoritative_tail() -> dict:
        identity = runtime.active_identity
        assert identity is not None
        return {
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "state": "NEXT_READY",
            "version": identity.tail_version + 1,
        }

    responses = iter(
        [_tool_response("call-local", "flowpilot"), _content_response("done")]
    )
    with (
        patch.object(
            FlowPilotRuntime, "_request_json", side_effect=control.request_json
        ),
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            side_effect=authoritative_tail,
        ),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
        patch(
            "openhands.sdk.agent.agent.make_llm_completion",
            side_effect=lambda *args, **kwargs: next(responses),
        ),
    ):
        conversation.run()

    assert control.released
    assert executions == ["executed"]
    conversation.close()


@pytest.mark.asyncio
async def test_async_agent_loop_defers_exact_hit_until_terminal_sync(
    tmp_path: Path,
) -> None:
    conversation = _dcs_conversation(tmp_path, on_execute=lambda: None)
    control = _DCSControl(
        conversation,
        [
            {
                "decision": "defer_with_cached_result",
                "result": {"items": [{"title": "async cached"}]},
                "provider_content": "async cached",
                "resolution_receipt": "receipt-async",
                "result_digest": "e" * 64,
            }
        ],
    )
    runtime = conversation._flowpilot_runtime
    assert runtime is not None

    def authoritative_tail() -> dict:
        identity = runtime.active_identity
        assert identity is not None
        return {
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "state": "NEXT_READY",
            "version": identity.tail_version + 1,
        }

    responses = iter(
        [_tool_response("call-async", "async"), _content_response("async done")]
    )

    async def completion(*args, **kwargs) -> LLMResponse:
        return next(responses)

    with (
        patch.object(
            FlowPilotRuntime, "_request_json", side_effect=control.request_json
        ),
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            side_effect=authoritative_tail,
        ),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
        patch(
            "openhands.sdk.agent.agent.amake_llm_completion",
            side_effect=completion,
        ),
    ):
        await conversation.arun()

    events = list(conversation.state.active_branch())
    assert [
        event.__class__.__name__
        for event in events
        if isinstance(event, (ActionEvent, ObservationEvent))
        or (isinstance(event, MessageEvent) and event.source == "agent")
    ] == ["ActionEvent", "ObservationEvent", "MessageEvent"]
    assert control.ack_payloads[-1]["new_context_digest"] == context_digest(events)
    conversation.close()


def test_agent_loop_acks_hidden_delta_before_local_tool(tmp_path: Path) -> None:
    execution_order: list[str] = []
    control_holder: list[_DCSControl] = []

    def execute() -> None:
        assert control_holder[0].acked
        execution_order.append("execute")

    conversation = _dcs_conversation(tmp_path, on_execute=execute)
    control = _DCSControl(
        conversation,
        [
            {
                "decision": "defer_with_cached_result",
                "result": {"items": [{"title": "cached"}]},
                "provider_content": "cached",
                "resolution_receipt": "receipt-1",
                "result_digest": "d" * 64,
            }
        ],
    )
    control_holder.append(control)
    runtime = conversation._flowpilot_runtime
    assert runtime is not None

    def authoritative_tail() -> dict:
        identity = runtime.active_identity
        assert identity is not None
        return {
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "state": "NEXT_READY",
            "version": identity.tail_version + 1,
        }

    responses = iter(
        [
            _tool_response("call-cached", "cached"),
            _local_tool_response("call-local", "local"),
            _content_response("finished after local execution"),
        ]
    )
    with (
        patch.object(
            FlowPilotRuntime, "_request_json", side_effect=control.request_json
        ),
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            side_effect=authoritative_tail,
        ),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
        patch(
            "openhands.sdk.agent.agent.make_llm_completion",
            side_effect=lambda *args, **kwargs: next(responses),
        ),
    ):
        conversation.run()

    assert execution_order == ["execute"]
    assert control.sync_payloads[-1]["barrier_reason"] == "local_tool"
    assert control.sync_payloads[-1]["pending_local_tool_call_ids"] == ["call-local"]
    assert (
        control.sync_payloads[-1]["barrier_messages"][0]["tool_calls"][0]["id"]
        == "call-local"
    )
    relevant = [
        event.__class__.__name__
        for event in conversation.state.active_branch()
        if isinstance(event, (ActionEvent, ObservationEvent))
        or (isinstance(event, MessageEvent) and event.source == "agent")
    ]
    assert relevant == [
        "ActionEvent",
        "ObservationEvent",
        "ActionEvent",
        "ObservationEvent",
        "MessageEvent",
    ]
    conversation.close()


@pytest.mark.asyncio
async def test_async_agent_loop_acks_hidden_delta_before_local_tool(
    tmp_path: Path,
) -> None:
    execution_order: list[str] = []
    control_holder: list[_DCSControl] = []

    def execute() -> None:
        assert control_holder[0].acked
        execution_order.append("execute")

    conversation = _dcs_conversation(tmp_path, on_execute=execute)
    control = _DCSControl(
        conversation,
        [
            {
                "decision": "defer_with_cached_result",
                "result": {"items": [{"title": "cached"}]},
                "provider_content": "cached",
                "resolution_receipt": "receipt-async-local",
                "result_digest": "f" * 64,
            }
        ],
    )
    control_holder.append(control)
    runtime = conversation._flowpilot_runtime
    assert runtime is not None

    def authoritative_tail() -> dict:
        identity = runtime.active_identity
        assert identity is not None
        return {
            "tail_request_id": identity.tail_request_id,
            "llm_call_id": identity.llm_call_id,
            "state": "NEXT_READY",
            "version": identity.tail_version + 1,
        }

    responses = iter(
        [
            _tool_response("call-cached", "cached"),
            _local_tool_response("call-local", "local"),
            _content_response("finished after async local execution"),
        ]
    )

    async def completion(*args, **kwargs) -> LLMResponse:
        return next(responses)

    with (
        patch.object(
            FlowPilotRuntime, "_request_json", side_effect=control.request_json
        ),
        patch.object(
            FlowPilotRuntime,
            "_fetch_authoritative_tail",
            side_effect=authoritative_tail,
        ),
        patch.object(FlowPilotRuntime, "_post_tool_event"),
        patch(
            "openhands.sdk.agent.agent.amake_llm_completion",
            side_effect=completion,
        ),
    ):
        await conversation.arun()

    assert execution_order == ["execute"]
    assert control.sync_payloads[-1]["barrier_reason"] == "local_tool"
    assert control.sync_payloads[-1]["pending_local_tool_call_ids"] == ["call-local"]
    conversation.close()


def test_dcs_recovery_manifest_is_restartable_and_metadata_only() -> None:
    store = InMemoryFileStore()
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    runtime.attach_file_store(store)
    identity = runtime.begin_request(
        context_sequence=3,
        base_context_cursor="head-2",
        context_digest="a" * 64,
    )
    with patch.object(
        FlowPilotRuntime,
        "_request_json",
        return_value={"delta_digest": "a" * 64},
    ):
        runtime._dcs_reference = runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={"messages": [{"role": "user", "content": "prompt"}]},
        )
    runtime._dcs_policy_version = 1
    runtime._dcs_base_context_sequence = 3
    runtime._dcs_base_context_digest = "a" * 64
    runtime._sync_reference = runtime._dcs_reference
    runtime._sync_response = {
        "first_seq": 1,
        "last_seq": 2,
        "delta_digest": "b" * 64,
    }
    event = MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text="done")]),
    )
    with patch.object(
        FlowPilotRuntime,
        "_request_json",
        return_value={"delta_digest": "a" * 64},
    ):
        runtime.persist_recovery_record(
            messages=(
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "secret result",
                },
            ),
            events=[event],
        )

    manifest = json.loads(store.read("flowpilot/recovery/pending.json"))
    payload = json.loads(store.read(manifest["payload_file"]))
    serialized = json.dumps({"manifest": manifest, "payload": payload})
    assert "prompt" not in serialized
    assert "secret result" not in serialized
    assert manifest["event_ids"] == [event.id]

    restarted = FlowPilotRuntime(runtime.config, "conversation-1")
    restarted.attach_file_store(store)
    applied: list[tuple[list, tuple[dict, ...]]] = []
    reference = runtime._dcs_reference
    assert reference is not None

    def request_json(path, payload, *, method):
        if path.endswith("/dcs/reconcile"):
            return {
                "status": "sync_required",
                "sync_required": True,
                "state": "syncing",
                "lease_id": reference.lease_id,
                "base_context_cursor": "head-2",
                "delta_digest": "b" * 64,
                "last_seq": 2,
            }
        if path.endswith("/sync/ack"):
            return {"pending_message_count": 0, "state": "acked"}
        raise AssertionError(path)

    failed_restart = FlowPilotRuntime(runtime.config, "conversation-1")
    failed_restart.attach_file_store(store)

    def fail_ack(path, payload, *, method):
        if path.endswith("/dcs/reconcile"):
            return request_json(path, payload, method=method)
        if path.endswith("/sync/ack"):
            raise OSError("crash before ACK")
        raise AssertionError(path)

    with (
        patch.object(FlowPilotRuntime, "_post_control"),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
        patch.object(FlowPilotRuntime, "_request_json", side_effect=fail_ack),
        pytest.raises(OSError, match="crash before ACK"),
    ):
        failed_restart.register(
            context_sequence=3,
            base_context_cursor="head-2",
            context_digest="a" * 64,
            apply_recovery=lambda events, messages: (event.id, "c" * 64),
        )
    assert failed_restart._read_recovery_manifest() is not None

    with (
        patch.object(FlowPilotRuntime, "_post_control"),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
        patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json),
    ):
        restarted.register(
            context_sequence=3,
            base_context_cursor="head-2",
            context_digest="a" * 64,
            apply_recovery=lambda events, messages: (
                applied.append((events, messages)) or (event.id, "c" * 64)
            ),
        )
    assert len(applied) == 1
    assert applied[0][0][0].id == event.id
    assert restarted._read_recovery_manifest() is None


def test_legacy_pending_wal_without_recovery_record_fails_closed() -> None:
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    runtime.attach_file_store(InMemoryFileStore())
    with (
        patch.object(FlowPilotRuntime, "_post_control"),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
        patch.object(
            FlowPilotRuntime,
            "_request_json",
            return_value={"status": "sync_required", "sync_required": True},
        ),
        pytest.raises(RuntimeError, match="requires recovery metadata"),
    ):
        runtime.register(
            context_sequence=1,
            base_context_cursor="head-1",
            context_digest="a" * 64,
        )


def test_dcs_recovery_resumes_after_first_chunk_ack() -> None:
    store = InMemoryFileStore()
    runtime = FlowPilotRuntime(
        _config(
            exact_reuse_enabled=True,
            reusable_web_tools=("web_search",),
            deferred_context_enabled=True,
        ),
        "conversation-1",
    )
    runtime.attach_file_store(store)
    identity = runtime.begin_request(
        context_sequence=1,
        base_context_cursor="head-0",
        context_digest="a" * 64,
    )
    with patch.object(
        FlowPilotRuntime,
        "_request_json",
        return_value={"delta_digest": "d" * 64},
    ):
        runtime._dcs_reference = runtime.grant_delegation(
            identity,
            api_kind="chat",
            request_snapshot={"messages": []},
        )
    runtime._sync_reference = runtime._dcs_reference
    runtime._sync_response = {
        "first_seq": 1,
        "last_seq": 1,
        "delta_digest": "1" * 64,
    }
    first_event = MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text="first")]),
    )
    second_event = MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text="second")]),
    )
    first_messages = ({"role": "assistant", "content": "first"},)
    second_messages = ({"role": "assistant", "content": "second"},)
    runtime.persist_recovery_record(
        messages=first_messages,
        events=[first_event],
        pending_batches=(
            ([first_event], first_messages),
            ([second_event], second_messages),
        ),
    )

    restarted = FlowPilotRuntime(runtime.config, "conversation-1")
    restarted.attach_file_store(store)
    applied: list[str] = []
    paths: list[str] = []
    reference = runtime._dcs_reference
    assert reference is not None

    def request_json(path, payload, *, method):
        paths.append(path)
        if path.endswith("/dcs/reconcile"):
            return {
                "status": "sync_required",
                "sync_required": True,
                "state": "syncing",
                "lease_id": reference.lease_id,
                "base_context_cursor": first_event.id,
                "base_context_digest": "c" * 64,
                "delta_digest": "2" * 64,
                "pending_message_count": 1,
            }
        if path.endswith("/sync/next"):
            return {
                "first_seq": 2,
                "last_seq": 2,
                "delta_digest": "2" * 64,
                "wal_delta_digest": "2" * 64,
                "messages": list(second_messages),
            }
        if path.endswith("/sync/ack"):
            assert payload["first_seq"] == 2
            return {"pending_message_count": 0, "state": "acked"}
        raise AssertionError(path)

    with (
        patch.object(FlowPilotRuntime, "_post_control"),
        patch.object(FlowPilotRuntime, "_find_authoritative_tail", return_value=None),
        patch.object(FlowPilotRuntime, "_request_json", side_effect=request_json),
    ):
        restarted.register(
            context_sequence=1,
            base_context_cursor="head-0",
            context_digest="a" * 64,
            apply_recovery=lambda events, _messages: (
                applied.extend(event.id for event in events)
                or (second_event.id, "e" * 64)
            ),
        )

    assert applied == [second_event.id]
    assert paths == [
        "/flowpilot/v1/dcs/reconcile",
        "/flowpilot/v1/dcs/sync/next",
        "/flowpilot/v1/dcs/sync/ack",
    ]
    assert restarted._read_recovery_manifest() is None


def test_event_batch_failure_leaves_head_unchanged_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    conversation = LocalConversation(
        agent=Agent(
            llm=LLM(model="gpt-4o"),
            tools=[],
            include_default_tools=[],
            tool_concurrency_limit=1,
        ),
        workspace=LocalWorkspace(working_dir=tmp_path / "workspace"),
        persistence_dir=tmp_path / "conversation",
        visualizer=None,
    )
    conversation.send_message("base")
    state = conversation.state
    old_ids = [event.id for event in state.active_branch()]
    events: list[Event] = [
        MessageEvent(
            source="agent",
            llm_message=Message(role="assistant", content=[TextContent(text="one")]),
        ),
        MessageEvent(
            source="agent",
            llm_message=Message(role="assistant", content=[TextContent(text="two")]),
        ),
    ]
    original_write = state._events._fs.write
    writes = 0

    def fail_second_event(path, contents):
        nonlocal writes
        if path.startswith("events/"):
            writes += 1
            if writes == 2:
                raise OSError("crash during event prepare")
        original_write(path, contents)

    with patch.object(state._events._fs, "write", side_effect=fail_second_event):
        with pytest.raises(OSError, match="crash during event prepare"):
            state.append_event_batch(events)
    assert [event.id for event in state.active_branch()] == old_ids

    state.append_event_batch(events)
    active = state.active_branch()
    assert [event.id for event in active[-2:]] == [event.id for event in events]
    assert len({event.id for event in active}) == len(active)

    next_event: list[Event] = [
        MessageEvent(
            source="agent",
            llm_message=Message(
                role="assistant", content=[TextContent(text="after head failure")]
            ),
        )
    ]
    committed_ids = [event.id for event in active]
    original_save = state._save_base_state
    with (
        patch.object(
            state,
            "_save_base_state",
            side_effect=OSError("crash while committing HEAD"),
        ),
        pytest.raises(OSError, match="crash while committing HEAD"),
    ):
        state.append_event_batch(next_event)
    assert [event.id for event in state.active_branch()] == committed_ids
    original_save(state._fs)
    state.append_event_batch(next_event)
    assert state.active_branch()[-1].id == next_event[0].id
    conversation.close()
