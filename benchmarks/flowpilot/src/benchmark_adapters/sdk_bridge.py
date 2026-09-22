import time
import uuid
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Literal

from openhands.sdk import LLM, Action, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from pydantic import Field, PrivateAttr

from .environment import file_operation_command
from .tracing import Budget, TraceRecorder


class RecordedLLM(LLM):
    _recorder: TraceRecorder | None = PrivateAttr(default=None)
    _budget: Budget | None = PrivateAttr(default=None)
    _logical_id: str = PrivateAttr(default="")
    _request_id: str = PrivateAttr(default="")

    def attach(self, recorder, budget):
        self._recorder, self._budget = recorder, budget

    def completion(self, *args, **kwargs):
        self._logical_id = uuid.uuid4().hex
        return super().completion(*args, **kwargs)

    def _prepare_transport_kwargs(self, **kwargs):
        if kwargs.get("enable_streaming"):
            raise ValueError(
                "Adapter tracing currently supports non-streaming Chat Completions only"
            )
        assert self._recorder is not None and self._budget is not None
        self._budget.request()
        payload = super()._prepare_transport_kwargs(**kwargs)
        payload["timeout"] = min(self.timeout or 180, self._budget.remaining())
        # Retries below this tracing boundary would be invisible physical requests.
        payload["max_retries"] = 0
        payload["num_retries"] = 0
        self._request_id = uuid.uuid4().hex
        self._recorder.request(
            self._request_id,
            payload,
            logical_request_id=self._logical_id,
            request_role="actor",
            snapshot_stage="litellm_transport_input",
            budget_at_t0=self._budget.snapshot(),
        )
        return payload

    def _transport_call(self, **kwargs):
        start = time.monotonic_ns()
        self._request_id = ""
        try:
            assert self._recorder is not None
            with self._recorder.activity("llm"):
                response = super()._transport_call(**kwargs)
            self._recorder.response_requests[str(response.id)] = self._request_id
            self._recorder.emit(
                "llm_response",
                request_id=self._request_id,
                logical_request_id=self._logical_id,
                llm_response_id=str(response.id),
                duration_ms=(time.monotonic_ns() - start) / 1e6,
                response=self._recorder.blob(response.model_dump(mode="json")),
            )
            self._recorder.response_decision(self._request_id, response.model_dump(mode="json"))
            return response
        except Exception as exc:
            if self._recorder is not None:
                self._recorder.emit(
                    "llm_error" if self._request_id else "llm_request_not_sent",
                    request_id=self._request_id or None,
                    logical_request_id=self._logical_id,
                    error_type=type(exc).__name__,
                )
            raise


@dataclass
class Binding:
    environment: Any
    recorder: TraceRecorder
    budget: Budget
    tool_timeout: int
    max_output_chars: int


_BINDINGS: dict[str, Binding] = {}
_LOCK = RLock()


def bind(binding):
    key = uuid.uuid4().hex
    with _LOCK:
        _BINDINGS[key] = binding
    return key


def unbind(key):
    with _LOCK:
        _BINDINGS.pop(key, None)


def get_binding(key):
    with _LOCK:
        return _BINDINGS[key]


class ShellAction(Action):
    command: str = Field(min_length=1, description="Bash command executed in the task repository.")
    timeout: int = Field(
        default=120, ge=1, le=1800, description="Requested timeout, capped by the task profile."
    )


class FileAction(Action):
    command: Literal["view", "create", "str_replace"]
    path: str = Field(
        min_length=1,
        description="Repository-relative path or absolute path inside the task repository.",
    )
    file_text: str = ""
    old_str: str = ""
    new_str: str = ""
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=2000)


class ContainerObservation(Observation):
    pass


class ContainerExecutor(ToolExecutor):
    def __init__(self, binding, tool_name):
        self.binding, self.tool_name = binding, tool_name

    def __call__(self, action, conversation=None):
        b = self.binding
        try:
            b.budget.tool()
        except Exception:
            b.recorder.emit(
                "tool_not_executed",
                tool_name=self.tool_name,
                reason=b.budget.reason,
                arguments=action.model_dump(mode="json"),
            )
            raise
        arguments = action.model_dump(mode="json")
        timeout = min(b.tool_timeout, b.budget.remaining())
        if self.tool_name in {"swe_terminal", "code_terminal"}:
            command = action.command
            timeout = min(timeout, action.timeout)
        else:
            command = file_operation_command(b.environment.repo_dir, arguments)
        link = b.recorder.start_tool(
            self.tool_name,
            arguments,
            effective_arguments={
                **arguments,
                "timeout": timeout,
                "max_output_chars": b.max_output_chars,
            },
        )
        start = time.monotonic_ns()
        try:
            with b.recorder.activity("tool"):
                result = b.environment.execute(
                    command, timeout=timeout, max_bytes=b.max_output_chars
                )
            observed = (
                result.stdout
                + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
                + f"\n[exit_code={result.exit_code}, timed_out={result.timed_out}, truncated={result.truncated}]"
            )
            b.recorder.emit(
                "tool_end",
                tool_name=self.tool_name,
                **link,
                executor_duration_ms=result.duration_ms,
                round_trip_ms=(time.monotonic_ns() - start) / 1e6,
                outcome=asdict(result),
                model_observation=observed,
            )
            return ContainerObservation.from_text(
                observed, is_error=result.exit_code != 0 or result.timed_out
            )
        except Exception as exc:
            b.recorder.emit(
                "tool_error",
                tool_name=self.tool_name,
                **link,
                error_type=type(exc).__name__,
                round_trip_ms=(time.monotonic_ns() - start) / 1e6,
            )
            raise


class SWETerminalTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=ShellAction,
                observation_type=ContainerObservation,
                description=(
                    "Run Bash in the task repository with its configured Python environment. "
                    "Each call starts a new shell: cd and environment changes do not persist. "
                    "Files persist. Use foreground commands; hard timeouts kill the command process group. "
                    f"Repository: {b.environment.repo_dir}."
                ),
                executor=ContainerExecutor(b, cls.name),
            )
        ]


class SWEFileEditorTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=FileAction,
                observation_type=ContainerObservation,
                description=(
                    "View numbered lines, create a NEW text file with file_text, or replace exactly "
                    "one occurrence of old_str with new_str in the task repository. "
                    "Paths outside the repository and .git metadata are rejected."
                ),
                executor=ContainerExecutor(b, cls.name),
            )
        ]


register_tool(SWETerminalTool.name, SWETerminalTool)
register_tool(SWEFileEditorTool.name, SWEFileEditorTool)


def swe_tools(binding_key):
    return [
        Tool(name=cls.name, params={"binding_key": binding_key})
        for cls in [SWETerminalTool, SWEFileEditorTool]
    ]


class CodeTerminalTool(SWETerminalTool):
    pass


class CodeFileEditorTool(SWEFileEditorTool):
    pass


register_tool(CodeTerminalTool.name, CodeTerminalTool)
register_tool(CodeFileEditorTool.name, CodeFileEditorTool)


def code_tools(binding_key):
    return [
        Tool(name=cls.name, params={"binding_key": binding_key})
        for cls in [CodeTerminalTool, CodeFileEditorTool]
    ]
