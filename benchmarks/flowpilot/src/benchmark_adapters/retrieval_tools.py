import json
import time

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from pydantic import Field

from .sdk_bridge import get_binding


class SearchAction(Action):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class NativeSearchAction(Action):
    query: str


class NativeGetDocumentAction(Action):
    docid: str


class HotpotReadAction(Action):
    doc_id: str
    start_sentence: int = Field(default=0, ge=0)
    max_sentences: int = Field(default=20, ge=1, le=100)


class BrowseReadAction(Action):
    docid: str
    offset: int = Field(default=0, ge=0)


class RetrievalObservation(Observation):
    pass


class RetrievalExecutor(ToolExecutor):
    def __init__(self, binding, name):
        self.binding, self.name = binding, name

    def __call__(self, action, conversation=None):
        b = self.binding
        native = b.environment.config.retrieval.backend == "browsecomp_mcp"
        arguments = action.model_dump(mode="json")
        if native:
            arguments = (
                {"query": action.query} if self.name == "search" else {"docid": action.docid}
            )
        try:
            b.budget.tool()
        except Exception:
            b.recorder.emit(
                "tool_not_executed",
                tool_name=self.name,
                reason=b.budget.reason,
                arguments=arguments,
            )
            raise
        link = b.recorder.start_tool(self.name, arguments)
        start = time.monotonic_ns()
        response = None
        try:
            with (
                b.recorder.activity("tool"),
                b.environment.operation_timeout(min(b.tool_timeout, b.budget.remaining())),
            ):
                if native:
                    response = b.environment.call_tool(self.name, arguments)
                    data = response.data
                    if self.name == "search" and isinstance(data, list):
                        b.recorder.retrieved_docids.update(
                            str(row["docid"])
                            for row in data
                            if isinstance(row, dict) and "docid" in row
                        )
                    elif self.name == "get_document" and isinstance(data, dict) and "docid" in data:
                        b.recorder.retrieved_docids.add(str(data["docid"]))
                elif self.name == "search":
                    data = b.environment.search(action.query, action.top_k)
                    b.recorder.retrieved_docids.update(r["docid"] for r in data)
                elif self.name == "read_document":
                    data = b.environment.read(
                        action.doc_id,
                        start_sentence=action.start_sentence,
                        max_sentences=action.max_sentences,
                    )
                    b.recorder.retrieved_docids.add(action.doc_id)
                else:
                    data = b.environment.read(action.docid, offset=action.offset)
                    b.recorder.retrieved_docids.add(action.docid)
            executor_duration_ms = None if native else (time.monotonic_ns() - start) / 1e6
            if native:
                assert response is not None
                text = response.text
            else:
                text = json.dumps(data, ensure_ascii=False)
            observation = RetrievalObservation.from_text(text)
            b.recorder.emit(
                "tool_end",
                tool_name=self.name,
                **link,
                executor_duration_ms=executor_duration_ms,
                round_trip_ms=(time.monotonic_ns() - start) / 1e6,
                effective_arguments=arguments,
                queue_wait_ms=None,
                executor_clock_domain=None if native else "host-executor",
                model_observation=text,
                output_bytes=len(text.encode("utf-8")),
                timed_out=False,
            )
            return observation
        except Exception as exc:
            text = str(exc)
            observation = RetrievalObservation.from_text(text, is_error=True)
            b.recorder.emit(
                "tool_error",
                tool_name=self.name,
                **link,
                executor_duration_ms=None if native else (time.monotonic_ns() - start) / 1e6,
                round_trip_ms=(time.monotonic_ns() - start) / 1e6,
                effective_arguments=arguments,
                queue_wait_ms=None,
                executor_clock_domain=None if native else "host-executor",
                model_observation=text,
                error_type=type(exc).__name__,
                timed_out=isinstance(exc, TimeoutError),
            )
            return observation


class SearchTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=NativeSearchAction
                if b.environment.config.retrieval.backend == "browsecomp_mcp"
                else SearchAction,
                observation_type=RetrievalObservation,
                description=(
                    b.environment.tool_definitions["search"].description
                    if b.environment.config.retrieval.backend == "browsecomp_mcp"
                    else f"Search the fixed corpus. Return docid, title, URL and snippet. top_k must be <= {b.environment.config.retrieval.top_k}."
                ),
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


class ReadDocumentTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=HotpotReadAction,
                observation_type=RetrievalObservation,
                description="Read original Wikipedia sentences by doc_id. Returns official title and original zero-based sentence IDs. Use these IDs in supporting_facts.",
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


class GetDocumentTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        return [
            cls(
                action_type=NativeGetDocumentAction
                if b.environment.config.retrieval.backend == "browsecomp_mcp"
                else BrowseReadAction,
                observation_type=RetrievalObservation,
                description=(
                    b.environment.tool_definitions["get_document"].description
                    if b.environment.config.retrieval.backend == "browsecomp_mcp"
                    else "Read a fixed-corpus document page by docid. offset is a character offset. Follow next_offset when truncated is true."
                ),
                executor=RetrievalExecutor(b, cls.name),
            )
        ]


for tool in [SearchTool, ReadDocumentTool, GetDocumentTool]:
    register_tool(tool.name, tool)


def make_tools(binding_key):
    b = get_binding(binding_key)
    read = ReadDocumentTool if b.environment.config.dataset.kind == "hotpot" else GetDocumentTool
    return [
        Tool(name=tool.name, params={"binding_key": binding_key}) for tool in [SearchTool, read]
    ]
