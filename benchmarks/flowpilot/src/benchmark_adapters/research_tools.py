"""OpenHands tools for SQL, paper retrieval, and BFCL web research."""

import json
import time
from typing import Any

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolAnnotations, ToolExecutor, register_tool
from openhands.sdk.tool.builtins.finish import FinishAction, FinishObservation
from pydantic import Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from .sdk_bridge import get_binding


class SchemaAction(Action):
    db_id: str = Field(min_length=1)


class SQLAction(SchemaAction):
    sql: str = Field(min_length=1)
    max_rows: int = Field(default=50, ge=1, le=200)


class PaperAction(Action):
    paper_id: str = Field(min_length=1)


class PaperSearchAction(PaperAction):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ParagraphAction(Action):
    paper_id: str = Field(min_length=1)
    paragraph_id: int = Field(ge=0)


class WebSearchAction(Action):
    keywords: str = Field(min_length=1)
    max_results: int = Field(default=10, ge=1, le=20)
    region: str = "wt-wt"


class URLAction(Action):
    url: str = Field(min_length=1)
    mode: str = Field(default="raw", pattern="^(raw|markdown|truncate)$")


ACTIONS = {
    "bird": {"get_schema": SchemaAction, "execute_sql": SQLAction},
    "qasper": {"search_paper": PaperSearchAction, "read_paragraph": ParagraphAction},
    "bfcl": {"search_engine_query": WebSearchAction, "fetch_url_content": URLAction},
}


def research_actions(config):
    actions = ACTIONS[config.dataset.kind]
    if config.dataset.kind == "qasper" and config.retrieval.qasper_context_mode == "document":
        return {"read_paper": PaperAction, **actions}
    return actions


class ResearchObservation(Observation):
    pass


class ResearchExecutor(ToolExecutor):
    def __init__(self, binding: Any, name: str):
        self.binding, self.name = binding, name

    def __call__(self, action: Action, conversation=None):
        b = self.binding
        arguments = action.model_dump(mode="json", exclude={"summary", "security_risk", "kind"})
        b.budget.tool()
        link = b.recorder.start_tool(self.name, arguments)
        start = time.monotonic()
        try:
            with (
                b.recorder.activity("tool"),
                b.environment.operation_timeout(min(b.tool_timeout, b.budget.remaining())),
            ):
                data = b.environment.call_tool(self.name, arguments)
            text = json.dumps(data, ensure_ascii=False)
            timing = getattr(b.environment, "last_timing", {})
            b.recorder.emit(
                "tool_end",
                tool_name=self.name,
                **link,
                effective_arguments=arguments,
                model_observation=text,
                executor_duration_ms=timing.get(
                    "executor_duration_ms", (time.monotonic() - start) * 1000
                ),
                round_trip_ms=(time.monotonic() - start) * 1000,
                executor_clock_domain=timing.get("executor_clock_domain", "host-executor"),
                executor_queue_wait_ms=timing.get("queue_wait_ms"),
                output_bytes=len(text.encode()),
                timed_out=False,
            )
            return ResearchObservation.from_text(text)
        except Exception as exc:
            b.recorder.emit(
                "tool_error",
                tool_name=self.name,
                **link,
                effective_arguments=arguments,
                model_observation=str(exc),
                executor_duration_ms=(time.monotonic() - start) * 1000,
                round_trip_ms=(time.monotonic() - start) * 1000,
                executor_clock_domain="host-executor",
                error_type=type(exc).__name__,
                timed_out=isinstance(exc, TimeoutError),
            )
            return ResearchObservation.from_text(str(exc), is_error=True)
        finally:
            for attempt in b.environment.last_http_attempts:
                b.recorder.emit("network_attempt", tool_name=self.name, **link, **attempt)


DESCRIPTIONS = {
    "read_paper": "Read this paper as a shared document with paragraph IDs. Call it once before answering in document mode; if truncated, retrieve later paragraphs with search_paper/read_paragraph.",
    "get_schema": "Read the SQLite table/view definitions for the explicit database ID.",
    "execute_sql": "Execute one read-only SQLite query in the explicit database. Results are capped; truncated indicates additional rows.",
    "search_paper": "Search paragraphs within the explicit paper ID. Returns complete paragraphs with stable integer paragraph_id, not answers.",
    "read_paragraph": "Read one complete paragraph from the explicit paper ID and paragraph ID.",
    "search_engine_query": "Search the configured live search engine for keywords. Returns URLs, titles and snippets. Wikipedia backend searches English Wikipedia only; use concise entity names or keywords.",
    "fetch_url_content": "Retrieve the supplied HTTP(S) URL. raw returns HTML, markdown returns Markdown, truncate returns cleaned text. HTML is cleaned while retaining article text and tables in truncate mode.",
}


class ResearchTool(ToolDefinition):
    @classmethod
    def create(cls, conv_state=None, **params):
        b = get_binding(params["binding_key"])
        name = params["research_name"]
        return [
            cls(
                action_type=research_actions(b.environment.config)[name],
                observation_type=ResearchObservation,
                description=DESCRIPTIONS[name],
                executor=ResearchExecutor(b, name),
                annotations=ToolAnnotations(readOnlyHint=True),
            )
        ]


class SQLFinishAction(FinishAction):
    message: SkipJsonSchema[str] = ""
    sql: str = Field(min_length=1, description="The final complete SQLite query")


class AnswerFinishAction(FinishAction):
    message: SkipJsonSchema[str] = ""
    answer: str = Field(min_length=1, description="Only the requested final answer")


class PaperFinishAction(AnswerFinishAction):
    evidence_ids: list[int] = Field(
        min_length=1, description="Integer paragraph IDs from this paper"
    )

    @field_validator("answer")
    @classmethod
    def short_answer(cls, value):
        if len(value.split()) > 32:
            raise ValueError("Answer must contain at most 32 whitespace-separated words")
        return value


class ResearchFinishExecutor(ToolExecutor):
    def __init__(self, binding):
        self.binding = binding

    def __call__(self, action, conversation=None):
        data = action.model_dump(mode="json", exclude={"kind", "message"})
        self.binding.environment.finished_data = data
        self.binding.recorder.final_text = json.dumps(data, ensure_ascii=False)
        return FinishObservation.from_text(self.binding.recorder.final_text)


class ResearchFinishTool(ToolDefinition):
    name = "finish"

    @classmethod
    def create(cls, conv_state=None, **params):
        binding = get_binding(params["binding_key"])
        kind = binding.environment.config.dataset.kind
        return [
            cls(
                action_type={
                    "bird": SQLFinishAction,
                    "qasper": PaperFinishAction,
                    "bfcl": AnswerFinishAction,
                }[kind],
                observation_type=FinishObservation,
                description="Finish the task with the required structured answer fields. Do not summarize your work in place of the answer.",
                executor=ResearchFinishExecutor(binding),
                annotations=ToolAnnotations(readOnlyHint=True),
            )
        ]


register_tool("research_finish", ResearchFinishTool)


def make_tools(binding_key):
    b = get_binding(binding_key)
    # One registered SDK ToolDefinition subclass per name preserves SDK identities.
    return [
        Tool(name=name, params={"binding_key": binding_key, "research_name": name})
        for name in research_actions(b.environment.config)
    ] + [Tool(name="research_finish", params={"binding_key": binding_key})]


for _kind in ACTIONS.values():
    for _name in _kind:
        _tool = type(_name, (ResearchTool,), {"__module__": __name__})
        register_tool(_name, _tool)

_read_paper_tool = type("read_paper", (ResearchTool,), {"__module__": __name__})
register_tool("read_paper", _read_paper_tool)
