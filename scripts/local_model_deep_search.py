#!/usr/bin/env python3
"""Run a deep-search style task against a local OpenAI-compatible model.

The script explicitly loads `.openhands/hooks.json`, so OpenHands pre/post tool
hooks record the agent's tool calls while it searches, reads, and writes files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.sdk.hooks import HookConfig
from openhands.sdk.llm.llm import LLMCallContext
from openhands.sdk.llm.llm_response import LLMResponse
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "deep_search_runs"
# Keep this aligned with the current pre/post hook implementations.
DEFAULT_TOOL_RECORDS = Path(
    "/home/liyachen/workspace/experiments/traces/tool-records/tool_calls.jsonl"
)
DEFAULT_LLM_TRACE_FILE = REPO_ROOT / "traces" / "llm-records" / "llm_calls.jsonl"
DEFAULT_COMPLETION_LOG_DIR = REPO_ROOT / "traces" / "completion_logs"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class TracingLLM(LLM):
    """LLM wrapper that records per-request latency and TTFT to JSONL."""

    trace_file: str
    include_payloads: bool = True

    def _record_llm_call(
        self,
        *,
        api_path: str,
        call_id: str,
        started_at: str,
        started_perf_ns: int,
        first_token_perf_ns: int | None,
        messages: list[Any],
        tools: Sequence[Any] | None,
        response: LLMResponse | None,
        error: BaseException | None,
        call_context: LLMCallContext | None,
    ) -> None:
        ended_perf_ns = time.perf_counter_ns()
        ended_at = _now_iso()
        latency_ms = round((ended_perf_ns - started_perf_ns) / 1_000_000, 3)
        ttft_ms = (
            round((first_token_perf_ns - started_perf_ns) / 1_000_000, 3)
            if first_token_perf_ns is not None
            else None
        )

        raw_response = response.raw_response if response is not None else None
        usage = getattr(raw_response, "usage", None)
        record: dict[str, Any] = {
            "schema_version": 1,
            "llm_call_id": call_id,
            "session_id": call_context.session_id if call_context else None,
            "api_path": api_path,
            "usage_id": self.usage_id,
            "model": self.model,
            "base_url": self.base_url,
            "started_at": started_at,
            "ended_at": ended_at,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "stream": self.stream,
            "response_id": response.id if response is not None else None,
            "usage": _jsonable(usage) if usage is not None else None,
            "metrics_snapshot": _jsonable(response.metrics) if response else None,
            "error": None
            if error is None
            else {"type": type(error).__name__, "message": str(error)},
        }
        if self.include_payloads:
            record["messages"] = _jsonable(messages)
            record["tools"] = _jsonable(tools or [])
            record["output_message"] = (
                _jsonable(response.message) if response is not None else None
            )

        trace_path = Path(self.trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _wrap_token_callback(self, on_token: Any, state: dict[str, Any]):
        def wrapped(chunk: Any) -> None:
            if state["first_token_perf_ns"] is None:
                state["first_token_perf_ns"] = time.perf_counter_ns()
            if on_token is not None:
                on_token(chunk)

        return wrapped

    def completion(
        self,
        messages: list[Any],
        tools: Sequence[Any] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: Any = None,
        call_context: LLMCallContext | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_id = f"llm-{uuid.uuid4().hex}"
        state: dict[str, Any] = {"first_token_perf_ns": None}
        started_at = _now_iso()
        started_perf_ns = time.perf_counter_ns()
        response = None
        error = None
        try:
            response = super().completion(
                messages=messages,
                tools=tools,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=self._wrap_token_callback(on_token, state)
                if (kwargs.get("stream") or self.stream)
                else on_token,
                call_context=call_context,
                **kwargs,
            )
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._record_llm_call(
                api_path="chat.completions",
                call_id=call_id,
                started_at=started_at,
                started_perf_ns=started_perf_ns,
                first_token_perf_ns=state["first_token_perf_ns"],
                messages=messages,
                tools=tools,
                response=response,
                error=error,
                call_context=call_context,
            )

    def responses(
        self,
        messages: list[Any],
        tools: Sequence[Any] | None = None,
        include: list[str] | None = None,
        store: bool | None = None,
        add_security_risk_prediction: bool = False,
        on_token: Any = None,
        call_context: LLMCallContext | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        call_id = f"llm-{uuid.uuid4().hex}"
        state: dict[str, Any] = {"first_token_perf_ns": None}
        started_at = _now_iso()
        started_perf_ns = time.perf_counter_ns()
        response = None
        error = None
        try:
            response = super().responses(
                messages=messages,
                tools=tools,
                include=include,
                store=store,
                add_security_risk_prediction=add_security_risk_prediction,
                on_token=self._wrap_token_callback(on_token, state)
                if (kwargs.get("stream") or self.stream)
                else on_token,
                call_context=call_context,
                **kwargs,
            )
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._record_llm_call(
                api_path="responses",
                call_id=call_id,
                started_at=started_at,
                started_perf_ns=started_perf_ns,
                first_token_perf_ns=state["first_token_perf_ns"],
                messages=messages,
                tools=tools,
                response=response,
                error=error,
                call_context=call_context,
            )


def normalize_model_for_base_url(model: str, base_url: str | None) -> str:
    """Use LiteLLM's OpenAI provider for bare names on custom OpenAI endpoints."""
    if not base_url or "/" in model:
        return model
    return f"openai/{model}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask OpenHands to run a deep-search task with a local model."
    )
    parser.add_argument(
        "query",
        help="Research question, e.g. 'Compare current open-source agent benchmarks'.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for the research report and source notes. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--min-sources",
        type=int,
        default=5,
        help="Minimum number of independent sources the agent should inspect.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "local-model"),
        help="Model name passed to the local OpenAI-compatible server.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", "dummy"),
        help="API key. Local servers commonly accept any non-empty value.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096")),
        help=(
            "Maximum output tokens per LLM call. "
            "Qwen reasoning models need enough room."
        ),
    )
    web_group = parser.add_mutually_exclusive_group()
    web_group.add_argument(
        "--browser",
        action="store_true",
        help="Enable BrowserToolSet. Requires Chromium or Playwright Chromium.",
    )
    web_group.add_argument(
        "--tavily",
        action="store_true",
        help=(
            "Enable the Tavily MCP server for web search/extraction. Requires "
            "TAVILY_API_KEY. This cannot be combined with --browser."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming so llm_calls.jsonl can include TTFT.",
    )
    parser.add_argument(
        "--llm-trace-file",
        default=str(DEFAULT_LLM_TRACE_FILE),
        help=(
            f"JSONL file for per-request LLM traces. Default: {DEFAULT_LLM_TRACE_FILE}"
        ),
    )
    parser.add_argument(
        "--completion-log-dir",
        default=str(DEFAULT_COMPLETION_LOG_DIR),
        help=(
            "Directory for OpenHands raw completion logs containing formatted "
            f"request/response payloads. Default: {DEFAULT_COMPLETION_LOG_DIR}"
        ),
    )
    parser.add_argument(
        "--no-llm-payloads",
        action="store_true",
        help="Do not duplicate full messages/tools/output in llm_calls.jsonl.",
    )
    return parser.parse_args()


def build_agent(llm: LLM, *, enable_browser: bool, enable_tavily: bool) -> Agent:
    tools = [
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
    # Keep Tavily runs focused on web-tool events. Terminal remains available
    # for the legacy/browser mode, where it is part of the original experiment.
    if not enable_tavily:
        tools.insert(0, Tool(name=TerminalTool.name))

    if enable_browser:
        from openhands.tools.browser_use import BrowserToolSet

        tools.append(Tool(name=BrowserToolSet.name))

    mcp_config: dict[str, Any] = {}
    if enable_tavily:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise RuntimeError(
                "--tavily requires TAVILY_API_KEY in the environment; "
                "the key is never written to trace files"
            )
        # Inject the already-resolved secret here. The low-level MCP client
        # does not expand ${TAVILY_API_KEY} placeholders by itself.
        mcp_config = {
            "mcpServers": {
                "tavily": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.2.1"],
                    "env": {"TAVILY_API_KEY": tavily_api_key},
                }
            }
        }

    return Agent(llm=llm, tools=tools, mcp_config=mcp_config)


def main() -> int:
    args = parse_args()
    model = normalize_model_for_base_url(args.model, args.base_url)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    llm_trace_file = Path(args.llm_trace_file).expanduser()
    if not llm_trace_file.is_absolute():
        llm_trace_file = REPO_ROOT / llm_trace_file
    completion_log_dir = Path(args.completion_log_dir).expanduser()
    if not completion_log_dir.is_absolute():
        completion_log_dir = REPO_ROOT / completion_log_dir
    report_path = output_dir / "report.md"
    sources_path = output_dir / "sources.json"
    notes_path = output_dir / "notes.md"
    tool_records = DEFAULT_TOOL_RECORDS.expanduser()

    hook_config = HookConfig.load(working_dir=REPO_ROOT)
    if hook_config.is_empty():
        print(f"Warning: no hooks loaded from {REPO_ROOT / '.openhands/hooks.json'}")

    llm = TracingLLM(
        usage_id="local-deep-search",
        model=model,
        api_key=SecretStr(args.api_key),
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        stream=args.stream,
        log_completions=True,
        log_completions_folder=str(completion_log_dir),
        trace_file=str(llm_trace_file),
        include_payloads=not args.no_llm_payloads,
    )
    agent = build_agent(llm, enable_browser=args.browser, enable_tavily=args.tavily)
    conversation = Conversation(
        agent=agent,
        workspace=str(REPO_ROOT),
        hook_config=hook_config,
        token_callbacks=[lambda _chunk: None] if args.stream else None,
    )

    web_policy = (
        "Use the Tavily MCP tools for web research. Use tavily-search for "
        "discovery and tavily-extract when you need the contents of a specific "
        "URL. Do not use browser tools or terminal commands for web research."
        if args.tavily
        else "Use terminal commands such as curl, python, or text extraction "
        "utilities to fetch and inspect pages. If browser tools are available, "
        "use them when useful."
    )
    prompt = f"""
You are running a deep-search service test for OpenHands with a local LLM.

Research question:
{args.query}

Use tools to perform the research. Do not answer from memory only.

Web-tool policy:
{web_policy}

Requirements:
1. Create this directory if needed: {output_dir}
2. Inspect at least {args.min_sources} independent sources. Prefer primary sources,
   official docs, papers, release notes, benchmark pages, or source repositories.
3. Follow the web-tool policy above and inspect the returned contents rather than
   answering from search snippets alone.
4. Save structured source metadata to {sources_path}. Each source item should
   include title, url or local path, publisher, access_time, and the specific claim
   it supports.
5. Save working notes to {notes_path}.
6. Write the final Chinese report to {report_path}.

The final report must include:
- 结论摘要
- 调研过程
- 关键证据与来源
- 不同来源之间的一致点和冲突点
- 风险、局限性和仍需人工确认的问题
- 后续可继续深挖的方向

After writing the files, verify that {report_path}, {sources_path}, and {notes_path}
exist, then report their paths.
""".strip()

    print(f"Model: {model}")
    print(f"Base URL: {args.base_url}")
    print(f"Max output tokens: {args.max_output_tokens}")
    print(f"Browser enabled: {args.browser}")
    print(f"Tavily enabled: {args.tavily}")
    print(f"Streaming enabled: {args.stream}")
    print(f"Workspace: {REPO_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Tool records: {tool_records}")
    print(f"LLM traces: {llm_trace_file}")
    print(f"Completion logs: {completion_log_dir}")
    print("-" * 80)

    conversation.send_message(prompt)
    conversation.run()

    cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
    print("-" * 80)
    print(f"Report: {report_path}")
    print(f"Sources: {sources_path}")
    print(f"Notes: {notes_path}")
    print(f"Tool records: {tool_records}")
    print(f"LLM traces: {llm_trace_file}")
    print(f"Completion logs: {completion_log_dir}")
    print(f"EXAMPLE_COST: {cost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
