"""Synchronous OpenHands bridge to the unmodified official BrowseComp MCP tools."""

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

from anyio.from_thread import start_blocking_portal
from fastmcp import Client
from mcp.types import TextContent


@dataclass(frozen=True)
class NativeObservation:
    text: str
    data: object


class NativeBrowseCompEnvironment:
    def __init__(self, config):
        self.config = config
        self.tool_definitions = {}
        self._portal_context = None
        self._client_context = None
        self._portal = None
        self._client = None
        self._timeout = config.runtime.tool_timeout

    def prepare(self):
        self._portal_context = start_blocking_portal()
        self._portal = self._portal_context.__enter__()
        try:
            self._client = Client(self.config.retrieval.mcp_url, timeout=self._timeout)
            client_context = self._portal.wrap_async_context_manager(self._client)
            client_context.__enter__()
            self._client_context = client_context
            tools = self._portal.call(self._client.list_tools)
            for name, field in (("search", "query"), ("get_document", "docid")):
                matches = [tool for tool in tools if tool.name == name]
                if len(matches) != 1:
                    raise ValueError(f"Official MCP service must expose {name}")
                tool = matches[0]
                if set(tool.inputSchema.get("properties", {})) != {field}:
                    raise ValueError(f"Unexpected official MCP schema for {name}")
                self.tool_definitions[name] = tool
            schemas = [
                self.tool_definitions[name].model_dump(mode="json")
                for name in sorted(self.tool_definitions)
            ]
            return {
                "backend": "browsecomp_mcp",
                "mcp_url": self.config.retrieval.mcp_url,
                "corpus_revision": self.config.retrieval.corpus_revision,
                "observed_tools_sha256": hashlib.sha256(
                    json.dumps(schemas, sort_keys=True).encode()
                ).hexdigest(),
                "tools": schemas,
                "executor_timing_available": False,
                "server_asset_identity_verified_by_bridge": False,
            }
        except BaseException:
            self.close()
            raise

    @contextmanager
    def operation_timeout(self, seconds):
        previous, self._timeout = self._timeout, seconds
        try:
            yield
        finally:
            self._timeout = previous

    def call_tool(self, name, arguments):
        assert self._portal is not None and self._client is not None
        result = self._portal.call(
            partial(self._client.call_tool, name, arguments, timeout=self._timeout)
        )
        texts = [content.text for content in result.content if isinstance(content, TextContent)]
        text = "\n".join(texts)
        data = result.data
        if data is None and len(texts) == 1:
            try:
                data = json.loads(texts[0])
            except ValueError:
                pass
        if not texts:
            text = json.dumps(data, ensure_ascii=False)
        return NativeObservation(text, data)

    def quiesce(self):
        pass

    def close(self):
        try:
            if self._client_context is not None:
                self._client_context.__exit__(None, None, None)
        finally:
            self._client_context = self._client = None
            if self._portal_context is not None:
                self._portal_context.__exit__(None, None, None)
            self._portal_context = self._portal = None


def create_retrieval_environment(config, verified_index=None):
    if config.retrieval.backend == "browsecomp_mcp":
        return NativeBrowseCompEnvironment(config)
    from .retrieval import RetrievalEnvironment

    return RetrievalEnvironment(config, verified_index=verified_index)
