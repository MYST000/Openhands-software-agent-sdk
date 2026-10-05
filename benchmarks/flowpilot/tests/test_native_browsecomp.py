import json

from fastmcp import Client, FastMCP
from openhands.sdk import LLM, Agent, LocalConversation
from openhands.sdk.flowpilot import FlowPilotConfig, FlowPilotRuntime

from benchmark_adapters import native_browsecomp
from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.retrieval_tools import (
    GetDocumentTool,
    NativeGetDocumentAction,
    NativeSearchAction,
    RetrievalExecutor,
    SearchTool,
    make_tools,
)
from benchmark_adapters.sdk_bridge import Binding, bind, unbind
from benchmark_adapters.tracing import Budget, TraceRecorder


def test_native_bridge_preserves_server_schema_and_full_observations(tmp_path, monkeypatch):
    server = FastMCP("controlled-fixture")
    full_text = "Unpaginated text " * 1000

    @server.tool
    def search(query: str) -> list[dict]:
        """Official-shaped search with a server-fixed k."""
        return [{"docid": "42", "snippet": query, "score": 1.25}]

    @server.tool
    def get_document(docid: str) -> dict:
        """Official-shaped full-document retrieval."""
        return {"docid": docid, "text": full_text}

    monkeypatch.setattr(native_browsecomp, "Client", lambda url, **kwargs: Client(server, **kwargs))
    config = Config(
        dataset=DatasetConfig(kind="browsecomp", id="Tevatron/browsecomp-plus"),
        retrieval=RetrievalConfig(backend="browsecomp_mcp", read_chars=3),
    )
    environment = native_browsecomp.create_retrieval_environment(config)
    recorder = TraceRecorder(tmp_path / "trace", {"clock_domain": "controller-fixture"})
    binding = Binding(environment, recorder, Budget(4, 1, 60), 30, 3)
    binding_key = bind(binding)
    try:
        identity = environment.prepare()
        assert identity["executor_timing_available"] is False
        search_tool = SearchTool.create(binding_key=binding_key)[0]
        read_tool = GetDocumentTool.create(binding_key=binding_key)[0]
        for tool in (search_tool, read_tool):
            assert tool.annotations is not None and tool.annotations.readOnlyHint is True
            # The read-only declaration must not change the model's tool schema.
            for predict_risk in (False, True):
                assert tool.to_openai_tool(predict_risk) == tool.model_copy(
                    update={"annotations": None}
                ).to_openai_tool(predict_risk)
                function = tool.to_openai_tool(predict_risk)["function"]
                assert "parameters" in function
                schema = function["parameters"]
                assert ("security_risk" in schema["properties"]) is predict_risk
        agent = Agent(
            llm=LLM(
                model="openai/Qwen3.5-27B",
                stream=False,
                caching_prompt=False,
                litellm_extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20,
                    "min_p": 0,
                    "presence_penalty": 1.5,
                    "repetition_penalty": 1,
                },
            ),
            tools=make_tools(binding_key),
            condenser=None,
            tool_concurrency_limit=1,
        )
        conversation = LocalConversation(agent=agent, workspace=str(tmp_path), visualizer=None)
        conversation.send_message("Search the fixed corpus")
        agent = conversation.agent
        assert isinstance(agent, Agent)
        # Eligibility uses the actual runtime/config without registering a remote job.
        conversation._flowpilot_runtime = FlowPilotRuntime(
            FlowPilotConfig(
                enabled=True,
                gateway_url="http://fixture",
                api_key="fixture",
                job_id="job-fixture",
                line_id="line-fixture",
                exact_reuse_enabled=True,
                semantic_reuse_enabled=True,
                deferred_context_enabled=True,
                reusable_web_tools=("search", "get_document"),
            ),
            conversation_id=str(conversation.state.id),
        )
        try:
            assert agent._flowpilot_delegation_is_safe(conversation, conversation.state, [])
        finally:
            conversation._flowpilot_runtime = None
            conversation.close()
        assert set(search_tool.action_type.model_fields) == set(NativeSearchAction.model_fields)
        assert "top_k" not in search_tool.action_type.model_fields
        assert "offset" not in read_tool.action_type.model_fields
        RetrievalExecutor(binding, "search")(NativeSearchAction(query="query"))
        RetrievalExecutor(binding, "get_document")(NativeGetDocumentAction(docid="42"))
        assert recorder.retrieved_docids == {"42"}
    finally:
        unbind(binding_key)
        environment.close()
        recorder.close()
    events = [
        json.loads(line) for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    completed = [event for event in events if event["event"] == "tool_end"]
    assert len(completed) == 2
    assert completed[0]["effective_arguments"] == {"query": "query"}
    assert all(event["executor_duration_ms"] is None for event in completed)
    assert all(event["round_trip_ms"] > 0 for event in completed)
    assert all(event["executor_clock_domain"] is None for event in completed)
    assert json.loads(completed[1]["model_observation"])["text"] == full_text
