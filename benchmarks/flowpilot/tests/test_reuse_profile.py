from dataclasses import replace

import pytest
from openhands.sdk.event import ObservationEvent
from openhands.sdk.flowpilot import FlowPilotConfig
from openhands.sdk.llm import TextContent

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.predictor_integration import (
    PredictorTraceRecorder,
    benchmark_flowpilot_config,
)
from benchmark_adapters.retrieval_tools import RetrievalObservation
from benchmark_adapters.reuse_profile import export_registry, retrieval_scope


@pytest.fixture
def retrieval_config():
    return Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa", revision="questions-v1"),
        retrieval=RetrievalConfig(corpus_revision="corpus-v1", index_path="/data/index.sqlite"),
    )


@pytest.mark.parametrize(
    "kind,backend,adapters",
    [
        ("hotpot", "sqlite", ["benchmark_search_v1", "benchmark_read_document_v1"]),
        ("hotpot", "hotpot_rpc", ["benchmark_search_v1", "benchmark_read_document_v1"]),
        ("browsecomp", "sqlite", ["benchmark_search_v1", "benchmark_get_document_v1"]),
        (
            "browsecomp",
            "browsecomp_mcp",
            ["benchmark_native_search_v1", "benchmark_native_get_document_v1"],
        ),
    ],
)
def test_export_matches_runtime_scope_and_reader_policy(
    monkeypatch, retrieval_config, kind, backend, adapters
):
    config = replace(
        retrieval_config,
        dataset=replace(retrieval_config.dataset, kind=kind),
        retrieval=replace(
            retrieval_config.retrieval, backend=backend, server_policy_revision="server-v1"
        ),
    )
    monkeypatch.setenv("FLOWPILOT_PREDICTOR_GATEWAY", "http://flowpilot")
    monkeypatch.setenv("FLOWPILOT_INGRESS_API_KEY", "test-key")
    monkeypatch.setenv("FLOWPILOT_REUSE_ENABLED", "1")
    monkeypatch.delenv("FLOWPILOT_EXPERIMENT_PROFILE", raising=False)
    runtime = benchmark_flowpilot_config(
        config, {"run_id": "job", "attempt_id": "try1", "task_id": "1"}
    )
    registry = export_registry(config)
    assert [row["adapter_id"] for row in registry] == adapters
    assert tuple(row["tool_name"] for row in registry) == runtime.reusable_web_tools
    assert runtime.exact_reuse_enabled
    for row in registry:
        assert tuple(row["required_data_source_constraints"]) == runtime.data_source_constraints
        assert row["semantic_reuse_enabled"] == (row["tool_name"] == "search")
        assert row["policy_digest"] == runtime.data_source_constraints[1].split(":")[1]


@pytest.mark.parametrize(
    "change",
    [
        dict(corpus_revision="corpus-v2"),
        dict(top_k=10),
        dict(snippet_chars=200),
        dict(read_chars=3000),
        dict(server_policy_revision="server-v2"),
        dict(backend="hotpot_rpc", server_policy_revision="server-v1"),
    ],
)
def test_execution_settings_remain_hard_constraints(retrieval_config, change):
    changed = replace(retrieval_config, retrieval=replace(retrieval_config.retrieval, **change))
    assert retrieval_scope(changed) != retrieval_scope(retrieval_config)


def test_remote_policy_and_corpus_must_be_declared(retrieval_config):
    for retrieval, message in (
        (replace(retrieval_config.retrieval, corpus_revision=""), "corpus_revision"),
        (replace(retrieval_config.retrieval, backend="hotpot_rpc"), "server_policy_revision"),
    ):
        with pytest.raises(ValueError, match=message):
            export_registry(replace(retrieval_config, retrieval=retrieval))


def test_prediction_alone_leaves_identity_to_conversation_and_reuse_disabled(
    monkeypatch, retrieval_config
):
    monkeypatch.setenv("FLOWPILOT_PREDICTOR_GATEWAY", "http://flowpilot")
    monkeypatch.setenv("FLOWPILOT_INGRESS_API_KEY", "test-key")
    monkeypatch.delenv("FLOWPILOT_REUSE_ENABLED", raising=False)
    monkeypatch.delenv("FLOWPILOT_EXPERIMENT_PROFILE", raising=False)
    identity = {"run_id": "job", "attempt_id": "try1", "task_id": "1"}
    left = benchmark_flowpilot_config(retrieval_config, identity)
    right = benchmark_flowpilot_config(
        replace(retrieval_config, dataset=replace(retrieval_config.dataset, kind="browsecomp")),
        identity,
    )
    assert left.job_id == right.job_id == ""
    assert left.line_id == right.line_id == ""
    assert not left.exact_reuse_enabled
    assert not left.data_source_constraints


def test_reused_mcp_json_lines_preserve_all_docids_without_timing_feedback(tmp_path):
    recorder = PredictorTraceRecorder(
        tmp_path,
        {"clock_domain": "fixture"},
        benchmark="browsecomp",
        flowpilot_config=FlowPilotConfig(gateway_url="http://unused"),
    )
    observation = RetrievalObservation.from_text(
        '{"docid":"1","snippet":"one"}\n{"docid":"2","snippet":"two"}'
    )
    observation.content.append(TextContent(text="\n[FlowPilot reuse provenance: fixture]"))
    try:
        recorder.sdk_event(
            ObservationEvent(
                tool_name="search",
                tool_call_id="current-call",
                action_id="current-action",
                observation=observation,
            )
        )
        assert recorder.retrieved_docids == {"1", "2"}
        assert not recorder.executed_counts
        assert not recorder.predictor_metrics
    finally:
        recorder.close()
