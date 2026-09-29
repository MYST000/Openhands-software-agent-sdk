import json

import pytest

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.retrieval import RetrievalEnvironment, build_index
from benchmark_adapters.retrieval_tools import HotpotReadAction, RetrievalExecutor, SearchAction
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, BudgetExceeded, TraceRecorder


def test_retrieval_tools_record_real_observations_and_preserve_failures(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(dict(docid="1", title="Alpha", sentences=["Alpha evidence."])) + "\n"
    )
    index = tmp_path / "index.sqlite3"
    build_index(corpus, index, "fixture")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa"),
        retrieval=RetrievalConfig(index_path=str(index), corpus_revision="fixture"),
    )
    environment = RetrievalEnvironment(config)
    recorder = TraceRecorder(tmp_path / "trace", {"clock_domain": "controller-test"})
    budget = Budget(3, 1, 30)
    binding = Binding(environment, recorder, budget, 10, 10000)
    try:
        environment.prepare()
        RetrievalExecutor(binding, "search")(SearchAction(query="Alpha", top_k=1))
        RetrievalExecutor(binding, "read_document")(HotpotReadAction(doc_id="1"))
        RetrievalExecutor(binding, "read_document")(HotpotReadAction(doc_id="missing"))
        with pytest.raises(BudgetExceeded):
            RetrievalExecutor(binding, "search")(SearchAction(query="Alpha"))
        assert recorder.retrieved_docids == {"1"}
    finally:
        environment.close()
        recorder.close()
    events = [
        json.loads(line) for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    completed = [event for event in events if event["event"] == "tool_end"]
    assert len(completed) == 2
    assert completed[0]["effective_arguments"]["top_k"] == 1
    assert all(event["round_trip_ms"] >= event["executor_duration_ms"] >= 0 for event in completed)
    assert all(event["clock_domain"] == "controller-test" for event in completed)
    assert all(event["queue_wait_ms"] is None for event in completed)
    assert any(
        event["event"] == "tool_error" and event["model_observation"] == "Unknown document ID"
        for event in events
    )
    assert events[-1]["event"] == "tool_not_executed"
