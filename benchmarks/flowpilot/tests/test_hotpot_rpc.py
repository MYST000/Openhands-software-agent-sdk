import json
import os
import threading
from dataclasses import replace

import pytest

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.hotpot_rpc import HotpotRPCEnvironment, HotpotServer
from benchmark_adapters.native_browsecomp import create_retrieval_environment
from benchmark_adapters.retrieval import build_index
from benchmark_adapters.retrieval_tools import HotpotReadAction, RetrievalExecutor, SearchAction
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, TraceRecorder


@pytest.fixture
def rpc_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"docid": "1", "title": "Alpha", "sentences": ["First alpha.", "Second."]})
        + "\n"
        + json.dumps({"docid": "2", "title": "Empty", "sentences": []})
        + "\n"
    )
    index = tmp_path / "index.sqlite3"
    build_index(corpus, index, "fixture-v1")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa"),
        retrieval=RetrievalConfig(index_path=str(index), corpus_revision="fixture-v1"),
    )
    server = HotpotServer(("127.0.0.1", 0), config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    remote = replace(
        config,
        retrieval=replace(
            config.retrieval,
            backend="hotpot_rpc",
            mcp_url=f"http://127.0.0.1:{server.server_port}",
        ),
    )
    try:
        yield server, remote, index
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_hotpot_rpc_preserves_ranking_sentences_and_separates_timing(rpc_corpus, tmp_path):
    server, config, _ = rpc_corpus
    environment = create_retrieval_environment(config)
    assert isinstance(environment, HotpotRPCEnvironment)
    recorder = TraceRecorder(tmp_path / "trace", {"clock_domain": "controller-fixture"})
    binding = Binding(environment, recorder, Budget(3, 1, 60), 30, 3)
    try:
        identity = environment.prepare()
        assert identity["index_sha256"] == server.identity["index_sha256"]
        query = 'alpha " OR 1=1 --'
        assert environment.search(query, 5) == server.environment.search(query, 5)
        assert environment.read("1", start_sentence=1, max_sentences=1) == {
            "docid": "1",
            "title": "Alpha",
            "sentences": [[1, "Second."]],
            "next_sentence": None,
        }
        assert environment.read("2")["sentences"] == []
        RetrievalExecutor(binding, "search")(SearchAction(query="alpha"))
        RetrievalExecutor(binding, "read_document")(HotpotReadAction(doc_id="1"))
        RetrievalExecutor(binding, "read_document")(HotpotReadAction(doc_id="absent"))
    finally:
        environment.close()
        recorder.close()
    events = [
        json.loads(line) for line in (tmp_path / "trace/events.jsonl").read_text().splitlines()
    ]
    completed = [event for event in events if event["event"] == "tool_end"]
    assert len(completed) == 2
    assert all(0 <= e["executor_duration_ms"] <= e["round_trip_ms"] for e in completed)
    assert all(e["executor_clock_domain"] == "hotpot-rpc-server-monotonic" for e in completed)
    errors = [event for event in events if event["event"] == "tool_error"]
    assert len(errors) == 1 and errors[0]["executor_duration_ms"] is None


def test_hotpot_rpc_rejects_corpus_mismatch_and_changed_index(rpc_corpus):
    _, config, index = rpc_corpus
    bad = HotpotRPCEnvironment(
        replace(config, retrieval=replace(config.retrieval, corpus_revision="wrong"))
    )
    try:
        with pytest.raises(ValueError, match="revision"):
            bad.prepare()
    finally:
        bad.close()
    environment = HotpotRPCEnvironment(config)
    try:
        environment.prepare()
        stat = index.stat()
        os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with pytest.raises(ValueError, match="changed"):
            environment.search("alpha", 5)
    finally:
        environment.close()
