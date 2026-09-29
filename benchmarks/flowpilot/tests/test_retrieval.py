import json
from dataclasses import replace

import pytest


def test_fixed_index_preserves_sentence_ids_and_handles_literal_queries(tmp_path):
    from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
    from benchmark_adapters.retrieval import RetrievalEnvironment, build_index, parse_hotpot_answer

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            dict(docid="1", title="Alpha", sentences=["First alpha.", "Second alpha.", "Third."])
        )
        + "\n"
    )
    index = tmp_path / "index.sqlite3"
    build_index(corpus, index, "corpus-v1")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa"),
        retrieval=RetrievalConfig(index_path=str(index), corpus_revision="corpus-v1"),
    )
    env = RetrievalEnvironment(config)
    try:
        env.prepare()
        assert env.search('alpha " OR 1=1 --', 5)[0]["docid"] == "1"
        document = env.read("1", start_sentence=1, max_sentences=1)
        assert document["sentences"] == [[1, "Second alpha."]]
        answer = parse_hotpot_answer('{"answer":"yes","supporting_facts":[["Alpha",1]]}', env)
        assert answer["sp"] == [["Alpha", 1]]
        wrong_citation = parse_hotpot_answer(
            '{"answer":"yes","supporting_facts":[["Alpha",99]]}', env
        )
        assert wrong_citation == {"answer": "yes", "sp": [["Alpha", 99]]}
    finally:
        env.close()
    bad = RetrievalEnvironment(
        replace(config, retrieval=replace(config.retrieval, corpus_revision="different"))
    )
    with pytest.raises(ValueError, match="revision"):
        bad.prepare()
    bad.close()


def test_browse_export_uses_observed_ids_and_true_completion():
    from types import SimpleNamespace

    from benchmark_adapters.retrieval import BrowseCompAdapter, export_answer

    task = BrowseCompAdapter.from_record(
        dict(query_id="q1", query="Question", answer="SECRET"),
        dataset_id="Tevatron/browsecomp-plus",
        revision="v",
        split="test",
    )
    trace = SimpleNamespace(
        final_text="Answer citing invented-document",
        retrieved_docids={"observed-document"},
        executed_counts={"search": 2},
    )
    result = export_answer(
        "browsecomp", task, trace, {"execution_status": "budget_exhausted"}, None
    )
    assert result["status"] != "completed"
    assert result["retrieved_docids"] == ["observed-document"]
    assert "SECRET" not in task.instruction
