import bz2
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.retrieval import (
    HotpotAdapter,
    RetrievalEnvironment,
    build_index,
    evaluate_retrieval,
)
from benchmark_adapters.retrieval_corpus import hotpot_record


def test_official_hotpot_shards_keep_sentence_boundaries(tmp_path):
    corpus = tmp_path / "abstracts" / "AA"
    corpus.mkdir(parents=True)
    sentences = ["Alpha is first. ", "Dr. Beta is second."]
    (corpus / "wiki_00.bz2").write_bytes(
        bz2.compress(
            (
                json.dumps(dict(id="17", title="Alpha", url="wiki/Alpha", text=sentences)) + "\n"
            ).encode()
        )
    )
    (corpus / "wiki_01.bz2").write_bytes(
        bz2.compress(
            (
                json.dumps(dict(id="18", title="Empty original", url="wiki/Empty", text=[])) + "\n"
            ).encode()
        )
    )
    index = tmp_path / "index.sqlite3"
    build_index(corpus.parent, index, "official-abstracts-fixture")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa"),
        retrieval=RetrievalConfig(
            index_path=str(index), corpus_revision="official-abstracts-fixture"
        ),
    )
    environment = RetrievalEnvironment(config)
    try:
        metadata = environment.prepare()
        assert metadata["document_count"] == 2
        assert metadata["empty_text_count"] == 1
        assert environment.read("18")["sentences"] == []
        assert environment.search("Beta", 1)[0]["docid"] == "17"
        assert environment.read("17")["sentences"] == list(map(list, enumerate(sentences)))
        with pytest.raises(TimeoutError):
            with environment.operation_timeout(0):
                environment.search("Alpha", 1)
        assert environment.search("Alpha", 1)[0]["docid"] == "17"
    finally:
        environment.close()


def test_browsecomp_parquet_is_indexed_without_question_contexts(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pq.write_table(
        pa.Table.from_pylist([dict(docid="42", url="https://example.org", text="Evidence alpha")]),
        corpus / "train.parquet",
    )
    index = tmp_path / "bc.sqlite3"
    build_index(corpus, index, "bc-fixture")
    config = Config(
        dataset=DatasetConfig(kind="browsecomp", id="Tevatron/browsecomp-plus"),
        retrieval=RetrievalConfig(
            index_path=str(index), corpus_revision="bc-fixture", read_chars=8
        ),
    )
    environment = RetrievalEnvironment(config)
    try:
        environment.prepare()
        assert environment.search("Evidence", 1)[0]["docid"] == "42"
        assert environment.read("42")["next_offset"] == 8
        assert environment.read("42", offset=8)["text"] == " alpha"
    finally:
        environment.close()


def test_hf_hotpot_public_projection_and_native_evaluation(tmp_path):
    rows = [
        dict(
            id="q1",
            question="Which city?",
            answer="The Paris",
            supporting_facts={"title": ["Alpha"], "sent_id": [1]},
            context={"title": ["Alpha"], "sentences": [["Private candidate."]]},
        ),
        dict(
            id="q2",
            question="Is that true?",
            answer="yes",
            supporting_facts={"title": ["Beta"], "sent_id": [0]},
            context={"title": [], "sentences": []},
        ),
    ]
    questions = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(rows), questions)
    tasks = HotpotAdapter.load(questions, dataset_id="hotpotqa", revision="fixture", split="dev")
    assert [task.task_id for task in tasks] == ["q1", "q2"]
    assert "Private candidate" not in tasks[0].instruction
    assert "The Paris" not in tasks[0].instruction
    assert hotpot_record(rows[0])["supporting_facts"] == [["Alpha", 1]]
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {
                "answer": {"q1": "paris", "q2": "no"},
                "sp": {"q1": [["Wrong title", 99]], "q2": [["Beta", 0], ["Beta", 0]]},
            }
        )
    )
    config = Config(
        dataset=DatasetConfig(
            kind="hotpot", id="hotpotqa", path=str(questions), revision="fixture", split="dev"
        )
    )
    plan = evaluate_retrieval(config, tmp_path / "run", ["q1", "q2"], predictions, True)
    assert plan["evaluation_status"] == "scored", plan
    assert plan["metrics"]["em"] == 0.5
    assert plan["metrics"]["sp_em"] == 0.5
    assert plan["metrics"]["joint_f1"] == 0
    frozen_gold = json.loads(
        (tmp_path / "run" / "evaluations" / Path(plan["cwd"]).name / "gold.json").read_text()
    )
    assert frozen_gold[0]["_id"] == "q1"
    assert frozen_gold[0]["supporting_facts"] == [["Alpha", 1]]
    corrupted = tmp_path / "not_official.py"
    corrupted.write_text("print({})")
    config = replace(config, evaluation=replace(config.evaluation, hotpot_script=str(corrupted)))
    with pytest.raises(ValueError, match="pinned official"):
        evaluate_retrieval(config, tmp_path / "run", ["q1"], predictions, False)
