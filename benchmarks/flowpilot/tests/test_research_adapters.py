import json
import sqlite3
from dataclasses import replace

import pytest
from pydantic import ValidationError

from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.research_environment import ResearchEnvironment, file_digest
from benchmark_adapters.research_tasks import load_tasks
from benchmark_adapters.research_tools import PaperFinishAction
from benchmark_adapters.reuse_profile import retrieval_scope


def configured(tmp_path):
    directory = tmp_path / "dbs" / "a"
    directory.mkdir(parents=True)
    db = directory / "a.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE values_table(x INTEGER)")
        connection.execute("INSERT INTO values_table VALUES (3)")
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"assets": [{"path": str(db), "sha256": file_digest(db)}]}))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "question_id": 1,
                "index": 0,
                "db_id": "a",
                "question": "Return x",
                "evidence": "x is an integer",
                "SQL": "PRIVATE_GOLD_MUST_NOT_LEAK",
            }
        )
        + "\n"
    )
    return Config(
        dataset=DatasetConfig(
            kind="bird",
            id="bird-bench/mini-dev",
            path=str(tasks),
            revision="v1",
            setting="openhands-sqlite-v1",
        ),
        retrieval=RetrievalConfig(
            index_path=str(directory.parent),
            asset_manifest=str(manifest),
            corpus_revision=file_digest(manifest),
        ),
    )


def test_agent_receives_question_and_public_hints_only(tmp_path):
    task = load_tasks(configured(tmp_path), ["1"])[0]
    assert "PRIVATE_GOLD" not in json.dumps(task.to_dict())
    assert task.public_metadata["db_id"] == "a"


@pytest.mark.parametrize(
    "sql", ["DELETE FROM values_table", "ATTACH ':memory:' AS other", "PRAGMA writable_schema=ON"]
)
def test_database_is_read_only_and_cannot_attach_other_resources(tmp_path, sql):
    environment = ResearchEnvironment(configured(tmp_path))
    environment.prepare()
    assert environment.execute_sql("a", "SELECT x FROM values_table")["rows"] == [(3,)]
    with pytest.raises(sqlite3.DatabaseError):
        environment.execute_sql("a", sql)
    environment.close()


def test_mutated_corpus_cannot_use_stale_reuse_namespace(tmp_path):
    config = configured(tmp_path)
    db = tmp_path / "dbs/a/a.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("UPDATE values_table SET x=4")
    with pytest.raises(ValueError, match="Corpus changed"):
        ResearchEnvironment(config).prepare()


def test_qasper_finish_enforces_short_answer_and_evidence():
    with pytest.raises(ValidationError):
        PaperFinishAction(answer="word " * 33, evidence_ids=[0])
    with pytest.raises(ValidationError):
        PaperFinishAction(answer="short", evidence_ids=[])


def test_paper_and_database_reuse_namespaces_differ(tmp_path):
    config = configured(tmp_path)
    other = replace(config, dataset=replace(config.dataset, kind="qasper"))
    assert retrieval_scope(config) != retrieval_scope(other)


def test_real_http_rate_limit_wait_is_included_in_tool_time(tmp_path):
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_GET(self):
            Handler.calls += 1
            if Handler.calls == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0.03")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"actual response")

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = ResearchEnvironment(configured(tmp_path))
    try:
        start = time.monotonic()
        assert environment.request(f"http://127.0.0.1:{server.server_port}/") == "actual response"
        duration = time.monotonic() - start
        attempts = environment.last_http_attempts
        assert [row["status_code"] for row in attempts] == [429, 200]
        assert attempts[0]["retry_wait_ms"] >= 25
        assert duration >= 0.03
        assert sum(row["duration_ms"] for row in attempts) / 1000 <= duration + 0.01
    finally:
        environment.close()
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def test_shared_paper_reading_preserves_ids_and_truncation(tmp_path):
    config = configured(tmp_path)
    document = {
        "paper_id": "a",
        "title": "Title",
        "abstract": "Abstract",
        "paragraphs": [
            {"paragraph_id": i, "section_name": "Introduction", "text": "word" * 10}
            for i in range(3)
        ],
    }
    path = tmp_path / "papers.jsonl"
    path.write_text(json.dumps(document) + "\n")
    manifest = tmp_path / "papers.assets.json"
    manifest.write_text(json.dumps({"assets": [{"path": str(path), "sha256": file_digest(path)}]}))
    config = replace(
        config,
        dataset=replace(config.dataset, kind="qasper"),
        retrieval=replace(
            config.retrieval,
            index_path=str(path),
            asset_manifest=str(manifest),
            corpus_revision=file_digest(manifest),
            qasper_context_mode="document",
        ),
        runtime=replace(config.runtime, max_output_chars=50),
    )
    environment = ResearchEnvironment(config)
    environment.prepare()
    try:
        result = environment.read_paper("a")
        assert result["paragraphs"] == document["paragraphs"][:1]
        assert result["truncated"] is True
        assert environment.read_paper("a") == result
        with pytest.raises(KeyError):
            environment.read_paper("other")
    finally:
        environment.close()


def test_native_finish_message_is_hidden_from_model_schema():
    schema = PaperFinishAction.model_json_schema()
    assert "message" not in schema["properties"]
    assert PaperFinishAction(answer="short", evidence_ids=[0]).message == ""


def test_unstructured_finish_recovery_is_bounded_and_budgeted():
    from types import SimpleNamespace

    from benchmark_adapters.research_tasks import recover_structured_finish

    environment = SimpleNamespace(finished_data=None)
    calls = []

    class Conversation:
        state = SimpleNamespace(execution_status=SimpleNamespace(value="finished"))

        def send_message(self, message):
            calls.append("prompt")

        def run(self):
            calls.append("run")

    budget = SimpleNamespace(check=lambda: calls.append("budget"))
    recorder = SimpleNamespace(emit=lambda *args, **kwargs: calls.append("retry"))
    recover_structured_finish(Conversation(), environment, recorder, budget)
    assert calls == ["budget", "retry", "prompt", "run"] * 2
    environment.finished_data = {"answer": "short", "evidence_ids": [0]}
    calls.clear()
    recover_structured_finish(Conversation(), environment, recorder, budget)
    assert calls == ["budget"]
