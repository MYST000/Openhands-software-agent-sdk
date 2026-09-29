import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_adapters import retrieval
from benchmark_adapters.config import Config, DatasetConfig, RetrievalConfig
from benchmark_adapters.native_browsecomp import create_retrieval_environment


@pytest.fixture
def verified_index(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps(dict(docid="1", title="Alpha", sentences=["Alpha."])) + "\n")
    path = tmp_path / "index.sqlite3"
    retrieval.build_index(corpus, path, "fixture")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id="hotpotqa"),
        retrieval=RetrievalConfig(index_path=str(path), corpus_revision="fixture"),
    )
    environment = create_retrieval_environment(config)
    try:
        metadata = environment.prepare()
    finally:
        environment.close()
    return config, metadata


def test_worker_verified_index_skips_hash_and_sql_scans(verified_index, monkeypatch):
    config, metadata = verified_index
    statements = []
    connect = retrieval.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    def forbidden_hash(*args, **kwargs):
        raise AssertionError("full index hash requested")

    monkeypatch.setattr(retrieval.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(retrieval.hashlib, "file_digest", forbidden_hash)
    environment = create_retrieval_environment(config, verified_index=metadata)
    assert isinstance(environment, retrieval.RetrievalEnvironment)
    try:
        current = environment.prepare()
        assert current["validation_mode"] == "identity_only"
        assert current["index_identity"] == metadata["index_identity"]
        assert statements == []
        assert environment.read("1")["sentences"] == [[0, "Alpha."]]
    finally:
        environment.close()
    with pytest.raises(AssertionError, match="full index hash"):
        create_retrieval_environment(config).prepare()


@pytest.mark.parametrize("change", ["stat", "manifest", "kind"])
def test_verified_index_identity_change_is_rejected(verified_index, change):
    config, metadata = verified_index
    path = Path(config.retrieval.index_path)
    if change == "stat":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    elif change == "manifest":
        manifest = Path(str(path) + ".manifest.json")
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    else:
        config = replace(config, dataset=replace(config.dataset, kind="browsecomp"))
    environment = create_retrieval_environment(config, verified_index=metadata)
    try:
        with pytest.raises(ValueError, match="Verified index identity changed"):
            environment.prepare()
    finally:
        environment.close()
