"""Readers for the released retrieval corpora without exposing question labels."""

import bz2
import json
from pathlib import Path

import pyarrow.parquet as pq


def hotpot_record(row):
    """Convert the Hugging Face column representation to official Hotpot JSON."""
    result = dict(row)
    if "_id" not in result:
        result["_id"] = str(result.pop("id"))
    if isinstance(result.get("supporting_facts"), dict):
        facts = result["supporting_facts"]
        result["supporting_facts"] = [
            list(pair) for pair in zip(facts["title"], facts["sent_id"], strict=True)
        ]
    if isinstance(result.get("context"), dict):
        context = result["context"]
        result["context"] = [
            list(pair) for pair in zip(context["title"], context["sentences"], strict=True)
        ]
    return result


def hotpot_public_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        id_key = "_id" if "_id" in parquet.schema.names else "id"
        for batch in parquet.iter_batches(columns=[id_key, "question"], batch_size=1024):
            for row in batch.to_pylist():
                yield hotpot_record(row)
    else:
        with path.open(encoding="utf-8") as stream:
            rows = (
                (json.loads(line) for line in stream if line.strip())
                if path.suffix == ".jsonl"
                else json.load(stream)
            )
            for row in rows:
                yield {
                    "_id": str(row["_id"] if "_id" in row else row["id"]),
                    "question": row["question"],
                }


def _hotpot_document(row):
    sentences = row["text"]
    if not isinstance(sentences, list) or not all(
        isinstance(sentence, str) for sentence in sentences
    ):
        raise ValueError("Use the official Hotpot abstracts with flat original sentence lists")
    return {
        "docid": str(row["id"]),
        "title": row["title"],
        "url": row["url"],
        "text": "".join(sentences),
        "sentences": sentences,
    }


def corpus_files(path):
    path = Path(path).resolve()
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Corpus path does not exist: {path}")
    files = sorted(path.rglob("*.bz2"))
    parquet = sorted(path.rglob("*.parquet"))
    if files and parquet:
        raise ValueError("Corpus directory mixes Hotpot bz2 and BrowseComp Parquet shards")
    files = files or parquet
    if not files:
        raise ValueError("Corpus directory must contain bz2 or Parquet shards")
    return files


def iter_corpus_documents(path):
    """Yield canonical documents; never reconstruct a corpus from question contexts."""
    for source in corpus_files(path):
        if source.name.endswith(".tar.bz2"):
            raise ValueError("Extract the official archive before indexing its bz2 shards")
        if source.suffix == ".bz2":
            with bz2.open(source, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield _hotpot_document(json.loads(line))
        elif source.suffix == ".parquet":
            parquet = pq.ParquetFile(source)
            columns = ["docid", "text", "url"]
            if "title" in parquet.schema.names:
                columns.append("title")
            for batch in parquet.iter_batches(columns=columns, batch_size=256):
                yield from batch.to_pylist()
        elif source.suffix == ".jsonl":
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        else:
            raise ValueError("Expected canonical JSONL, official Hotpot bz2, or corpus Parquet")
