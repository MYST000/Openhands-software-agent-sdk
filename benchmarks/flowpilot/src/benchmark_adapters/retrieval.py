import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from .contracts import Task
from .swe import checked_dataset_digest, load_records, select_records
from .tracing import write_json


class HotpotAdapter:
    @staticmethod
    def from_record(row, *, dataset_id, revision, split):
        question, task_id = row["question"], str(row["_id"])
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Hotpot question is empty")
        instruction = (
            "Answer using the fixed Wikipedia corpus. Return a JSON object in your finish message with "
            '"answer" (string) and "supporting_facts" (list of [official title, zero-based sentence ID]). '
            "Use the original sentence numbers returned by the tools.\n\nQuestion: " + question
        )
        return Task(dataset_id, revision, split, task_id, instruction)

    @classmethod
    def load(cls, path, *, dataset_id, revision, split, ids=None, limit=0):
        rows = select_records(load_records(path), "_id", ids, limit)
        return [
            cls.from_record(r, dataset_id=dataset_id, revision=revision, split=split) for r in rows
        ]


class BrowseCompAdapter:
    @staticmethod
    def from_record(row, *, dataset_id, revision, split):
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("BrowseComp-Plus query is empty")
        instruction = (
            "Answer this research question using only the fixed corpus. Search and read documents as needed. "
            "Finish with a concise answer and supporting document IDs.\n\nQuestion: " + row["query"]
        )
        return Task(dataset_id, revision, split, str(row["query_id"]), instruction)

    @classmethod
    def load(cls, path, *, dataset_id, revision, split, ids=None, limit=0):
        if Path(path).suffix == ".jsonl":
            with Path(path).open() as f:
                rows = []
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        rows.append({"query_id": row["query_id"], "query": row["query"]})
        else:
            rows = load_records(path)
        rows = select_records(rows, "query_id", ids, limit)
        return [
            cls.from_record(r, dataset_id=dataset_id, revision=revision, split=split) for r in rows
        ]


def build_index(corpus, output, revision):
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if not revision:
        raise ValueError("retrieval.corpus_revision is required")
    if output.exists():
        raise ValueError("Index already exists; use a new versioned path instead of overwriting")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise ValueError("An incomplete index exists; inspect and remove it before rebuilding")
    con = sqlite3.connect(partial)
    count, source_hash = 0, hashlib.sha256()
    try:
        con.execute(
            "CREATE TABLE docs(docid TEXT PRIMARY KEY,title TEXT NOT NULL,url TEXT NOT NULL,text TEXT NOT NULL,sentences TEXT NOT NULL)"
        )
        con.execute(
            "CREATE VIRTUAL TABLE docs_fts USING fts5(docid UNINDEXED,text,content='docs',content_rowid='rowid')"
        )
        with corpus.open("rb") as f:
            for line in f:
                source_hash.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                docid = str(row["docid"])
                sentences = row.get("sentences", [])
                if not isinstance(sentences, list) or not all(
                    isinstance(s, str) for s in sentences
                ):
                    raise ValueError("sentences must contain original sentence strings")
                text = row.get("text") or "\n".join(sentences)
                if not docid or not isinstance(text, str) or not text:
                    raise ValueError(
                        "Each corpus document requires docid and nonempty text/sentences"
                    )
                con.execute(
                    "INSERT INTO docs VALUES (?,?,?,?,?)",
                    (
                        docid,
                        str(row.get("title", docid)),
                        str(row.get("url", "")),
                        text,
                        json.dumps(sentences),
                    ),
                )
                count += 1
        if not count:
            raise ValueError("Empty corpus")
        con.execute("INSERT INTO docs_fts(rowid,docid,text) SELECT rowid,docid,text FROM docs")
        con.execute("INSERT INTO docs_fts(docs_fts,rank) VALUES('integrity-check',1)")
        con.commit()
    except BaseException:
        con.close()
        partial.unlink(missing_ok=True)
        raise
    con.close()
    partial.replace(output)
    with output.open("rb") as f:
        index_hash = hashlib.file_digest(f, "sha256").hexdigest()
    write_json(
        str(output) + ".manifest.json",
        dict(
            schema_version=1,
            corpus_revision=revision,
            source_sha256=source_hash.hexdigest(),
            index_sha256=index_hash,
            document_count=count,
            retriever="SQLite FTS5 BM25; OR of quoted word tokens",
            sqlite_version=sqlite3.sqlite_version,
        ),
    )


_VERIFIED_INDEXES = set()


class RetrievalEnvironment:
    def __init__(self, config):
        self.config = config
        self.connection = None

    def prepare(self):
        path = Path(self.config.retrieval.index_path).resolve()
        metadata = json.loads(Path(str(path) + ".manifest.json").read_text())
        if metadata["corpus_revision"] != self.config.retrieval.corpus_revision:
            raise ValueError("Corpus revision does not match the profile")
        stat = path.stat()
        key = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, metadata["index_sha256"])
        if key not in _VERIFIED_INDEXES:
            with path.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != metadata["index_sha256"]:
                    raise ValueError("Index checksum mismatch")
            _VERIFIED_INDEXES.add(key)
        self.connection = sqlite3.connect(
            path.as_uri() + "?mode=ro", uri=True, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        if (
            self.connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            != metadata["document_count"]
        ):
            raise ValueError("Index document count mismatch")
        if self.config.dataset.kind == "hotpot":
            if self.connection.execute(
                "SELECT 1 FROM docs WHERE sentences='[]' LIMIT 1"
            ).fetchone():
                raise ValueError("Hotpot requires original sentences for every document")
            if self.connection.execute(
                "SELECT title FROM docs GROUP BY title HAVING COUNT(*)>1 LIMIT 1"
            ).fetchone():
                raise ValueError("Hotpot corpus requires unique canonical titles")
        return metadata

    def search(self, query, top_k):
        if not 1 <= top_k <= self.config.retrieval.top_k:
            raise ValueError("top_k exceeds the fixed profile limit")
        tokens = re.findall(r"\w+", query, re.UNICODE)[:64]
        if not tokens:
            return []
        match = " OR ".join('"' + t + '"' for t in tokens)
        rows = self.connection.execute(
            "SELECT d.docid,d.title,d.url,substr(d.text,1,?) snippet FROM docs_fts f "
            "JOIN docs d ON d.rowid=f.rowid WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts),d.docid LIMIT ?",
            (self.config.retrieval.snippet_chars, match, top_k),
        ).fetchall()
        return [dict(row) for row in rows]

    def read(self, docid, *, offset=0, start_sentence=0, max_sentences=20):
        row = self.connection.execute("SELECT * FROM docs WHERE docid=?", (docid,)).fetchone()
        if row is None:
            raise ValueError("Unknown document ID")
        if self.config.dataset.kind == "hotpot":
            sentences = json.loads(row["sentences"])
            if start_sentence < 0 or not 1 <= max_sentences <= 100:
                raise ValueError("Invalid sentence range")
            end = min(len(sentences), start_sentence + max_sentences)
            return dict(
                docid=docid,
                title=row["title"],
                sentences=[[i, sentences[i]] for i in range(start_sentence, end)],
                next_sentence=end if end < len(sentences) else None,
            )
        if offset < 0:
            raise ValueError("offset must be nonnegative")
        end = min(len(row["text"]), offset + self.config.retrieval.read_chars)
        return dict(
            docid=docid,
            title=row["title"],
            url=row["url"],
            text=row["text"][offset:end],
            offset=offset,
            truncated=end < len(row["text"]),
            next_offset=end if end < len(row["text"]) else None,
        )

    def quiesce(self):
        pass

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def parse_hotpot_answer(text, environment):
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    data = json.loads(text)
    if not isinstance(data.get("answer"), str) or not isinstance(
        data.get("supporting_facts"), list
    ):
        raise ValueError("Expected answer and supporting_facts in final JSON")
    facts = data["supporting_facts"]
    for fact in facts:
        if (
            not isinstance(fact, list)
            or len(fact) != 2
            or not isinstance(fact[0], str)
            or type(fact[1]) is not int
        ):
            raise ValueError("Supporting facts must be [title, integer sentence ID]")
        row = environment.connection.execute(
            "SELECT sentences FROM docs WHERE title=?", (fact[0],)
        ).fetchone()
        if row is None or not 0 <= fact[1] < len(json.loads(row["sentences"])):
            raise ValueError("Supporting fact has unknown title or sentence ID")
    return {"answer": data["answer"], "sp": facts}


def export_answer(kind, task, recorder, result, environment):
    if kind == "hotpot":
        return dict(task_id=task.task_id, **parse_hotpot_answer(recorder.final_text, environment))
    complete = result["execution_status"] == "completed" and bool(recorder.final_text.strip())
    return dict(
        query_id=task.task_id,
        tool_call_counts=dict(recorder.executed_counts),
        status="completed"
        if complete
        else result["execution_status"]
        if result["execution_status"] != "completed"
        else "empty",
        retrieved_docids=sorted(recorder.retrieved_docids),
        result=[dict(type="output_text", output=recorder.final_text)]
        if recorder.final_text
        else [],
    )


def retrieval_tools(binding_key):
    from .retrieval_tools import make_tools

    return make_tools(binding_key)


def evaluate_retrieval(config, run_dir, ids, prediction, execute):
    checked_dataset_digest(config)
    kind = config.dataset.kind
    key = "_id" if kind == "hotpot" else "query_id"
    rows = select_records(load_records(config.dataset.path), key, ids)
    prediction = Path(prediction).resolve()
    files = sorted(prediction.glob("*.json")) if prediction.is_dir() else [prediction]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + path.read_bytes())
    digest.update(json.dumps(rows, sort_keys=True).encode())
    digest.update(json.dumps(config.to_dict()["evaluation"], sort_keys=True).encode())
    root = Path(run_dir).resolve() / "evaluations" / ("eval_" + digest.hexdigest()[:20])
    root.mkdir(parents=True, exist_ok=True)
    python = config.evaluation.python or sys.executable
    if kind == "hotpot":
        gold = root / "gold.json"
        write_json(gold, rows)
        script = Path(config.evaluation.hotpot_script)
        if not script.is_file():
            raise ValueError("Set evaluation.hotpot_script to the official hotpot_evaluate_v1.py")
        command = [python, str(script.resolve()), str(prediction), str(gold)]
    else:
        repo = Path(config.evaluation.browsecomp_repo)
        gold = root / "gold.jsonl"
        gold.write_text("".join(json.dumps(r) + "\n" for r in rows))
        command = [
            python,
            str(repo / "scripts_evaluation/evaluate_run.py"),
            "--input_dir",
            str(prediction),
            "--ground_truth",
            str(gold),
            "--eval_dir",
            str(root / "judge"),
            "--model",
            config.evaluation.judge_model,
            "--qrel_evidence",
            str(repo / "topics-qrels/qrel_evidence.txt"),
        ]
    plan = dict(
        evaluator=kind, command=command, cwd=str(root), task_ids=ids, evaluation_status="pending"
    )
    write_json(root / "evaluation_plan.json", plan)
    if execute:
        with (root / "evaluator.log").open("w") as log:
            proc = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        plan.update(
            returncode=proc.returncode,
            evaluation_status="executed_review_report"
            if proc.returncode == 0
            else "evaluator_error",
        )
        write_json(root / "evaluation_status.json", plan)
    return plan
