import json
from dataclasses import replace

import pytest

from benchmark_adapters.cli import export_run, tasks_for
from benchmark_adapters.config import Config, DatasetConfig, EvaluationConfig, RetrievalConfig
from benchmark_adapters.twowiki import TwoWikiAdapter, evaluate_twowiki, export_answer


def test_public_prompt_does_not_include_labels_or_question_context():
    row = {
        "_id": "q1",
        "question": "Who directed the film?",
        "answer": "PRIVATE_ANSWER",
        "context": [["PRIVATE_TITLE", ["PRIVATE_CONTEXT"]]],
        "evidences": [["PRIVATE_SUBJECT", "relation", "PRIVATE_OBJECT"]],
    }
    task = TwoWikiAdapter.from_record(
        row, dataset_id="Alab-NII/2wikimultihop", revision="r1", split="dev"
    )
    assert "PRIVATE_" not in task.instruction
    assert "evidences" in task.instruction
    assert task.dataset_id == "Alab-NII/2wikimultihop"


def test_evidence_triples_are_required_and_exported():
    task = TwoWikiAdapter.from_record(
        {"_id": "q1", "question": "Who?"},
        dataset_id="Alab-NII/2wikimultihop",
        revision="r1",
        split="dev",
    )
    answer = {
        "answer": "A",
        "supporting_facts": [["Page", 0]],
        "evidences": [["Film", "director", "A"]],
    }
    assert export_answer(task, json.dumps(answer), None)["evidence"] == answer["evidences"]
    answer["evidences"] = [["incomplete", "triple"]]
    with pytest.raises(ValueError, match="string triples"):
        export_answer(task, json.dumps(answer), None)


def test_pinned_official_scorer_and_tamper_detection(tmp_path):
    import hashlib

    gold = [
        {
            "_id": "q1",
            "question": "Who?",
            "answer": "A",
            "supporting_facts": [["Page", 0]],
            "evidences": [["Film", "director", "A"]],
        }
    ]
    dataset = tmp_path / "dev.json"
    dataset.write_text(json.dumps(gold))
    config = Config(
        dataset=DatasetConfig(
            kind="twowiki",
            id="Alab-NII/2wikimultihop",
            path=str(dataset),
            revision="fixture",
            split="dev",
            sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        ),
        evaluation=EvaluationConfig(),
    )
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps(
            {
                "answer": {"q1": "A"},
                "sp": {"q1": [["Page", 0]]},
                "evidence": {"q1": [["Film", "director", "A"]]},
            }
        )
    )
    result = evaluate_twowiki(config, tmp_path / "evaluation", ["q1"], prediction, True)
    assert result["metrics"]["joint_em"] == 100.0
    corrupt = tmp_path / "corrupt.py"
    corrupt.write_text("print('fake metrics')")
    config = replace(config, evaluation=replace(config.evaluation, twowiki_script=str(corrupt)))
    with pytest.raises(ValueError, match="pinned official"):
        evaluate_twowiki(config, tmp_path / "bad", ["q1"], prediction, True)


def test_cli_loads_live_twowiki_and_exports_official_evidence(tmp_path):
    dataset = tmp_path / "dev.json"
    dataset.write_text(json.dumps([{"_id": "q1", "question": "Who?", "answer": "PRIVATE"}]))
    config = Config(
        dataset=DatasetConfig(kind="twowiki", id="Alab-NII/2wikimultihop", path=str(dataset)),
        retrieval=RetrievalConfig(backend="wikipedia_live"),
    )
    task = tasks_for(config, ["q1"], limit=0)[0]
    assert "evidences" in task.instruction
    assert "live English Wikipedia" in task.instruction
    assert "PRIVATE" not in task.instruction
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "tasks": [{"task_id": "q1"}, {"task_id": "missing"}]}
        )
    )
    attempt = tmp_path / "tasks/q1/attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "submission.json").write_text(
        json.dumps(
            {
                "task_id": "q1",
                "answer": "A",
                "sp": [["Page", 0]],
                "evidence": [["Film", "director", "A"]],
            }
        )
    )
    prediction = json.loads(export_run(tmp_path).read_text())
    assert prediction["evidence"] == {"q1": [["Film", "director", "A"]], "missing": []}
    assert prediction["answer"] == {"q1": "A", "missing": ""}
