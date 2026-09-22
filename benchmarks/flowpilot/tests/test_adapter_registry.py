import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmark_adapters.adapter_registry import ADAPTERS, load_adapter_tasks, resolve_adapter
from benchmark_adapters.config import Config, DatasetConfig


def config_for(kind, **updates):
    registration = ADAPTERS[kind]
    return Config(
        dataset=DatasetConfig(
            kind=kind,
            id=registration.dataset_id,
            setting=registration.setting,
            revision="fixture-revision",
            split="dev",
            **updates,
        )
    )


@pytest.mark.parametrize(
    "kind,expected_tools,code",
    [
        ("livecodebench", ("code_terminal", "code_file_editor"), True),
        ("quixbugs", ("code_terminal", "code_file_editor"), True),
        ("hotpot", ("search", "read_document"), False),
        ("browsecomp", ("search", "get_document"), False),
    ],
)
def test_declared_identity_selects_only_its_environment_capabilities(kind, expected_tools, code):
    registration = resolve_adapter(config_for(kind))
    assert registration.environment_tools == expected_tools
    assert registration.code_workspace is code
    assert len(registration.environment_tools) == 2


def test_unknown_adapter_and_mismatched_identity_are_rejected():
    config = config_for("hotpot")
    for updates in (
        {"kind": "unknown"},
        {"id": ADAPTERS["browsecomp"].dataset_id},
        {"setting": ADAPTERS["livecodebench"].setting},
        {"revision": ""},
        {"revision": "   "},
    ):
        with pytest.raises(ValueError):
            resolve_adapter(replace(config, dataset=replace(config.dataset, **updates)))


@pytest.mark.parametrize("kind", ["hotpot", "browsecomp"])
@pytest.mark.parametrize("ids", [[], None, ["x", "x"]])
def test_registry_requires_explicit_nonempty_unique_selection(tmp_path, kind, ids):
    source = tmp_path / "questions.jsonl"
    source.write_text(
        json.dumps({"_id": "x", "question": "Question?", "query_id": "x", "query": "Question?"})
        + "\n"
    )
    with pytest.raises(ValueError):
        load_adapter_tasks(config_for(kind, path=str(source)), ids)


@pytest.mark.parametrize("kind", ["hotpot", "browsecomp"])
def test_retrieval_routing_preserves_order_and_hides_private_annotations(tmp_path, kind):
    source = tmp_path / "questions.jsonl"
    rows = [
        {
            "_id": str(i),
            "question": f"Question {i}?",
            "query_id": str(i),
            "query": f"Query {i}?",
            "answer": "PRIVATE_SENTINEL",
            "supporting_facts": [["PRIVATE_SENTINEL", 0]],
            "context": "PRIVATE_SENTINEL",
            "gold_doc_ids": ["PRIVATE_SENTINEL"],
        }
        for i in range(2)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = config_for(kind, path=str(source))
    selected, provenance = load_adapter_tasks(config, ["1", "0"])
    assert [task.task_id for task, _ in selected] == ["1", "0"]
    assert all(bundle is None for _, bundle in selected)
    assert all(task.dataset_id == config.dataset.id for task, _ in selected)
    assert "PRIVATE_SENTINEL" not in json.dumps([task.to_dict() for task, _ in selected])
    assert provenance["revision"] == "fixture-revision"
    with pytest.raises(ValueError, match="missing"):
        load_adapter_tasks(config, ["not-present"])


@pytest.mark.parametrize("kind", ["livecodebench", "quixbugs"])
def test_code_routing_delegates_identity_checks_to_code_loader(monkeypatch, kind):
    from benchmark_adapters import code_collection

    called = []
    task = object()
    bundle = SimpleNamespace(task=task)

    def load_selected(config, ids, checker, *, check_isolation):
        called.append((config, ids, checker, check_isolation))
        return [bundle], {"verified_source": "fixture"}

    monkeypatch.setattr(code_collection, "load_selected", load_selected)
    config = config_for(kind)
    selected, provenance = load_adapter_tasks(config, ["q"], checker_path="checker.py")
    assert selected == [(task, bundle)]
    assert called == [(config, ["q"], "checker.py", False)]
    assert provenance == {"verified_source": "fixture"}
