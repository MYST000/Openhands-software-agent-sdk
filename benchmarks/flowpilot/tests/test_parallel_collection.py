"""Frozen queue, split provenance and code-isolation gates without actor execution."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from benchmark_adapters import native_browsecomp
from benchmark_adapters import parallel_collection as collection
from benchmark_adapters.config import Config, DatasetConfig
from benchmark_adapters.contracts import Task


@pytest.fixture
def campaign_files(tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "sdk_source_state", lambda *_args: {})
    monkeypatch.setattr(collection, "adapter_source_digest", lambda: "adapter-fixture-v1")
    dataset = tmp_path / "questions.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "_id": f"q{index}",
                    "question": f"Public fixture question {index}?",
                    "answer": f"PRIVATE_ANSWER_{index}",
                    "supporting_facts": [["PRIVATE_GOLD_TITLE", index]],
                }
                for index in range(3)
            ]
        )
    )
    profile = tmp_path / "hotpot.toml"
    profile.write_text(
        '[dataset]\nkind="hotpot"\nid="hotpotqa"\npath="questions.json"\n'
        'revision="fixture-revision"\nsplit="dev"\n'
        'setting="fullwiki-fixed-corpus-v1"\n'
    )
    path = tmp_path / "campaign.json"
    value = {
        "version": 1,
        "run_id": "fixture",
        "runs_dir": "runs",
        "concurrency": 4,
        "queue_policy": "ordered",
        "entries": [{"config": "hotpot.toml", "task_ids": ["q2", "q0"]}],
    }
    path.write_text(json.dumps(value))
    return path, value, dataset


def test_prepare_freezes_public_tasks_queue_and_rejects_input_tampering(campaign_files):
    path, _, dataset = campaign_files
    root = collection.prepare_campaign(path)
    _, manifest = collection._checked_manifest(root)
    jobs = manifest["jobs"]
    payloads = [collection._payload(job) for job in jobs]
    assert [payload["task"]["task_id"] for payload in payloads] == ["q2", "q0"]
    assert [job["trace_context"]["queue_position"] for job in jobs] == [0, 1]
    assert manifest["concurrency"] == 4
    assert set(manifest["adapter_concurrency"].values()) == {4}
    assert manifest["intra_task_tool_concurrency"] == 1
    assert manifest["evaluators_run_after_collection"] is True
    assert all(payload["bundle"] is None for payload in payloads)
    assert "PRIVATE_" not in json.dumps(payloads)
    assert root.stat().st_mode & 0o777 == 0o700
    assert all(collection.Path(job["input_path"]).stat().st_mode & 0o777 == 0o600 for job in jobs)
    dataset.write_text("[]\n")
    assert collection._payload(jobs[0])["task"] == payloads[0]["task"]
    collection.Path(jobs[0]["input_path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="Frozen task input changed"):
        collection._checked_manifest(root)


def test_prepared_campaign_rejects_adapter_drift_and_existing_run(campaign_files, monkeypatch):
    path, _, _ = campaign_files
    root = collection.prepare_campaign(path)
    original = (root / "campaign.json").read_bytes()
    with pytest.raises(FileExistsError):
        collection.prepare_campaign(path)
    assert (root / "campaign.json").read_bytes() == original
    monkeypatch.setattr(collection, "adapter_source_digest", lambda: "changed-adapter")
    with pytest.raises(ValueError, match="Adapter changed"):
        collection._checked_manifest(root)


@pytest.mark.parametrize("case", ["empty_queue", "empty_ids", "duplicate_ids", "duplicate_entry"])
def test_prepare_rejects_empty_or_repeated_selection(campaign_files, case):
    path, campaign, _ = campaign_files
    if case == "empty_queue":
        campaign["entries"] = []
    elif case == "empty_ids":
        campaign["entries"][0]["task_ids"] = []
    elif case == "duplicate_ids":
        campaign["entries"][0]["task_ids"] = ["q0", "q0"]
    else:
        campaign["entries"].append(dict(campaign["entries"][0]))
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="nonempty|unique|Repeated task"):
        collection.prepare_campaign(path)
    assert not (path.parent / "runs/fixture/campaign.json").exists()


@pytest.fixture
def research_inputs(tmp_path):
    task = Task("hotpotqa", "fixture-revision", "dev", "q0", "Public question?")
    config = Config(
        dataset=DatasetConfig(kind="hotpot", id=task.dataset_id, revision=task.revision)
    )
    row = {
        "dataset_id": task.dataset_id,
        "dataset_revision": task.revision,
        "task_id": task.task_id,
        "research_split": "fit",
        "historically_exposed": False,
        "task_group_id": "fixture-group-0",
        "instruction_sha256": hashlib.sha256(task.instruction.encode()).hexdigest(),
    }
    return tmp_path, task, config, row


def manifest_entry(root, rows):
    path = root / "splits.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "research_split": "fit",
        "split_manifest": path.name,
        "split_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_research_manifest_cannot_inject_gold_or_rewrite_actor_task(research_inputs):
    root, task, config, row = research_inputs
    entry = manifest_entry(
        root,
        [
            {
                **row,
                "answer": "INJECTED_PRIVATE_ANSWER",
                "instruction": "INJECTED_INSTRUCTION",
                "public_metadata": {"solution_path": "INJECTED_PATH"},
            }
        ],
    )
    pairs = collection._research_tasks([(task, None)], entry, config, root)
    assigned, bundle = pairs[0]
    assert bundle is None
    assert assigned.split == assigned.public_metadata["research_split"] == "fit"
    assert assigned.public_metadata["task_group_id"] == row["task_group_id"]
    assert assigned.instruction == task.instruction
    assert task.split == "dev" and task.public_metadata == {}
    assert "INJECTED" not in json.dumps(assigned.to_dict())


@pytest.mark.parametrize("conflict", ["group", "identical_text", "historical_exposure"])
def test_research_manifest_rejects_cross_split_leakage_in_unselected_rows(
    research_inputs, conflict
):
    root, task, config, row = research_inputs
    other = {
        **row,
        "task_id": "unselected",
        "research_split": "test",
        "task_group_id": "fixture-group-1",
        "instruction_sha256": hashlib.sha256(b"Another question").hexdigest(),
    }
    if conflict == "group":
        other["task_group_id"] = row["task_group_id"]
    elif conflict == "identical_text":
        other["instruction_sha256"] = row["instruction_sha256"]
    else:
        other["historically_exposed"] = True
    entry = manifest_entry(root, [row, other])
    with pytest.raises(ValueError, match="crosses research splits|Exposed task"):
        collection._research_tasks([(task, None)], entry, config, root)


def test_formal_split_requires_unchanged_manifest_and_preserves_quixbugs_auxiliary(research_inputs):
    root, task, config, row = research_inputs
    with pytest.raises(ValueError, match="frozen, hash-checked"):
        collection._research_tasks([(task, None)], {"research_split": "fit"}, config, root)
    entry = manifest_entry(root, [row])
    (root / entry["split_manifest"]).write_text("{}\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        collection._research_tasks([(task, None)], entry, config, root)
    quixbugs = replace(config, dataset=replace(config.dataset, kind="quixbugs"))
    with pytest.raises(ValueError, match="historical development"):
        collection._research_tasks([(task, None)], entry, quixbugs, root)


@pytest.mark.parametrize("entrypoint", [collection.validate_campaign, collection.run_campaign])
def test_nonroot_code_gate_rejects_before_any_actor_or_validation_work(
    tmp_path, monkeypatch, entrypoint
):
    manifest = {"jobs": [{"adapter": "quixbugs", "uid_base": 63200}], "concurrency": 4}
    monkeypatch.setattr(collection, "_checked_manifest", lambda _root: (tmp_path, manifest))
    monkeypatch.setattr(collection.os, "geteuid", lambda: 1015)
    called = []
    monkeypatch.setattr(
        collection, "run_parallel", lambda *_args, **_kwargs: called.append("actor")
    )
    monkeypatch.setattr(collection, "check_sdk", lambda *_args: called.append("sdk"))
    with pytest.raises(ValueError, match="requires root.*Same-UID shell fallback"):
        entrypoint(tmp_path)
    assert not called
    assert not list(tmp_path.iterdir())
    collection._isolation_preflight(tmp_path, {"jobs": [{"adapter": "hotpot"}]})


def test_four_workers_have_distinct_actor_and_evaluator_uids():
    identities = [uid for slot in range(4) for uid in collection.worker_uids(63200, slot)]
    assert len(set(identities)) == 8
    with pytest.raises(ValueError, match="dedicated UIDs"):
        collection.worker_uids(64999, 0)


def test_campaign_passes_each_validated_index_to_its_worker(campaign_files, monkeypatch):
    path, campaign, _ = campaign_files
    campaign["entries"].append({"config": "hotpot.toml", "task_ids": ["q1"]})
    path.write_text(json.dumps(campaign))
    root = collection.prepare_campaign(path)
    _, manifest = collection._checked_manifest(root)
    indexes = [{"index_sha256": "fixture-index-a"}, {"index_sha256": "fixture-index-b"}]
    collection.write_json(
        root / "validation.json",
        {
            "all_checks_passed": True,
            "campaign_sha256": collection.sha(root / "campaign.json"),
            "checks": [
                {"entry": entry, "ok": True, "metadata": metadata}
                for entry, metadata in enumerate(indexes)
            ],
        },
    )
    dispatched = []

    def capture_jobs(jobs, worker, **kwargs):
        assert worker is collection.collect_job
        dispatched.extend(jobs)
        return {"completed": len(jobs)}

    monkeypatch.setattr(collection, "check_sdk", lambda _runtime: None)
    monkeypatch.setattr(collection, "run_parallel", capture_jobs)
    monkeypatch.setattr(collection, "write_prediction_dataset", lambda *_args: None)
    assert collection.run_campaign(root)["status"] == "collected"
    assert len(dispatched) == 3
    for job in dispatched:
        entry = collection._payload(job)["entry_index"]
        assert job["verified_retrieval_metadata"] == indexes[entry]
    assert all("verified_retrieval_metadata" not in job for job in manifest["jobs"])


def test_collect_job_passes_verified_index_to_factory_and_closes_environment(
    campaign_files, monkeypatch
):
    path, _, _ = campaign_files
    root = collection.prepare_campaign(path)
    _, manifest = collection._checked_manifest(root)
    verified = {"index_sha256": "fixture-verified-index", "index_size_bytes": 123}
    job = {**manifest["jobs"][0], "verified_retrieval_metadata": verified}
    attempt = collection.Path(job["attempt_dir"])
    attempt.mkdir(parents=True)
    phases, closed, factories = [], [], []
    environment = SimpleNamespace(close=lambda: closed.append(True))

    def factory(config, verified_index=None):
        factories.append((config.dataset.kind, verified_index))
        return environment

    def run_task(config, task, output, **kwargs):
        assert output == attempt
        assert kwargs["environment"] is environment
        assert kwargs["trace_context"]["worker_slot"] == 2
        return {"status": "fixture-completed"}

    monkeypatch.setattr(native_browsecomp, "create_retrieval_environment", factory)
    monkeypatch.setattr(collection, "run_task", run_task)
    monkeypatch.setattr(collection, "build_prediction_rows", lambda _attempt: [])
    context = SimpleNamespace(slot=2, set_phase=phases.append)
    assert collection.collect_job(job, context) == {"status": "fixture-completed"}
    assert factories == [("hotpot", verified)]
    assert factories[0][1] is verified
    assert phases == ["actor"]
    assert closed == [True]
