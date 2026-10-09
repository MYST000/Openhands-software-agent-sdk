"""Real-model stock/JITServe smoke using the unchanged Hotpot adapter and scorer.

The tiny synthetic corpus checks integration, not public-benchmark accuracy.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import httpx

from benchmark_adapters.config import (
    JITSERVE_SDK_COMMIT,
    Config,
    DatasetConfig,
    JITServeConfig,
    LLMConfig,
    RetrievalConfig,
    RuntimeConfig,
)
from benchmark_adapters.retrieval import HotpotAdapter, build_index
from benchmark_adapters.retrieval_evaluation import evaluator_source
from benchmark_adapters.runner import run_task
from benchmark_adapters.sdk_provenance import check_sdk
from benchmark_adapters.tracing import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["stock-v1", "jitserve-v1-port"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"
    corpus = args.output / "corpus.jsonl"
    docs = [
        {"docid": "1", "title": "Mira Vale", "sentences": ["Mira Vale supervises Lumen Station."]},
        {
            "docid": "2",
            "title": "Lumen Station",
            "sentences": ["Lumen Station uses access code cedar-47."],
        },
    ]
    corpus.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    index = args.output / "index.sqlite3"
    build_index(corpus, index, "jitserve-controlled-corpus-v1")
    row = {
        "_id": "jitserve-controlled-two-hop",
        "question": "What access code does the station supervised by Mira Vale use?",
        "answer": "cedar-47",
        "supporting_facts": [["Mira Vale", 0], ["Lumen Station", 0]],
    }
    dataset = args.output / "questions.json"
    write_json(dataset, [row])
    config = Config(
        dataset=DatasetConfig(
            kind="hotpot",
            id="hotpotqa",
            path=str(dataset.resolve()),
            revision="jitserve-controlled-task-v1",
            setting="fullwiki-fixed-corpus-v1",
        ),
        llm=LLMConfig(
            model="openai/Qwen3.5-27B-jitserve",
            base_url=args.base_url,
            enable_thinking=False,
            seed=42,
            max_input_tokens=16384,
            max_output_tokens=512,
            timeout=180,
            num_retries=0,
        ),
        runtime=replace(
            RuntimeConfig(),
            sdk_commit=JITSERVE_SDK_COMMIT,
            max_iterations=12,
            task_timeout=240,
        ),
        retrieval=RetrievalConfig(
            index_path=str(index.resolve()), corpus_revision="jitserve-controlled-corpus-v1"
        ),
        jitserve=JITServeConfig(
            enabled=True, backend=args.backend, output_len=128, workflow_budget_seconds=120
        ),
    )
    task = HotpotAdapter.from_record(
        row, dataset_id=config.dataset.id, revision=config.dataset.revision, split="dev"
    )
    write_json(args.output / "config.json", config.to_dict())
    write_json(args.output / "sdk-provenance.json", check_sdk(config.runtime))
    summaries = []
    with httpx.Client(
        base_url=args.base_url.removesuffix("/v1"), timeout=30, trust_env=False
    ) as client:
        client.get("/health").raise_for_status()
        for concurrency in args.concurrency:
            phase = args.output / f"concurrency-{concurrency}"
            phase.mkdir()
            before = client.get("/metrics")
            before.raise_for_status()
            (phase / "metrics-before.txt").write_text(before.text)
            started = time.monotonic()
            for iteration in range(args.rounds):

                def run(slot):
                    attempt = phase / f"round-{iteration}-slot-{slot}"
                    result = run_task(
                        config, task, attempt, run_id="jitserve-controlled-comparison"
                    )
                    events = [
                        json.loads(line)
                        for line in (attempt / "events.jsonl").read_text().splitlines()
                    ]
                    identities = [
                        e["jitserve"] for e in events if e["event"] == "jitserve_request_identity"
                    ]
                    assert result["execution_status"] == "completed", result
                    assert result["llm_requests"] >= 2 and result["tool_calls"] >= 1, result
                    assert len(identities) == result["llm_requests"]
                    assert {e["deadline"] for e in identities} == {result["slo"]["deadline"]}
                    assert {e["job_id"] for e in identities} == {result["slo"]["job_id"]}
                    assert any(e["event"] == "tool_end" for e in events)
                    submission = json.loads((attempt / "submission.json").read_text())
                    prediction = {
                        "answer": {task.task_id: submission["answer"]},
                        "sp": {task.task_id: submission["sp"]},
                    }
                    write_json(attempt / "prediction.json", prediction)
                    scorer, scorer_identity = evaluator_source(config)
                    with (attempt / "scorer.log").open("w") as log:
                        subprocess.run(
                            [
                                sys.executable,
                                str(scorer),
                                str(attempt / "prediction.json"),
                                str(dataset),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    from benchmark_adapters.retrieval_evaluation import hotpot_metrics

                    score = hotpot_metrics(attempt / "scorer.log")
                    assert score["joint_em"] == 1.0, score
                    summary = {
                        "attempt": str(attempt),
                        "result": result,
                        "score": score,
                        "scorer": scorer_identity,
                    }
                    write_json(attempt / "validation.json", summary)
                    return summary

                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    batch = list(pool.map(run, range(concurrency)))
                assert len({s["result"]["slo"]["job_id"] for s in batch}) == concurrency
                summaries.extend(batch)
                write_json(args.output / "progress.json", summaries)
            after = client.get("/metrics")
            after.raise_for_status()
            (phase / "metrics-after.txt").write_text(after.text)
            write_json(
                phase / "timing.json",
                {
                    "wall_seconds": time.monotonic() - started,
                    "workflows": concurrency * args.rounds,
                },
            )
    write_json(
        args.output / "summary.json",
        {
            "passed": True,
            "evidence": "real-model-controlled-corpus",
            "backend": args.backend,
            "workflows": summaries,
        },
    )


if __name__ == "__main__":
    main()
