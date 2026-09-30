"""Export the same retrieval contracts used by the benchmark FlowPilot runtime."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openhands.sdk.flowpilot_reuse import payload_digest
from openhands.sdk.tool import Action

from .config import Config, load_config
from .retrieval_tools import (
    BrowseReadAction,
    HotpotReadAction,
    NativeGetDocumentAction,
    NativeSearchAction,
    SearchAction,
)


def retrieval_actions(config: Config) -> dict[str, type[Action]]:
    kind, backend = config.dataset.kind, config.retrieval.backend
    if kind == "hotpot" and backend in {"sqlite", "hotpot_rpc"}:
        return {"search": SearchAction, "read_document": HotpotReadAction}
    if kind == "browsecomp" and backend == "sqlite":
        return {"search": SearchAction, "get_document": BrowseReadAction}
    if kind == "browsecomp" and backend == "browsecomp_mcp":
        return {"search": NativeSearchAction, "get_document": NativeGetDocumentAction}
    raise ValueError(f"Unsupported retrieval reuse profile: {kind}/{backend}")


def retrieval_scope(config: Config) -> tuple[str, ...]:
    if not config.retrieval.corpus_revision:
        raise ValueError("Retrieval reuse requires an immutable corpus_revision")
    if config.retrieval.backend != "sqlite" and not config.retrieval.server_policy_revision:
        raise ValueError("Remote retrieval reuse requires server_policy_revision")
    profile = {
        "benchmark": config.dataset.kind,
        "retrieval": asdict(config.retrieval),
        "observation": "retrieval-observation-v1",
        "actions": {
            name: action.model_json_schema() for name, action in retrieval_actions(config).items()
        },
    }
    return (f"benchmark:{config.dataset.kind}", f"benchmark-retrieval:{payload_digest(profile)}")


def export_registry(config: Config, *, semantic_mode: str = "shadow") -> list[dict[str, Any]]:
    if semantic_mode not in {"shadow", "candidate", "active"}:
        raise ValueError("Invalid semantic_mode")
    scope = retrieval_scope(config)
    native = config.retrieval.backend == "browsecomp_mcp"
    return [
        {
            "protocol_version": "flowpilot-phase3-reuse-v3",
            "tool_name": name,
            "canonical_tool_family": f"{config.dataset.kind}_{name}",
            "tool_version": "1",
            "adapter_id": f"benchmark_{'native_' if native else ''}{name}_v1",
            "adapter_version": "1",
            "input_schema_digest": payload_digest(action.to_mcp_schema()),
            "result_schema_version": "retrieval-observation-v1",
            "policy_digest": scope[1].split(":", 1)[1],
            "required_data_source_constraints": list(scope),
            "read_only": True,
            "exact_reuse_enabled": True,
            "semantic_reuse_enabled": name == "search",
            "semantic_query_fields": ["query"],
            "semantic_mode": semantic_mode,
        }
        for name, action in retrieval_actions(config).items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-mode", choices=("shadow", "candidate", "active"), default="shadow"
    )
    args = parser.parse_args()
    entries = {}
    for path in args.config:
        for entry in export_registry(load_config(path), semantic_mode=args.semantic_mode):
            entries[(entry["tool_name"], entry["policy_digest"])] = entry
    args.output.write_text(json.dumps(list(entries.values()), indent=2) + "\n")


if __name__ == "__main__":
    main()
