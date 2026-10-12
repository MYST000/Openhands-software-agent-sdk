import json
import math
import sys

import pytest
from openhands.sdk.llm import TextContent

from benchmark_adapters.rdkit_probe import (
    ConformerAction,
    ConformerEnvironment,
    ConformerExecutor,
)
from benchmark_adapters.sdk_bridge import Binding
from benchmark_adapters.tracing import Budget, TraceRecorder


def test_rdkit_executor_resolves_fixed_molecule_and_records_errors(tmp_path):
    pytest.importorskip("rdkit")
    recorder = TraceRecorder(tmp_path, {"benchmark": "rdkit-conformer-pilot"})
    binding = Binding(
        ConformerEnvironment("hexadecane", sys.executable),
        recorder,
        Budget(8, 10, 1800),
        600,
        16000,
    )
    executor = ConformerExecutor(binding)
    try:
        observation = executor(ConformerAction(molecule_id="hexadecane"))
        assert not observation.is_error
        content = observation.content[0]
        assert isinstance(content, TextContent)
        data = json.loads(content.text)
        assert data["canonical_smiles"] == "C" * 16
        assert data["heavy_atoms"] == 16
        assert data["embedded_conformers"] == 128
        assert math.isfinite(data["minimum_energy_kcal_mol"])
        assert executor(ConformerAction(molecule_id="octacosane")).is_error
        assert executor(ConformerAction(molecule_id="hexadecane", seed=42)).is_error
    finally:
        recorder.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events].count("tool_end") == 1
    assert [e["event"] for e in events].count("tool_error") == 2
    end = next(e for e in events if e["event"] == "tool_end")
    assert end["effective_arguments"]["smiles"] == "C" * 16
    assert end["round_trip_ms"] >= (data["embed_s"] + data["optimize_s"]) * 1000


def test_rdkit_worker_preserves_seeded_computation_and_rejects_invalid_input():
    pytest.importorskip("rdkit")
    from benchmark_adapters.rdkit_conformers import compute

    arguments = {"smiles": "CCC", "num_confs": 2, "seed": 41, "max_iterations": 50}
    first, second = compute(arguments), compute(arguments)
    for field in ("embedded_conformers", "converged_conformers", "minimum_energy_kcal_mol"):
        assert first[field] == second[field]
    assert first["num_threads"] == 1
    with pytest.raises(ValueError, match="Invalid SMILES"):
        compute({**arguments, "smiles": "invalid"})
