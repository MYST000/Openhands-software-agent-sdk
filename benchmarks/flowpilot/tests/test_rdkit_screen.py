import gzip
import hashlib
import io
import json
import pickle
from types import SimpleNamespace

import pytest

from benchmark_adapters.rdkit_screen import (
    ScreenAction,
    ScreenEnvironment,
    ScreenExecutor,
    ScreenTargetTool,
)
from benchmark_adapters.sdk_bridge import Binding, bind, unbind
from benchmark_adapters.tracing import Budget, TraceRecorder


@pytest.fixture
def screening_data(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from benchmark_adapters.rdkit_screen_worker import input_paths

    case = dict(
        id="platform-MUV832-rdk7",
        dataset="MUV",
        target="832",
        fingerprint="rdk7",
        num_queries=5,
        num_splits=50,
    )
    actives = ["CC", "CCC", "CCCC", "CCO", "CCN", "CCCO", "CCCCN"]
    decoys = ["CCC", "CCCl", "CO", "CCBr", "CCF", "CCOC"]
    paths = input_paths(tmp_path, case)
    for path, smiles, prefix in zip(paths[:2], (actives, decoys), ("A", "D")):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as stream:
            stream.write("# compound table\n")
            for i, value in enumerate(smiles):
                stream.write(f"{i}\t{prefix}{i}\t{value}\n")
    paths[2].parent.mkdir(parents=True, exist_ok=True)
    with paths[2].open("wb") as stream:
        for _ in range(50):
            # Exclude five active queries and decoy 1 from the test candidates.
            pickle.dump([0, 1, 2, 3, 4, 1], stream, protocol=2)
    library = SimpleNamespace(
        CalculateFP=lambda _, smiles: Chem.RDKFingerprint(
            Chem.MolFromSmiles(smiles), maxPath=7, fpSize=1024, nBitsPerHash=2
        )
    )
    return case, library, actives + decoys


def test_screening_preserves_splits_dice_scores_and_both_counts(tmp_path, screening_data):
    import numpy as np
    from rdkit import DataStructs

    from benchmark_adapters.rdkit_screen_worker import execute

    case, library, smiles = screening_data
    result = execute(case, tmp_path, tmp_path / "output", library)
    payload = result["payload"]
    assert payload["num_splits"] == payload["original_algorithm_sample_matches"] == 50
    assert payload["test_candidates_per_split"] == 7
    assert payload["test_candidate_occurrences_all_splits"] == 350
    from pathlib import Path

    path = Path(payload["artifact_scores"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == payload["artifact_sha256"]
    fingerprints = [library.CalculateFP("rdk7", value) for value in smiles]
    indices = [5, 6, 7, 9, 10, 11, 12]
    reference = [
        max(DataStructs.DiceSimilarity(fingerprints[i], fingerprints[q]) for q in range(5))
        for i in indices
    ]
    with np.load(path, allow_pickle=False) as artifact:
        assert artifact["scores"].shape == (50, 7)
        np.testing.assert_array_equal(artifact["indices"], [indices] * 50)
        np.testing.assert_allclose(artifact["scores"], [reference] * 50, rtol=0, atol=1e-15)
    assert payload["first_split_top5"][0] == {"id": "D0", "similarity": 1.0}


def test_screening_rejects_incomplete_splits_and_changed_parameters(tmp_path, screening_data):
    from benchmark_adapters.rdkit_screen_worker import execute, input_paths

    case, library, _ = screening_data
    with pytest.raises(ValueError, match="5 queries and 50 splits"):
        execute({**case, "num_splits": 1}, tmp_path, tmp_path / "output", library)
    input_paths(tmp_path, case)[2].write_bytes(pickle.dumps([0, 1, 2, 3, 4]))
    with pytest.raises(EOFError):
        execute(case, tmp_path, tmp_path / "output", library)


def test_query_pickle_cannot_load_globals():
    pytest.importorskip("rdkit")
    from benchmark_adapters.rdkit_screen_worker import SplitUnpickler

    with pytest.raises(ValueError, match="primitive values"):
        SplitUnpickler(io.BytesIO(b"cbuiltins\neval\n.")).load()


@pytest.mark.parametrize(
    "error",
    [ValueError("wrong parameters"), RuntimeError("worker failure"), TimeoutError("deadline")],
)
def test_executor_marks_errors_and_never_emits_success(tmp_path, error):
    def screen(arguments, timeout):
        raise error

    recorder = TraceRecorder(tmp_path, {"benchmark": "rdkit-screen"})
    binding = Binding(SimpleNamespace(screen=screen), recorder, Budget(8, 10, 600), 600, 16000)
    key = bind(binding)
    try:
        tool = ScreenTargetTool.create(binding_key=key)[0]
        assert tool.name == "screen_target"
        observation = tool.executor(ScreenAction(target="832", fingerprint="rdk7"))
        assert observation.is_error
    finally:
        recorder.close()
        unbind(key)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert not any(e["event"] == "tool_end" for e in events)
    failure = next(e for e in events if e["event"] == "tool_error")
    assert failure["timed_out"] == isinstance(error, TimeoutError)


def test_environment_refuses_changed_task_before_worker_call(tmp_path):
    environment = ScreenEnvironment(
        tmp_path, tmp_path, dataset="MUV", target="832", fingerprint="rdk7", python="unused"
    )
    with pytest.raises(ValueError, match="exactly the requested parameters"):
        environment.screen({**environment.arguments, "target": "600"}, 10)
    with pytest.raises(RuntimeError, match="not running"):
        environment.screen(environment.arguments, 10)
    environment.close()


def test_executor_records_success_and_effective_arguments(tmp_path):
    payload = {"test_candidates_per_split": 7, "test_candidate_occurrences_all_splits": 350}
    recorder = TraceRecorder(tmp_path, {"benchmark": "rdkit-screen"})
    env = SimpleNamespace(screen=lambda arguments, timeout: {"payload": payload, "stages": {}})
    try:
        observation = ScreenExecutor(Binding(env, recorder, Budget(8, 10, 600), 600, 16000))(
            ScreenAction(target="832", fingerprint="rdk7")
        )
        assert not observation.is_error
        assert json.loads(observation.text) == payload
    finally:
        recorder.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event = next(e for e in events if e["event"] == "tool_end")
    assert event["effective_arguments"] == dict(
        target="832", fingerprint="rdk7", num_queries=5, num_splits=50
    )
