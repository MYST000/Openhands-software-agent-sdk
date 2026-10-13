"""Isolated RDKit screening worker with the historical Dice/max numerical algorithm."""

import argparse
import contextlib
import gzip
import hashlib
import importlib.util
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
from rdkit import DataStructs, RDLogger, rdBase

SOURCE_REVISION = "f37fc620cd9d74e3ccb89b778d54770b6c6080d7"
FINGERPRINT_SHA256 = "b89cac0ad7133437b06147bc04eee4ad612dc43d40118bd6c7c56bf163ddcc4b"


class SplitUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Official query splits must contain only primitive values")


def validate_case(case):
    if (
        case["dataset"] not in {"MUV", "DUD", "ChEMBL"}
        or not re.fullmatch(r"[A-Za-z0-9_-]+", case["target"])
        or case["fingerprint"] not in {"rdk7", "ecfp4"}
        or type(case["num_queries"]) is not int
        or case["num_queries"] != 5
        or type(case["num_splits"]) is not int
        or case["num_splits"] != 50
    ):
        raise ValueError("Screening requires a target, rdk7/ecfp4, 5 queries and 50 splits")


def input_paths(root, case):
    ds, target = case["dataset"], case["target"]
    decoy = "zinc" if ds == "ChEMBL" else target
    return [
        root / f"compounds/{ds}/cmp_list_{ds}_{target}_actives.dat.gz",
        root / f"compounds/{ds}/cmp_list_{ds}_{decoy}_decoys.dat.gz",
        root / f"query_lists/data_sets_I/{ds}/training_{ds}_{target}_5.pkl",
        root / "scoring/fingerprint_lib.py",
    ]


def runtime_identity():
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "rdkit_build": rdBase.rdkitBuild,
        "numpy_version": np.__version__,
    }


def load_compounds(path):
    with gzip.open(path, "rt") as f:
        return [line.split() for line in f if line.strip() and not line.startswith("#")]


def execute(c, root, output, fingerprint_lib):
    validate_case(c)
    start = time.monotonic()
    ds, target = c["dataset"], c["target"]
    act = load_compounds(root / f"compounds/{ds}/cmp_list_{ds}_{target}_actives.dat.gz")
    decoy_name = (
        f"cmp_list_{ds}_zinc_decoys.dat.gz"
        if ds == "ChEMBL"
        else f"cmp_list_{ds}_{target}_decoys.dat.gz"
    )
    dec = load_compounds(root / "compounds" / ds / decoy_name)
    rows = act + dec
    fps = [fingerprint_lib.CalculateFP(c["fingerprint"], row[2]) for row in rows]
    if any(fp is None for fp in fps):
        raise ValueError("Fingerprint generation failed; molecules are not silently dropped")
    stages = {"fingerprint_s": time.monotonic() - start}
    splits = []
    with (root / f"query_lists/data_sets_I/{ds}/training_{ds}_{target}_5.pkl").open("rb") as f:
        for _ in range(c["num_splits"]):
            splits.append(SplitUnpickler(f, encoding="latin1").load())
        if f.read(1):
            raise ValueError("Expected exactly 50 published query splits")
    all_scores, all_indices, checks = [], [], []
    rank_start = time.monotonic()
    for training in splits:
        if (
            not isinstance(training, list)
            or len(training) < 5
            or any(type(i) is not int for i in training)
            or len(set(training[:5])) != 5
            or any(not 0 <= i < len(act) for i in training[:5])
            or any(not 0 <= i < len(dec) for i in training[5:])
        ):
            raise ValueError("Invalid published training indices")
        queries = training[: c["num_queries"]]
        train_act, train_dec = set(queries), set(training[c["num_queries"] :])
        indices = [i for i in range(len(act)) if i not in train_act] + [
            len(act) + i for i in range(len(dec)) if i not in train_dec
        ]
        values = np.max(
            np.array([DataStructs.BulkDiceSimilarity(fps[q], fps) for q in queries]),
            axis=0,
        )[indices]
        all_scores.append(values)
        all_indices.append(np.array(indices, dtype=np.int32))
        sample = indices[:20] + indices[-20:]
        reference = [
            max(DataStructs.BulkDiceSimilarity(fps[i], [fps[q] for q in queries])) for i in sample
        ]
        actual = np.max(
            np.array(
                [DataStructs.BulkDiceSimilarity(fps[q], [fps[i] for i in sample]) for q in queries]
            ),
            axis=0,
        )
        checks.append(bool(np.allclose(reference, actual, rtol=0, atol=1e-15)))
    stages["rank_s"] = time.monotonic() - rank_start
    if not all(checks):
        raise AssertionError("Original per-candidate algorithm comparison failed")
    output.mkdir(parents=True, exist_ok=True)
    path = output / (c["id"] + "-" + str(time.monotonic_ns()) + ".npz")
    np.savez_compressed(
        path,
        scores=np.array(all_scores),
        indices=np.array(all_indices),
        labels=np.array([1] * len(act) + [0] * len(dec)),
        ids=np.array([v[1] for v in rows]),
    )
    first = sorted(
        [
            (float(v), rows[int(i)][1], int(int(i) < len(act)))
            for i, v in zip(all_indices[0], all_scores[0])
        ],
        reverse=True,
    )
    payload = {
        "target": target,
        "dataset": ds,
        "fingerprint": c["fingerprint"],
        "similarity": "Dice",
        "num_queries": c["num_queries"],
        "num_splits": len(splits),
        "actives": len(act),
        "decoys": len(dec),
        "protocol": "rdkit-screen-counts-v2",
        "test_candidates_per_split": len(first),
        "test_candidate_occurrences_all_splits": sum(len(v) for v in all_indices),
        "first_split_top5": [{"id": v[1], "similarity": v[0]} for v in first[:5]],
        "original_algorithm_sample_matches": sum(checks),
        "sampled_comparisons": sum(min(20, len(v)) * 2 for v in all_indices),
        "artifact_scores": str(path),
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    stages["worker_elapsed_s"] = time.monotonic() - start
    return {"ok": True, "case_id": c["id"], "payload": payload, "stages": stages}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("MUV", "DUD", "ChEMBL"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--fingerprint", choices=("rdk7", "ecfp4"), default="rdk7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = dict(
        id=f"platform-{args.dataset}{args.target}-{args.fingerprint}",
        dataset=args.dataset,
        target=args.target,
        fingerprint=args.fingerprint,
        num_queries=5,
        num_splits=50,
    )
    validate_case(case)
    if rdBase.rdkitVersion != "2025.09.6":
        raise ValueError("Install the pinned rdkit==2025.9.6 for this screening protocol")
    root = args.source_root.resolve()
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in input_paths(root, case)
    }
    if hashes["scoring/fingerprint_lib.py"] != FINGERPRINT_SHA256:
        raise ValueError(f"Use official fingerprint_lib.py from {SOURCE_REVISION}")
    spec = importlib.util.spec_from_file_location(
        "rdkit_official_fingerprint", root / "scoring/fingerprint_lib.py"
    )
    fingerprint_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fingerprint_lib)
    RDLogger.DisableLog("rdApp.warning")
    print(
        json.dumps(
            {
                "ready": True,
                "source_revision": SOURCE_REVISION,
                "input_sha256": hashes,
                **runtime_identity(),
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "stop":
                break
            if request != {"case_id": case["id"]}:
                raise ValueError("Worker accepts only its fixed task")
            with contextlib.redirect_stdout(sys.stderr):
                result = execute(case, root, args.output.resolve(), fingerprint_lib)
        except Exception as exc:
            result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
