"""Deterministic CPU conformer computation for the native tool-latency pilot."""

import json
import sys
import time

from rdkit import Chem, rdBase
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


def compute(arguments):
    molecule = Chem.MolFromSmiles(arguments["smiles"])
    if molecule is None:
        raise ValueError("Invalid SMILES")
    canonical = Chem.MolToSmiles(molecule)
    heavy_atoms = molecule.GetNumHeavyAtoms()
    molecule = Chem.AddHs(molecule)
    parameters = rdDistGeom.ETKDGv3()
    parameters.randomSeed = arguments["seed"]
    parameters.numThreads = 1
    parameters.pruneRmsThresh = -1.0
    started = time.monotonic()
    conformers = list(
        rdDistGeom.EmbedMultipleConfs(molecule, numConfs=arguments["num_confs"], params=parameters)
    )
    embed_s = time.monotonic() - started
    if not conformers:
        raise RuntimeError("No conformers embedded")
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("Molecule lacks MMFF parameters")
    started = time.monotonic()
    optimized = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
        molecule,
        numThreads=1,
        maxIters=arguments["max_iterations"],
        mmffVariant="MMFF94s",
    )
    return {
        "rdkit_version": rdBase.rdkitVersion,
        "canonical_smiles": canonical,
        "heavy_atoms": heavy_atoms,
        "seed": arguments["seed"],
        "num_threads": 1,
        "requested_conformers": arguments["num_confs"],
        "embedded_conformers": len(conformers),
        "converged_conformers": sum(status == 0 for status, _ in optimized),
        "minimum_energy_kcal_mol": min(energy for _, energy in optimized),
        "embed_s": embed_s,
        "optimize_s": time.monotonic() - started,
    }


if __name__ == "__main__":
    print(json.dumps(compute(json.load(sys.stdin))))
