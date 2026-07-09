#!/usr/bin/env python
"""End-to-end smoke test: encode a handful of real molecules, decode them back,
and report exact canonical-SMILES reconstruction matches.

Run in the ``jax-cddd`` environment:
    micromamba run -n jax-cddd python scripts/reconstruct_smoke_test.py
"""
from __future__ import annotations

from rdkit import Chem

from jax_cddd.inference import CDDDModel

MOLECULES = {
    "benzene": "c1ccccc1",
    "ethanol": "CCO",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "pyridine": "c1ccncc1",
    "naphthalene": "c1ccc2ccccc2c1",
}


def canonical(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def main():
    model = CDDDModel()
    names = list(MOLECULES)
    smiles_list = [MOLECULES[n] for n in names]

    embeddings = model.seq_to_emb(smiles_list)
    reconstructed = model.emb_to_seq(embeddings, beam_width=10)

    matches = 0
    for name, original, recon in zip(names, smiles_list, reconstructed):
        ok = canonical(original) == canonical(recon)
        matches += int(ok)
        status = "OK" if ok else "MISMATCH"
        print(f"{name:15s} {original:40s} -> {recon:40s} {status}")

    print(f"\n{matches}/{len(names)} exact canonical reconstructions")


if __name__ == "__main__":
    main()
