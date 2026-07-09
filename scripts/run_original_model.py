#!/usr/bin/env python
"""Run the *actual, unmodified* original CDDD implementation (TF1, with
``tf.contrib``) end-to-end on a handful of molecules, and save its embeddings +
reconstructed SMILES as a fixture for ``tests/test_against_original_code.py``.

This is a genuinely independent, end-to-end check: unlike
``scripts/validate_against_tf1_reference.py`` (which recomputes the math
from scratch in NumPy), this script imports and executes
``src/jax_cddd/_cddd_ref_/cddd-master/cddd`` directly -- the pristine upstream
source, not the (differently simplified) copies under ``src/jax_cddd/_legacy_tf1/``
-- building the real ``tf.contrib.rnn.MultiRNNCell``/``tf.contrib.seq2seq.BeamSearchDecoder``
graph and restoring the real checkpoint via ``tf.train.Saver``.

Run this in the ``jax-cddd-convert`` environment, which carries a legacy
TensorFlow 1.15 (the last TF1 release, still with a full ``tf.contrib``) for
exactly this purpose:

    micromamba run -n jax-cddd-convert python scripts/run_original_model.py \\
        [--zip PATH_TO_default_model.zip] [--out tests/fixtures/original_code_reference.npz]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
CDDD_MASTER = REPO_ROOT / "src" / "jax_cddd" / "_cddd_ref_" / "cddd-master"
DEFAULT_ZIP = REPO_ROOT / "src" / "jax_cddd" / "_cddd_ref_" / "default_model.zip"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "original_code_reference.npz"

sys.path.insert(0, str(CDDD_MASTER))  # the pristine original "cddd" package

# A handful of real molecules -- same curated set used in
# tests/test_roundtrip_smiles.py, so results are directly comparable.
CURATED_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "c1ccncc1",
    "CCN(CC)CC",
]

BEAM_WIDTH = 10


def _extract_model_dir(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    model_dirs = [p for p in extract_dir.rglob("checkpoint") if p.is_file()]
    if len(model_dirs) != 1:
        raise RuntimeError(f"Expected exactly one extracted model dir, found {model_dirs}")
    return model_dirs[0].parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    from cddd.inference import InferenceModel  # the real, original implementation

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _extract_model_dir(args.zip, Path(tmp))

        model = InferenceModel(
            model_dir=str(model_dir),
            use_gpu=False,
            gpu_mem_frac=0.0,
            beam_width=BEAM_WIDTH,
            num_top=1,
            maximum_iterations=150,
            cpu_threads=2,
        )

        embeddings = model.seq_to_emb(CURATED_SMILES)
        reconstructed = model.emb_to_seq(embeddings.copy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        smiles=np.array(CURATED_SMILES),
        embeddings=embeddings.astype(np.float32),
        reconstructed=np.array(reconstructed),
        beam_width=np.array(BEAM_WIDTH),
    )
    print(f"Wrote original-code embeddings/reconstructions for {len(CURATED_SMILES)} molecules to {args.out}")
    for s, r in zip(CURATED_SMILES, reconstructed):
        print(f"  {s!r:45s} -> {r!r}")


if __name__ == "__main__":
    main()
