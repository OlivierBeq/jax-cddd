#!/usr/bin/env python
"""Generate ground-truth encoder/decoder activations from the original TF1
checkpoint, for the JAX port's numerical-fidelity test (``tests/test_against_tf1_reference.py``).

Run this in the ``jax-cddd-convert`` environment (CPU TensorFlow).

Why this doesn't execute the original TF1 graph: this script deliberately
recomputes the encoder/decoder forward pass with an independent, from-scratch
NumPy implementation of the documented ``tf.nn.rnn_cell.GRUCell`` formula --
*not* importing ``jax_cddd.gru`` (which implements the same formula in JAX) and
*not* running the original ``cddd`` graph code -- so this is a cross-check of
the JAX port's math against the formula itself, independent of any particular
TF version or the original repo's code. (For a complementary, genuinely
end-to-end check that runs the actual unmodified original ``cddd`` package, see
``scripts/run_original_model.py`` / ``tests/test_against_original_code.py``.)
This script only needs ``tf.train.load_checkpoint`` (the checkpoint's raw
variable-name->tensor map), so it works under any TF1 or TF2 install.

Usage:
    micromamba run -n jax-cddd-convert python scripts/validate_against_tf1_reference.py \\
        [--zip PATH_TO_default_model.zip] [--out tests/fixtures/tf1_reference.npz]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # use jax_cddd from source, no install needed

from jax_cddd.vocab import Vocabulary  # noqa: E402

DEFAULT_ZIP = REPO_ROOT / "src" / "jax_cddd" / "_cddd_ref_" / "default_model.zip"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "tf1_reference.npz"

CELL_SIZES = (512, 1024, 2048)

# A deliberately diverse set of test molecules: short/long, branched, aromatic,
# charged, multi-char halogens, stereochemistry, and a few real drugs.
TEST_SMILES = [
    "C",
    "CC",
    "CCO",
    "CCCCCCCC",
    "CC(C)C",
    "C1CCCCC1",
    "c1ccccc1",
    "c1ccncc1",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "ClCCl",
    "BrC1=CC=CC=C1",
    "C1=CC=C(C=C1)Cl",
    "[Na+].[Cl-]",
    "CC(N)C(=O)O",
    "C[C@H](N)C(=O)O",  # stereochemistry
    "C[C@@H](O)C(=O)O",
    "O=C(O)c1ccccc1",
    "Nc1ccccc1",
    "c1ccc2ccccc2c1",  # naphthalene
    "CCN(CC)CC",
    "OCC(O)CO",  # glycerol
    "C1=CC=C2C(=C1)C=CC=C2",
    "S(=O)(=O)(O)O",
    "N#Cc1ccccc1",
    "FC(F)(F)c1ccccc1",
    "CC1=CC(=O)C=CC1=O",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clip before exponentiating to avoid overflow warnings on saturated gates;
    # sigmoid is already indistinguishable from 0.0/1.0 at float32 precision well
    # before |x| reaches this range, so this doesn't affect the result.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _gru_step(layer, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Independent NumPy re-derivation of tf.nn.rnn_cell.GRUCell.call, from the
    documented formula (not imported from jax_cddd.gru)."""
    gate_in = np.concatenate([x, h], axis=-1)
    gates = _sigmoid(gate_in @ layer["gate_kernel"] + layer["gate_bias"])
    r, u = np.split(gates, 2, axis=-1)
    cand_in = np.concatenate([x, r * h], axis=-1)
    c = np.tanh(cand_in @ layer["candidate_kernel"] + layer["candidate_bias"])
    return u * h + (1.0 - u) * c


def _run_stack(layers, x_seq: np.ndarray):
    """x_seq: [T, in_dim] (no batch axis -- one molecule at a time).

    Returns (final_states: list of [hidden_i], outputs: [T, hidden_last]).
    """
    states = [np.zeros(layer["candidate_bias"].shape[-1], dtype=np.float32) for layer in layers]
    outputs = []
    for t in range(x_seq.shape[0]):
        layer_input = x_seq[t]
        new_states = []
        for layer, h in zip(layers, states):
            new_h = _gru_step(layer, layer_input, h)
            new_states.append(new_h)
            layer_input = new_h
        states = new_states
        outputs.append(layer_input)
    return states, np.stack(outputs, axis=0)


def _layers_for(tensors, tf_scope: str):
    layers = []
    for i in range(len(CELL_SIZES)):
        prefix = f"{tf_scope}/cell_{i}/gru_cell/"
        layers.append(
            {
                "gate_kernel": tensors[prefix + "gates/kernel"],
                "gate_bias": tensors[prefix + "gates/bias"],
                "candidate_kernel": tensors[prefix + "candidate/kernel"],
                "candidate_bias": tensors[prefix + "candidate/bias"],
            }
        )
    return layers


def encode(tensors, ids: np.ndarray) -> np.ndarray:
    embedding = tensors["char_embedding"]
    x_seq = embedding[ids]
    enc_layers = _layers_for(tensors, "Encoder/rnn/multi_rnn_cell")
    final_states, _ = _run_stack(enc_layers, x_seq)
    concat = np.concatenate(final_states, axis=-1)
    bottleneck = concat @ tensors["Encoder/dense/kernel"] + tensors["Encoder/dense/bias"]
    return np.tanh(bottleneck)


def decode_teacher_forced(tensors, descriptor: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    flat = descriptor @ tensors["Decoder/dense/kernel"] + tensors["Decoder/dense/bias"]
    split_points = [sum(CELL_SIZES[: i + 1]) for i in range(len(CELL_SIZES) - 1)]
    init_states = np.split(flat, split_points)

    dec_layers = _layers_for(tensors, "Decoder/decoder/multi_rnn_cell")
    embedding = tensors["char_embedding"]
    x_seq = embedding[target_ids]

    states = list(init_states)
    outputs = []
    for t in range(x_seq.shape[0]):
        layer_input = x_seq[t]
        new_states = []
        for layer, h in zip(dec_layers, states):
            new_h = _gru_step(layer, layer_input, h)
            new_states.append(new_h)
            layer_input = new_h
        states = new_states
        outputs.append(layer_input)
    outputs = np.stack(outputs, axis=0)
    return outputs @ tensors["Decoder/decoder/dense/kernel"]


def _resolve_checkpoint_prefix(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    index_files = list(extract_dir.rglob("*.index"))
    if len(index_files) != 1:
        raise RuntimeError(f"Expected exactly one .index file, found {index_files}")
    return index_files[0].with_suffix("")


def _load_all_tensors(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_prefix = _resolve_checkpoint_prefix(zip_path, Path(tmp))
        reader = tf.train.load_checkpoint(str(ckpt_prefix))
        names = [
            "char_embedding",
            "Encoder/dense/kernel",
            "Encoder/dense/bias",
            "Decoder/dense/kernel",
            "Decoder/dense/bias",
            "Decoder/decoder/dense/kernel",
        ]
        for scope in ("Encoder/rnn/multi_rnn_cell", "Decoder/decoder/multi_rnn_cell"):
            for i in range(len(CELL_SIZES)):
                prefix = f"{scope}/cell_{i}/gru_cell/"
                names += [prefix + s for s in ("gates/kernel", "gates/bias", "candidate/kernel", "candidate/bias")]
        return {name: reader.get_tensor(name) for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tensors = _load_all_tensors(args.zip)
    vocab = Vocabulary.default()

    fixture = {"smiles": np.array(TEST_SMILES)}
    for i, smiles in enumerate(TEST_SMILES):
        ids = np.array(vocab.encode(smiles), dtype=np.int32)
        descriptor = encode(tensors, ids)
        logits = decode_teacher_forced(tensors, descriptor, ids)
        fixture[f"ids_{i}"] = ids
        fixture[f"embedding_{i}"] = descriptor
        fixture[f"logits_{i}"] = logits

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **fixture)
    print(f"Wrote reference activations for {len(TEST_SMILES)} molecules to {args.out}")


if __name__ == "__main__":
    main()
