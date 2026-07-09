#!/usr/bin/env python
"""Convert the original CDDD TF1 checkpoint (``default_model.zip``) into the JAX
port's flat ``.npz`` weight format.

Run this in the ``jax-cddd-convert`` environment (CPU TensorFlow only -- this
script never executes the old graph, it only reads raw tensors out of the
checkpoint by name via ``tf.train.load_checkpoint``, which works fine under any
modern TF2 install regardless of the removed ``tf.contrib`` ops the original graph
used).

Usage:
    micromamba run -n jax-cddd-convert python scripts/convert_checkpoint.py \\
        [--zip PATH_TO_default_model.zip] [--out OUTPUT_NPZ_PATH]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from jax_cddd.param_names import CELL_SIZES, NUM_LAYERS, layer_names

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = REPO_ROOT / "src" / "jax_cddd" / "_cddd_ref_" / "default_model.zip"
DEFAULT_OUT = REPO_ROOT / "src" / "jax_cddd" / "data" / "default_model_params.npz"

# Tensors we deliberately don't port: the auxiliary QSAR feature-regression head
# (out of scope -- see project plan), and training-only bookkeeping (Adam slot
# variables, which appear suffixed "/optimizer" and "/optimizer_1", plus
# global_step and the Adam beta power scalars).
_IGNORED_PREFIXES = ("Feature_Regression/", "Training/")
_IGNORED_EXACT = {"global_step"}


def _is_ignored(name: str) -> bool:
    if name in _IGNORED_EXACT:
        return True
    if any(name.startswith(p) for p in _IGNORED_PREFIXES):
        return True
    if name.endswith("/optimizer") or name.endswith("/optimizer_1"):
        return True
    return False


# Checkpoint tensor name -> flat .npz key, for everything we DO port.
def _checkpoint_to_flat_map(num_layers: int = NUM_LAYERS):
    mapping = {"char_embedding": "embedding.table"}

    def add_layers(tf_scope: str, flat_side: str):
        for i in range(num_layers):
            tf_prefix = f"{tf_scope}/cell_{i}/gru_cell/"
            flat_prefix = f"{flat_side}.layer{i}."
            mapping[tf_prefix + "gates/kernel"] = flat_prefix + "gate_kernel"
            mapping[tf_prefix + "gates/bias"] = flat_prefix + "gate_bias"
            mapping[tf_prefix + "candidate/kernel"] = flat_prefix + "candidate_kernel"
            mapping[tf_prefix + "candidate/bias"] = flat_prefix + "candidate_bias"

    add_layers("Encoder/rnn/multi_rnn_cell", "encoder")
    add_layers("Decoder/decoder/multi_rnn_cell", "decoder")

    mapping["Encoder/dense/kernel"] = "encoder.bottleneck_kernel"
    mapping["Encoder/dense/bias"] = "encoder.bottleneck_bias"
    mapping["Decoder/dense/kernel"] = "decoder.init_state_kernel"
    mapping["Decoder/dense/bias"] = "decoder.init_state_bias"
    mapping["Decoder/decoder/dense/kernel"] = "decoder.output_proj_kernel"
    return mapping


def _resolve_checkpoint_prefix(zip_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    index_files = list(extract_dir.rglob("*.index"))
    if len(index_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .index file in {zip_path}, found {len(index_files)}: {index_files}"
        )
    return index_files[0].with_suffix("")  # strip ".index" -> checkpoint prefix


def convert(zip_path: Path, out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_prefix = _resolve_checkpoint_prefix(zip_path, Path(tmp))
        reader = tf.train.load_checkpoint(str(ckpt_prefix))
        shape_map = reader.get_variable_to_shape_map()

        checkpoint_to_flat = _checkpoint_to_flat_map()
        expected_tf_names = set(checkpoint_to_flat)

        available = set(shape_map)
        missing = expected_tf_names - available
        if missing:
            raise RuntimeError(
                "Checkpoint is missing expected tensors, aborting conversion:\n  "
                + "\n  ".join(sorted(missing))
            )

        unexpected = {n for n in available if not _is_ignored(n) and n not in expected_tf_names}
        if unexpected:
            print(
                "WARNING: checkpoint contains tensors this converter doesn't "
                "recognize and is NOT porting (verify these are safe to skip):",
                file=sys.stderr,
            )
            for name in sorted(unexpected):
                print(f"  {name}  {shape_map[name]}", file=sys.stderr)

        flat = {}
        for tf_name, flat_name in checkpoint_to_flat.items():
            flat[flat_name] = reader.get_tensor(tf_name)

        # Sanity-check shapes against the expected architecture before saving.
        expected_shapes = _expected_shapes()
        for flat_name, array in flat.items():
            expected = expected_shapes[flat_name]
            if tuple(array.shape) != expected:
                raise RuntimeError(
                    f"Shape mismatch for {flat_name}: got {array.shape}, expected {expected}"
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **flat)
        print(f"Wrote {len(flat)} tensors to {out_path}")


def _expected_shapes():
    char_emb, vocab = 32, 40
    shapes = {"embedding.table": (vocab, char_emb)}

    def add_layers(side: str):
        d = char_emb
        for i, size in enumerate(CELL_SIZES):
            prefix = f"{side}.layer{i}."
            shapes[prefix + "gate_kernel"] = (d + size, 2 * size)
            shapes[prefix + "gate_bias"] = (2 * size,)
            shapes[prefix + "candidate_kernel"] = (d + size, size)
            shapes[prefix + "candidate_bias"] = (size,)
            d = size

    add_layers("encoder")
    add_layers("decoder")
    total = sum(CELL_SIZES)
    emb_size = 512
    shapes["encoder.bottleneck_kernel"] = (total, emb_size)
    shapes["encoder.bottleneck_bias"] = (emb_size,)
    shapes["decoder.init_state_kernel"] = (emb_size, total)
    shapes["decoder.init_state_bias"] = (total,)
    shapes["decoder.output_proj_kernel"] = (CELL_SIZES[-1], vocab)
    return shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.zip.exists():
        raise FileNotFoundError(f"Checkpoint zip not found: {args.zip}")
    convert(args.zip, args.out)


if __name__ == "__main__":
    main()
