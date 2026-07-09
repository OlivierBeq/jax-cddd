"""Parameter pytree definitions for the CDDD model, ``.npz`` (de)serialization, and
a loader for the pretrained ``default_model`` weights.

The nested dataclasses here mirror the checkpoint variable names 1:1 (see
``scripts/convert_checkpoint.py`` for the exact tensor-name -> field mapping), so
that converting the original TF1 checkpoint requires no reshaping/transposition
beyond simple concatenation bookkeeping.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from jax_cddd.gru import GRULayerParams
from jax_cddd.param_names import CELL_SIZES, CHAR_EMBEDDING_SIZE, EMB_SIZE, NUM_LAYERS, VOCAB_SIZE

__all__ = [
    "CELL_SIZES",
    "CHAR_EMBEDDING_SIZE",
    "EMB_SIZE",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "EncoderParams",
    "DecoderParams",
    "CDDDParams",
    "to_flat_dict",
    "from_flat_dict",
    "save_npz",
    "load_npz",
    "default_params_path",
    "load_default_model",
]

_DEFAULT_PARAMS_FILENAME = "default_model_params.npz"
# TODO: point this at a hosted release asset once one exists. Until then,
# load_default_model() falls back to a clear error pointing at
# scripts/convert_checkpoint.py, which produces this file locally from the
# (untracked, locally-present) default_model.zip.
_DEFAULT_PARAMS_URL = None


@dataclass
class EncoderParams:
    layers: Tuple[GRULayerParams, ...]
    bottleneck_kernel: jnp.ndarray  # [sum(CELL_SIZES), EMB_SIZE]
    bottleneck_bias: jnp.ndarray    # [EMB_SIZE]


@dataclass
class DecoderParams:
    layers: Tuple[GRULayerParams, ...]
    init_state_kernel: jnp.ndarray    # [EMB_SIZE, sum(CELL_SIZES)]
    init_state_bias: jnp.ndarray      # [sum(CELL_SIZES)]
    output_proj_kernel: jnp.ndarray   # [CELL_SIZES[-1], VOCAB_SIZE], no bias


@dataclass
class CDDDParams:
    embedding: jnp.ndarray  # [VOCAB_SIZE, CHAR_EMBEDDING_SIZE], shared encoder/decoder
    encoder: EncoderParams
    decoder: DecoderParams


def _layer_to_flat(flat: Dict[str, np.ndarray], prefix: str, layer: GRULayerParams) -> None:
    flat[prefix + "gate_kernel"] = np.asarray(layer.gate_kernel)
    flat[prefix + "gate_bias"] = np.asarray(layer.gate_bias)
    flat[prefix + "candidate_kernel"] = np.asarray(layer.candidate_kernel)
    flat[prefix + "candidate_bias"] = np.asarray(layer.candidate_bias)


def to_flat_dict(params: CDDDParams) -> Dict[str, np.ndarray]:
    """Flatten a ``CDDDParams`` pytree into a ``{dotted.name: array}`` dict, the
    format used for ``.npz`` storage (npz has no native nesting)."""
    flat: Dict[str, np.ndarray] = {"embedding.table": np.asarray(params.embedding)}
    for i, layer in enumerate(params.encoder.layers):
        _layer_to_flat(flat, f"encoder.layer{i}.", layer)
    flat["encoder.bottleneck_kernel"] = np.asarray(params.encoder.bottleneck_kernel)
    flat["encoder.bottleneck_bias"] = np.asarray(params.encoder.bottleneck_bias)
    for i, layer in enumerate(params.decoder.layers):
        _layer_to_flat(flat, f"decoder.layer{i}.", layer)
    flat["decoder.init_state_kernel"] = np.asarray(params.decoder.init_state_kernel)
    flat["decoder.init_state_bias"] = np.asarray(params.decoder.init_state_bias)
    flat["decoder.output_proj_kernel"] = np.asarray(params.decoder.output_proj_kernel)
    return flat


def _layers_from_flat(flat, prefix: str, num_layers: int) -> Tuple[GRULayerParams, ...]:
    layers = []
    for i in range(num_layers):
        p = f"{prefix}.layer{i}."
        layers.append(
            GRULayerParams(
                gate_kernel=jnp.asarray(flat[p + "gate_kernel"]),
                gate_bias=jnp.asarray(flat[p + "gate_bias"]),
                candidate_kernel=jnp.asarray(flat[p + "candidate_kernel"]),
                candidate_bias=jnp.asarray(flat[p + "candidate_bias"]),
            )
        )
    return tuple(layers)


def from_flat_dict(flat: Dict[str, np.ndarray], num_layers: int = NUM_LAYERS) -> CDDDParams:
    encoder = EncoderParams(
        layers=_layers_from_flat(flat, "encoder", num_layers),
        bottleneck_kernel=jnp.asarray(flat["encoder.bottleneck_kernel"]),
        bottleneck_bias=jnp.asarray(flat["encoder.bottleneck_bias"]),
    )
    decoder = DecoderParams(
        layers=_layers_from_flat(flat, "decoder", num_layers),
        init_state_kernel=jnp.asarray(flat["decoder.init_state_kernel"]),
        init_state_bias=jnp.asarray(flat["decoder.init_state_bias"]),
        output_proj_kernel=jnp.asarray(flat["decoder.output_proj_kernel"]),
    )
    return CDDDParams(embedding=jnp.asarray(flat["embedding.table"]), encoder=encoder, decoder=decoder)


def save_npz(params: CDDDParams, path) -> None:
    np.savez(path, **to_flat_dict(params))


def load_npz(path, num_layers: int = NUM_LAYERS) -> CDDDParams:
    with np.load(path) as data:
        flat = {k: data[k] for k in data.files}
    return from_flat_dict(flat, num_layers=num_layers)


def default_params_path():
    return Path(__file__).parent / "data" / _DEFAULT_PARAMS_FILENAME


def _download(url: str, dest) -> None:
    dest_path = str(dest)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)


def load_default_model() -> CDDDParams:
    """Load the pretrained CDDD weights, already converted to this module's JAX
    pytree/``.npz`` format.

    Looks for a local cached copy first (produced by ``scripts/convert_checkpoint.py``
    from the locally-present ``default_model.zip`` reference checkpoint). No hosted
    download exists yet, so if no local copy is found and no URL is configured this
    raises with actionable instructions rather than failing silently.
    """
    path = default_params_path()
    if not os.path.exists(path):
        if _DEFAULT_PARAMS_URL:
            _download(_DEFAULT_PARAMS_URL, path)
        else:
            raise FileNotFoundError(
                f"No converted weights found at {path}. Run "
                "scripts/convert_checkpoint.py (in the `jax-cddd-convert` "
                "environment) to produce it from the local default_model.zip."
            )
    return load_npz(path)
