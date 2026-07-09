"""Parameter pytree definitions for the CDDD model, ``.npz`` (de)serialization, and
a loader for the pretrained ``default_model`` weights.

The nested dataclasses here mirror the checkpoint variable names 1:1 (see
``scripts/convert_checkpoint.py`` for the exact tensor-name -> field mapping), so
that converting the original TF1 checkpoint requires no reshaping/transposition
beyond simple concatenation bookkeeping.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import struct

from jax_cddd._download import download_file
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
    "download_default_params",
    "load_default_model",
]

_DEFAULT_PARAMS_FILENAME = "default_model_params.npz"

# Can be overridden without a code change via the JAX_CDDD_WEIGHTS_URL
# environment variable (which takes precedence over this constant).
_DEFAULT_PARAMS_URL: Optional[str] = (
    "https://github.com/OlivierBeq/jax-cddd/releases/download/model_weights/default_model_params.npz"
)
_DEFAULT_PARAMS_SHA256: Optional[str] = (
    "0c4bd08a593c78221c20506eca8e8b88b823b98e65ef3b3784ed4a63f266bd23"
)

_PARAMS_URL_ENV_VAR = "JAX_CDDD_WEIGHTS_URL"


@struct.dataclass
class EncoderParams:
    """A ``flax.struct.dataclass`` (registered as a JAX pytree), so instances
    can be passed directly into ``jax.jit``-compiled functions as ordinary
    (traced) arguments -- not as opaque static Python objects."""

    layers: Tuple[GRULayerParams, ...]
    bottleneck_kernel: jnp.ndarray  # [sum(CELL_SIZES), EMB_SIZE]
    bottleneck_bias: jnp.ndarray    # [EMB_SIZE]


@struct.dataclass
class DecoderParams:
    layers: Tuple[GRULayerParams, ...]
    init_state_kernel: jnp.ndarray    # [EMB_SIZE, sum(CELL_SIZES)]
    init_state_bias: jnp.ndarray      # [sum(CELL_SIZES)]
    output_proj_kernel: jnp.ndarray   # [CELL_SIZES[-1], VOCAB_SIZE], no bias


@struct.dataclass
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


def download_default_params(
    url: Optional[str] = None,
    dest: Optional[Union[str, Path]] = None,
    expected_sha256: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download the converted ``default_model`` weights (``.npz``) from the
    GitHub release and place them at ``dest``.

    A no-op if ``dest`` already exists (unless ``force=True``), so this is safe
    to call unconditionally on every model load -- see ``load_default_model()``.

    Args:
        url: Overrides the download URL. Defaults to the
            ``JAX_CDDD_WEIGHTS_URL`` environment variable, then the
            ``_DEFAULT_PARAMS_URL`` constant in this module.
        dest: Overrides the destination path. Defaults to
            ``default_params_path()``.
        expected_sha256: Overrides the expected checksum (defaults to
            ``_DEFAULT_PARAMS_SHA256``); verified after download, raising
            ``RuntimeError`` on mismatch (the bad download is removed, not left
            behind).
        force: Re-download even if ``dest`` already exists.

    Returns:
        The path the weights are available at.
    """
    dest = Path(dest) if dest is not None else default_params_path()
    resolved_url = url or os.environ.get(_PARAMS_URL_ENV_VAR) or _DEFAULT_PARAMS_URL
    if not dest.exists() and not resolved_url:
        raise RuntimeError(
            f"No converted weights found at {dest}, and no download URL is "
            f"configured. Either set the {_PARAMS_URL_ENV_VAR} environment "
            "variable, pass url=..., or run scripts/convert_checkpoint.py (in "
            "the `jax-cddd-convert` environment) to produce them locally from "
            "default_model.zip."
        )
    return download_file(
        resolved_url,
        dest,
        expected_sha256=expected_sha256 if expected_sha256 is not None else _DEFAULT_PARAMS_SHA256,
        force=force,
        label="pretrained CDDD weights",
    )


def load_default_model() -> CDDDParams:
    """Load the pretrained CDDD weights, already converted to this module's JAX
    pytree/``.npz`` format.

    Downloads them first (via ``download_default_params()``) if no local
    cached copy is found -- see that function for how to configure the URL.
    """
    path = download_default_params()
    return load_npz(path)
