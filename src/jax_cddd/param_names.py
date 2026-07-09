"""Shared naming/shape constants for the CDDD parameter layout.

This module has **no JAX dependency** (only stdlib) so it can be imported from
either the main ``jax-cddd`` environment (via ``params.py``) or the TF-only
``jax-cddd-convert`` environment (via ``scripts/convert_checkpoint.py`` and
``scripts/validate_against_tf1_reference.py``), keeping the flat ``.npz`` naming
scheme defined in exactly one place.
"""
from __future__ import annotations

CELL_SIZES = (512, 1024, 2048)
CHAR_EMBEDDING_SIZE = 32
EMB_SIZE = 512
VOCAB_SIZE = 40
NUM_LAYERS = len(CELL_SIZES)

GRU_FIELDS = ("gate_kernel", "gate_bias", "candidate_kernel", "candidate_bias")


def layer_names(side: str, num_layers: int = NUM_LAYERS):
    """Flat .npz key names for one side's ('encoder' or 'decoder') GRU layers."""
    names = []
    for i in range(num_layers):
        prefix = f"{side}.layer{i}."
        names += [prefix + field for field in GRU_FIELDS]
    return names


def flat_param_names(num_layers: int = NUM_LAYERS):
    """Every flat .npz key name expected in a converted CDDDParams archive."""
    names = ["embedding.table"]
    names += layer_names("encoder", num_layers)
    names += ["encoder.bottleneck_kernel", "encoder.bottleneck_bias"]
    names += layer_names("decoder", num_layers)
    names += ["decoder.init_state_kernel", "decoder.init_state_bias", "decoder.output_proj_kernel"]
    return names
