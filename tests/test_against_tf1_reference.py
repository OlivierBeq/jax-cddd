"""Core numerical-fidelity gate: compares the JAX encoder/decoder against
reference activations computed directly from the original TF1 checkpoint's raw
tensors (see scripts/validate_against_tf1_reference.py for how the fixture file
was generated -- it needs TensorFlow and is not required to run these tests).
"""
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jax_cddd.modules import decode_teacher_forced, encode
from jax_cddd.params import default_params_path, load_default_model

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tf1_reference.npz"

ATOL = 1e-4
RTOL = 1e-4

pytestmark = [
    pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="tf1_reference.npz fixture missing"),
    pytest.mark.skipif(
        not os.path.exists(default_params_path()),
        reason="default_model_params.npz not found; run scripts/convert_checkpoint.py first",
    ),
]


@pytest.fixture(scope="module")
def params():
    return load_default_model()


@pytest.fixture(scope="module")
def fixture():
    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _molecule_indices(fixture):
    return [i for i in range(len(fixture["smiles"]))]


def test_encoder_matches_tf1_reference(params, fixture):
    max_abs_err = 0.0
    for i in _molecule_indices(fixture):
        ids = fixture[f"ids_{i}"]
        expected_emb = fixture[f"embedding_{i}"]

        jax_emb = encode(params, jnp.asarray(ids)[None, :], jnp.asarray([len(ids)]))
        jax_emb = np.asarray(jax_emb[0])

        err = np.max(np.abs(jax_emb - expected_emb))
        max_abs_err = max(max_abs_err, err)
        np.testing.assert_allclose(
            jax_emb,
            expected_emb,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"encoder mismatch for molecule {i} ({fixture['smiles'][i]!r})",
        )
    print(f"max abs encoder error across {len(fixture['smiles'])} molecules: {max_abs_err:.2e}")


def test_decoder_teacher_forced_logits_match_tf1_reference(params, fixture):
    max_abs_err = 0.0
    for i in _molecule_indices(fixture):
        ids = fixture[f"ids_{i}"]
        expected_emb = fixture[f"embedding_{i}"]
        expected_logits = fixture[f"logits_{i}"]

        jax_emb = encode(params, jnp.asarray(ids)[None, :], jnp.asarray([len(ids)]))
        jax_logits = decode_teacher_forced(params, jax_emb, jnp.asarray(ids)[None, :])
        jax_logits = np.asarray(jax_logits[0])

        err = np.max(np.abs(jax_logits - expected_logits))
        max_abs_err = max(max_abs_err, err)
        np.testing.assert_allclose(
            jax_logits,
            expected_logits,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"decoder logits mismatch for molecule {i} ({fixture['smiles'][i]!r})",
        )
    print(f"max abs decoder logit error across {len(fixture['smiles'])} molecules: {max_abs_err:.2e}")


def test_decoder_argmax_matches_tf1_reference(params, fixture):
    """Stricter, more interpretable check: predicted next-token ids must match
    exactly, not just the raw logits within tolerance."""
    for i in _molecule_indices(fixture):
        ids = fixture[f"ids_{i}"]
        expected_logits = fixture[f"logits_{i}"]

        jax_emb = encode(params, jnp.asarray(ids)[None, :], jnp.asarray([len(ids)]))
        jax_logits = decode_teacher_forced(params, jax_emb, jnp.asarray(ids)[None, :])
        jax_logits = np.asarray(jax_logits[0])

        expected_argmax = np.argmax(expected_logits, axis=-1)
        jax_argmax = np.argmax(jax_logits, axis=-1)
        np.testing.assert_array_equal(
            jax_argmax,
            expected_argmax,
            err_msg=f"argmax mismatch for molecule {i} ({fixture['smiles'][i]!r})",
        )
