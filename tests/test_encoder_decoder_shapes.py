import os

import jax.numpy as jnp
import numpy as np
import pytest

from jax_cddd.modules import decode_step, decode_teacher_forced, decoder_initial_states, encode
from jax_cddd.params import default_params_path, load_default_model
from jax_cddd.vocab import Vocabulary, encode_batch

requires_default_model = pytest.mark.skipif(
    not os.path.exists(default_params_path()),
    reason="default_model_params.npz not found; run scripts/convert_checkpoint.py first",
)


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.default()


@pytest.fixture(scope="module")
def params():
    return load_default_model()


@requires_default_model
def test_encode_shape(params, vocab):
    ids, lengths = encode_batch(["CCO", "c1ccccc1"], vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))
    assert descriptor.shape == (2, 512)
    assert np.all(np.abs(np.asarray(descriptor)) <= 1.0)  # tanh-bounded


@requires_default_model
def test_decoder_initial_states_shapes(params, vocab):
    ids, lengths = encode_batch(["CCO"], vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))
    states = decoder_initial_states(params, descriptor)
    assert [s.shape[-1] for s in states] == [512, 1024, 2048]
    assert all(s.shape[0] == 1 for s in states)


@requires_default_model
def test_teacher_forced_logits_shape(params, vocab):
    smiles = "CCO"
    ids, lengths = encode_batch([smiles], vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))
    target_ids = jnp.asarray(ids)  # decode the same sequence back (batch=1, no padding)
    logits = decode_teacher_forced(params, descriptor, target_ids)
    assert logits.shape == (1, ids.shape[1], 40)


@requires_default_model
def test_decode_step_matches_teacher_forced_first_step(params, vocab):
    smiles = "CCO"
    ids, lengths = encode_batch([smiles], vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))
    target_ids = jnp.asarray(ids)

    tf_logits = decode_teacher_forced(params, descriptor, target_ids)

    states = decoder_initial_states(params, descriptor)
    prev_ids = target_ids[:, 0]  # <s>
    _, step_logits = decode_step(params, states, prev_ids)

    np.testing.assert_allclose(np.asarray(step_logits), np.asarray(tf_logits[:, 0]), atol=1e-5, rtol=1e-5)
