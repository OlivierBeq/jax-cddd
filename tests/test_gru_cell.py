import numpy as np
import jax.numpy as jnp
import pytest

from jax_cddd.gru import (
    GRULayerParams,
    gru_cell_apply,
    run_stacked_gru,
    stacked_gru_step,
    init_states,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _numpy_gru_reference(params: GRULayerParams, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Independent NumPy re-derivation of the exact TF1 GRUCell formula, used to
    cross-check the JAX implementation (not shared code with gru.py)."""
    gate_in = np.concatenate([x, h], axis=-1)
    gates = _sigmoid(gate_in @ params.gate_kernel + params.gate_bias)
    r, u = np.split(gates, 2, axis=-1)
    cand_in = np.concatenate([x, r * h], axis=-1)
    c = np.tanh(cand_in @ params.candidate_kernel + params.candidate_bias)
    return u * h + (1.0 - u) * c


def _random_layer(rng, in_dim, hidden_dim) -> GRULayerParams:
    return GRULayerParams(
        gate_kernel=jnp.asarray(rng.normal(size=(in_dim + hidden_dim, 2 * hidden_dim)) * 0.1),
        gate_bias=jnp.asarray(rng.normal(size=(2 * hidden_dim,)) * 0.1),
        candidate_kernel=jnp.asarray(rng.normal(size=(in_dim + hidden_dim, hidden_dim)) * 0.1),
        candidate_bias=jnp.asarray(rng.normal(size=(hidden_dim,)) * 0.1),
    )


def test_gru_cell_matches_numpy_reference():
    rng = np.random.default_rng(0)
    in_dim, hidden_dim, batch = 5, 8, 4
    layer = _random_layer(rng, in_dim, hidden_dim)
    x = rng.normal(size=(batch, in_dim)).astype(np.float32)
    h = rng.normal(size=(batch, hidden_dim)).astype(np.float32)

    got = np.asarray(gru_cell_apply(layer, jnp.asarray(x), jnp.asarray(h)))
    expected = _numpy_gru_reference(layer, x, h)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-6)


def test_gate_order_is_reset_then_update():
    # Force update gate u->1 (so new_h == h regardless of candidate) by making the
    # *second* half of the gate output saturate positive, and check state is
    # unchanged -- this pins down that split() gives (r, u) in that order.
    hidden_dim = 4
    in_dim = 1
    gate_kernel = jnp.zeros((in_dim + hidden_dim, 2 * hidden_dim))
    gate_bias = jnp.concatenate([jnp.zeros(hidden_dim), jnp.full((hidden_dim,), 20.0)])
    candidate_kernel = jnp.zeros((in_dim + hidden_dim, hidden_dim))
    candidate_bias = jnp.zeros(hidden_dim)
    layer = GRULayerParams(gate_kernel, gate_bias, candidate_kernel, candidate_bias)

    h = jnp.asarray(np.random.default_rng(1).normal(size=(2, hidden_dim)).astype(np.float32))
    x = jnp.zeros((2, in_dim))
    new_h = gru_cell_apply(layer, x, h)
    np.testing.assert_allclose(np.asarray(new_h), np.asarray(h), atol=1e-5)


def test_stacked_gru_step_chains_layers():
    rng = np.random.default_rng(2)
    sizes = [3, 5]
    in_dim = 4
    batch = 2
    layers = []
    d = in_dim
    for size in sizes:
        layers.append(_random_layer(rng, d, size))
        d = size
    states = init_states(layers, batch)
    x = jnp.asarray(rng.normal(size=(batch, in_dim)).astype(np.float32))

    new_states, top_out = stacked_gru_step(layers, x, states)

    # manual chain, reusing only the already-verified single-cell function
    h0 = gru_cell_apply(layers[0], x, states[0])
    h1 = gru_cell_apply(layers[1], h0, states[1])
    np.testing.assert_allclose(np.asarray(new_states[0]), np.asarray(h0))
    np.testing.assert_allclose(np.asarray(new_states[1]), np.asarray(h1))
    np.testing.assert_allclose(np.asarray(top_out), np.asarray(h1))


def test_run_stacked_gru_matches_manual_loop_without_masking():
    rng = np.random.default_rng(3)
    sizes = [3, 5]
    in_dim = 4
    batch, time = 2, 6
    layers = []
    d = in_dim
    for size in sizes:
        layers.append(_random_layer(rng, d, size))
        d = size
    x_seq = jnp.asarray(rng.normal(size=(batch, time, in_dim)).astype(np.float32))

    final_states, outputs = run_stacked_gru(layers, x_seq, seq_len=None)

    states = init_states(layers, batch)
    manual_outputs = []
    for t in range(time):
        states, top_out = stacked_gru_step(layers, x_seq[:, t], states)
        manual_outputs.append(top_out)
    manual_outputs = jnp.stack(manual_outputs, axis=1)

    np.testing.assert_allclose(np.asarray(outputs), np.asarray(manual_outputs), atol=1e-6)
    for got, expected in zip(final_states, states):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6)


def test_run_stacked_gru_masking_freezes_state_past_seq_len():
    rng = np.random.default_rng(4)
    sizes = [3, 5]
    in_dim = 4
    batch, time = 3, 8
    layers = []
    d = in_dim
    for size in sizes:
        layers.append(_random_layer(rng, d, size))
        d = size
    seq_len = jnp.asarray([2, 5, 8])

    x_seq = jnp.asarray(rng.normal(size=(batch, time, in_dim)).astype(np.float32))
    final_states, _ = run_stacked_gru(layers, x_seq, seq_len=seq_len)

    # Reference: run each batch row independently only up to its own true length.
    for row in range(batch):
        length = int(seq_len[row])
        x_row = x_seq[row : row + 1, :length]
        row_final_states, _ = run_stacked_gru(layers, x_row, seq_len=None)
        for got, expected in zip(final_states, row_final_states):
            np.testing.assert_allclose(
                np.asarray(got[row]), np.asarray(expected[0]), atol=1e-5, rtol=1e-5
            )


def test_run_stacked_gru_padding_content_does_not_affect_final_state():
    # Two batches identical up to seq_len, differing garbage after -- final state
    # must match exactly, since this is what the encoder relies on for padded batches.
    rng = np.random.default_rng(5)
    sizes = [4]
    in_dim = 3
    time = 6
    layers = [_random_layer(rng, in_dim, sizes[0])]
    seq_len = jnp.asarray([3, 3])

    common = rng.normal(size=(1, 3, in_dim)).astype(np.float32)
    tail_a = rng.normal(size=(1, 3, in_dim)).astype(np.float32)
    tail_b = rng.normal(size=(1, 3, in_dim)).astype(np.float32)
    x_seq = jnp.asarray(
        np.concatenate(
            [
                np.concatenate([common, tail_a], axis=1),
                np.concatenate([common, tail_b], axis=1),
            ],
            axis=0,
        )
    )

    final_states, _ = run_stacked_gru(layers, x_seq, seq_len=seq_len)
    np.testing.assert_allclose(
        np.asarray(final_states[0][0]), np.asarray(final_states[0][1]), atol=1e-6
    )
