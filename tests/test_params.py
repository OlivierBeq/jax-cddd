import numpy as np
import jax.numpy as jnp

from jax_cddd.gru import GRULayerParams
from jax_cddd.params import (
    CDDDParams,
    DecoderParams,
    EncoderParams,
    from_flat_dict,
    load_npz,
    save_npz,
    to_flat_dict,
)


def _random_layer(rng, in_dim, hidden_dim) -> GRULayerParams:
    return GRULayerParams(
        gate_kernel=jnp.asarray(rng.normal(size=(in_dim + hidden_dim, 2 * hidden_dim)).astype(np.float32)),
        gate_bias=jnp.asarray(rng.normal(size=(2 * hidden_dim,)).astype(np.float32)),
        candidate_kernel=jnp.asarray(rng.normal(size=(in_dim + hidden_dim, hidden_dim)).astype(np.float32)),
        candidate_bias=jnp.asarray(rng.normal(size=(hidden_dim,)).astype(np.float32)),
    )


def _make_synthetic_params(rng) -> CDDDParams:
    sizes = [4, 6, 8]  # small synthetic cell_size stand-in, not the real 512/1024/2048
    char_emb, vocab, emb_size = 3, 5, 7

    enc_layers = []
    d = char_emb
    for size in sizes:
        enc_layers.append(_random_layer(rng, d, size))
        d = size
    dec_layers = []
    d = char_emb
    for size in sizes:
        dec_layers.append(_random_layer(rng, d, size))
        d = size

    total = sum(sizes)
    encoder = EncoderParams(
        layers=tuple(enc_layers),
        bottleneck_kernel=jnp.asarray(rng.normal(size=(total, emb_size)).astype(np.float32)),
        bottleneck_bias=jnp.asarray(rng.normal(size=(emb_size,)).astype(np.float32)),
    )
    decoder = DecoderParams(
        layers=tuple(dec_layers),
        init_state_kernel=jnp.asarray(rng.normal(size=(emb_size, total)).astype(np.float32)),
        init_state_bias=jnp.asarray(rng.normal(size=(total,)).astype(np.float32)),
        output_proj_kernel=jnp.asarray(rng.normal(size=(sizes[-1], vocab)).astype(np.float32)),
    )
    return CDDDParams(
        embedding=jnp.asarray(rng.normal(size=(vocab, char_emb)).astype(np.float32)),
        encoder=encoder,
        decoder=decoder,
    )


def _assert_params_equal(a: CDDDParams, b: CDDDParams):
    np.testing.assert_array_equal(np.asarray(a.embedding), np.asarray(b.embedding))
    for la, lb in zip(a.encoder.layers, b.encoder.layers):
        for field in GRULayerParams._fields:
            np.testing.assert_array_equal(np.asarray(getattr(la, field)), np.asarray(getattr(lb, field)))
    for la, lb in zip(a.decoder.layers, b.decoder.layers):
        for field in GRULayerParams._fields:
            np.testing.assert_array_equal(np.asarray(getattr(la, field)), np.asarray(getattr(lb, field)))
    np.testing.assert_array_equal(np.asarray(a.encoder.bottleneck_kernel), np.asarray(b.encoder.bottleneck_kernel))
    np.testing.assert_array_equal(np.asarray(a.decoder.output_proj_kernel), np.asarray(b.decoder.output_proj_kernel))


def test_flat_roundtrip():
    rng = np.random.default_rng(0)
    params = _make_synthetic_params(rng)
    flat = to_flat_dict(params)
    restored = from_flat_dict(flat, num_layers=3)
    _assert_params_equal(params, restored)


def test_npz_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    params = _make_synthetic_params(rng)
    path = tmp_path / "params.npz"
    save_npz(params, path)
    restored = load_npz(path, num_layers=3)
    _assert_params_equal(params, restored)
