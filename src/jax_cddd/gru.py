"""Hand-written GRU cell matching TensorFlow 1.x's ``tf.nn.rnn_cell.GRUCell`` math
exactly, plus utilities to run a stack of such cells the way
``tf.contrib.rnn.MultiRNNCell`` + ``tf.nn.dynamic_rnn`` did in the original CDDD graph.

Flax's/most JAX libraries' built-in GRU cells use a different gate layout (separate
input/recurrent kernels, different gate order), so weights from the pretrained
TF1 checkpoint cannot be loaded into them directly -- this module exists so the
checkpoint's tensors can be used completely unmodified (no transposition/reshaping
beyond simple concatenation-order bookkeeping).
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional, Sequence, Tuple

# This model is small enough that JAX's default GPU preallocation (~75-90% of
# total device memory) is wasteful and, on small GPUs, noisily probes down from
# an unallocatable size before settling (harmless, but looks like an error) --
# must be set before jax's backend initializes, hence before `import jax`.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax
import jax.numpy as jnp

# On GPU, JAX's default float32 matmul precision uses a reduced-precision
# (TF32-like) algorithm, which introduces ~1e-3 relative error relative to true
# float32 -- enough to break the 1e-4 numerical-fidelity target against the TF1
# reference (which always ran true float32 matmuls). This project cares about
# bit-level-ish fidelity more than matmul throughput, so force full precision.
jax.config.update("jax_default_matmul_precision", "highest")


class GRULayerParams(NamedTuple):
    """One stacked-GRU layer's parameters, named after the checkpoint's tensor names."""

    gate_kernel: jnp.ndarray       # [in_dim + hidden_dim, 2*hidden_dim]
    gate_bias: jnp.ndarray         # [2*hidden_dim]
    candidate_kernel: jnp.ndarray  # [in_dim + hidden_dim, hidden_dim]
    candidate_bias: jnp.ndarray    # [hidden_dim]

    @property
    def hidden_size(self) -> int:
        return self.candidate_bias.shape[-1]


def gru_cell_apply(params: GRULayerParams, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
    """One GRU step, replicating ``tf.nn.rnn_cell.GRUCell.call`` exactly::

        gate_inputs = concat([x, h]) @ gate_kernel + gate_bias
        r, u = split(sigmoid(gate_inputs), 2)      # reset, update -- in this order
        candidate = concat([x, r * h]) @ candidate_kernel + candidate_bias
        c = tanh(candidate)
        new_h = u * h + (1 - u) * c
    """
    gate_in = jnp.concatenate([x, h], axis=-1)
    gates = jax.nn.sigmoid(gate_in @ params.gate_kernel + params.gate_bias)
    r, u = jnp.split(gates, 2, axis=-1)
    cand_in = jnp.concatenate([x, r * h], axis=-1)
    c = jnp.tanh(cand_in @ params.candidate_kernel + params.candidate_bias)
    return u * h + (1.0 - u) * c


def init_states(
    layers: Sequence[GRULayerParams], batch_size: int, dtype=jnp.float32
) -> Tuple[jnp.ndarray, ...]:
    return tuple(jnp.zeros((batch_size, layer.hidden_size), dtype=dtype) for layer in layers)


def stacked_gru_step(
    layers: Sequence[GRULayerParams],
    x: jnp.ndarray,
    states: Sequence[jnp.ndarray],
) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
    """Run one timestep through a stack of GRU layers.

    Matches ``MultiRNNCell``: each layer's output at this timestep feeds the next
    layer's input at this same timestep (not "layer-by-layer over the whole
    sequence").

    Returns:
        (new_states, top_layer_output)
    """
    new_states = []
    layer_input = x
    for layer_params, h in zip(layers, states):
        new_h = gru_cell_apply(layer_params, layer_input, h)
        new_states.append(new_h)
        layer_input = new_h
    return tuple(new_states), layer_input


def run_stacked_gru(
    layers: Sequence[GRULayerParams],
    x_seq: jnp.ndarray,
    seq_len: Optional[jnp.ndarray] = None,
    init: Optional[Sequence[jnp.ndarray]] = None,
):
    """Run a stack of GRU layers over a full ``[batch, time, in_dim]`` sequence.

    Matches ``tf.nn.dynamic_rnn(..., sequence_length=seq_len)`` semantics: once a
    sequence's true length is exceeded, its per-layer state stops updating (frozen
    at its last valid value) -- this is what the encoder relies on to extract each
    sequence's correct final state regardless of padding.

    Args:
        layers: one ``GRULayerParams`` per stacked layer.
        x_seq: ``[batch, time, in_dim]`` input sequence (already embedded).
        seq_len: optional int32 ``[batch]`` true lengths. If ``None``, every
            timestep updates state unconditionally (used for decoding, where the
            caller handles stopping/masking itself).
        init: optional initial per-layer states; zeros if ``None``.

    Returns:
        final_states: tuple of ``[batch, hidden_i]`` per-layer final states.
        outputs: ``[batch, time, hidden_last]`` top-layer output at every timestep
            (timesteps beyond a sequence's true length are zeroed when ``seq_len``
            is given).
    """
    batch_size = x_seq.shape[0]
    if init is None:
        init = init_states(layers, batch_size, dtype=x_seq.dtype)

    x_seq_t = jnp.swapaxes(x_seq, 0, 1)  # [time, batch, in_dim]

    if seq_len is not None:
        seq_len = jnp.asarray(seq_len)
        time_idx = jnp.arange(x_seq_t.shape[0])

        def scan_fn(states, elem):
            x_t, t = elem
            new_states, top_out = stacked_gru_step(layers, x_t, states)
            mask = (t < seq_len)[:, None]
            new_states = tuple(
                jnp.where(mask, ns, s) for ns, s in zip(new_states, states)
            )
            top_out = jnp.where(mask, top_out, jnp.zeros_like(top_out))
            return new_states, top_out

        final_states, outputs_t = jax.lax.scan(scan_fn, tuple(init), (x_seq_t, time_idx))
    else:

        def scan_fn(states, x_t):
            new_states, top_out = stacked_gru_step(layers, x_t, states)
            return new_states, top_out

        final_states, outputs_t = jax.lax.scan(scan_fn, tuple(init), x_seq_t)

    outputs = jnp.swapaxes(outputs_t, 0, 1)  # [batch, time, hidden_last]
    return final_states, outputs
