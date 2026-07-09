"""Encoder/Decoder forward passes for CDDD.

Implemented as plain functions over the ``jax_cddd.params.CDDDParams`` pytree
(built on the hand-written GRU math in ``jax_cddd.gru``) rather than as
``flax.linen.Module`` subclasses: every weight used here always comes from
converting the pretrained TF1 checkpoint (``jax_cddd.params.load_default_model``),
never from flax-managed init/training, so flax's parameter-management machinery
(``self.param``, ``nn.scan`` broadcasting, etc.) would add indirection without
functional benefit for this project's current inference-only scope. flax/optax
remain project dependencies for future training work.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from jax_cddd.gru import run_stacked_gru, stacked_gru_step
from jax_cddd.params import CDDDParams


def embed_ids(embedding: jnp.ndarray, ids: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(embedding, ids, axis=0)


def encode(params: CDDDParams, input_ids: jnp.ndarray, seq_len: jnp.ndarray) -> jnp.ndarray:
    """SMILES token ids ``[batch, time]`` (already ``<s>``/``</s>``-wrapped and
    padded) -> ``[batch, emb_size]`` CDDD descriptor.

    Matches ``GRUSeq2Seq._encoder``: char-embed -> 3-layer stacked GRU -> concat
    final per-layer states -> dense -> tanh.
    """
    x = embed_ids(params.embedding, input_ids)
    final_states, _ = run_stacked_gru(params.encoder.layers, x, seq_len=seq_len)
    concat = jnp.concatenate(final_states, axis=-1)
    bottleneck = concat @ params.encoder.bottleneck_kernel + params.encoder.bottleneck_bias
    return jnp.tanh(bottleneck)


def decoder_initial_states(params: CDDDParams, descriptor: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """``[batch, emb_size]`` descriptor -> per-layer initial GRU states for the
    decoder (dense projection then split, matching ``GRUSeq2Seq._decoder``)."""
    flat = descriptor @ params.decoder.init_state_kernel + params.decoder.init_state_bias
    sizes = [layer.hidden_size for layer in params.decoder.layers]
    split_points = [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)]
    return tuple(jnp.split(flat, split_points, axis=-1))


def decode_teacher_forced(
    params: CDDDParams,
    descriptor: jnp.ndarray,
    target_ids: jnp.ndarray,
    seq_len: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Teacher-forced decoding, for validation against the TF1 reference (this is
    what ``TrainingHelper``/``BasicDecoder`` computes, without the loss/argmax).

    ``target_ids`` is the full ground-truth sequence *including* the leading
    ``<s>`` (i.e. not shifted) -- the input at step ``t`` is ``target_ids[:, t]``,
    and the returned ``logits[:, t]`` is this model's prediction for
    ``target_ids[:, t + 1]`` (so callers comparing against ground truth should
    compare ``logits[:, :-1]`` against ``target_ids[:, 1:]``).

    Returns:
        logits: ``[batch, time, vocab_size]``.
    """
    init = decoder_initial_states(params, descriptor)
    x = embed_ids(params.embedding, target_ids)
    _, outputs = run_stacked_gru(params.decoder.layers, x, seq_len=seq_len, init=init)
    return outputs @ params.decoder.output_proj_kernel


def decode_step(
    params: CDDDParams,
    states: Tuple[jnp.ndarray, ...],
    prev_ids: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, ...], jnp.ndarray]:
    """One autoregressive decoder step (used by greedy/beam-search decoding).

    Args:
        states: per-layer decoder states, e.g. from ``decoder_initial_states``.
        prev_ids: ``[batch]`` previous token ids (``<s>`` for the first step).

    Returns:
        (new_states, logits): ``logits`` is ``[batch, vocab_size]``.
    """
    x = embed_ids(params.embedding, prev_ids)
    new_states, top_out = stacked_gru_step(params.decoder.layers, x, states)
    logits = top_out @ params.decoder.output_proj_kernel
    return new_states, logits
