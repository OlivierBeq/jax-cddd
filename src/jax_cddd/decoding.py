"""Greedy and beam-search decoding for the CDDD decoder (descriptor -> SMILES),
matching the original TF1 ``BeamSearchDecoder``'s behavior with
``length_penalty_weight=0.0`` (the ``default_model``'s configuration).

The actual numeric work (``_greedy_decode_core``/``_beam_search_decode_core``)
is jit-compiled and takes only arrays/static ints -- never the ``Vocabulary``
object itself (a plain Python dataclass holding dicts, which isn't a valid jit
argument) -- so repeated calls with the same ``(batch, beam_width, max_len)``
shape reuse a cached compiled executable instead of re-tracing/recompiling
and re-dispatching each op eagerly.
"""
from __future__ import annotations

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_cddd.modules import decode_step, decoder_initial_states
from jax_cddd.params import CDDDParams
from jax_cddd.vocab import Vocabulary

NEG_INF = -1e9

# Real SMILES are essentially never this long (drug-like molecules are
# typically well under 100 tokens); this is a generous cap chosen to keep the
# default decode latency low while still being large enough that legitimate
# molecules are never truncated. Override max_len= directly for unusually
# large inputs (e.g. peptides/oligomers).
DEFAULT_MAX_LEN = 140


def truncate_at_stop(ids_row: np.ndarray, stop_id: int) -> np.ndarray:
    """Cut a decoded id sequence at (and including) its first stop-token
    occurrence, discarding anything generated after. Greedy decoding here has no
    built-in early stopping (it always runs to ``max_len``), so this must be
    applied as a post-processing step before turning ids back into a string."""
    stop_positions = np.where(ids_row == stop_id)[0]
    if len(stop_positions) == 0:
        return ids_row
    return ids_row[: stop_positions[0] + 1]


@partial(jax.jit, static_argnames=("max_len",))
def _greedy_decode_core(
    params: CDDDParams, descriptor: jnp.ndarray, start_id: jnp.ndarray, max_len: int
) -> jnp.ndarray:
    batch = descriptor.shape[0]
    states = decoder_initial_states(params, descriptor)
    prev_ids = jnp.full((batch,), start_id, dtype=jnp.int32)

    def step(carry, _):
        states, prev_ids = carry
        new_states, logits = decode_step(params, states, prev_ids)
        next_ids = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return (new_states, next_ids), next_ids

    _, all_ids = jax.lax.scan(step, (states, prev_ids), xs=None, length=max_len)
    return jnp.swapaxes(all_ids, 0, 1)  # [batch, max_len]


def greedy_decode(
    params: CDDDParams, descriptor: jnp.ndarray, vocab: Vocabulary, max_len: int = DEFAULT_MAX_LEN
) -> np.ndarray:
    """``[batch, emb_size]`` descriptor -> ``[batch, max_len]`` int32 token ids
    (argmax at every step; no in-loop early stopping -- see ``truncate_at_stop``).
    """
    ids = _greedy_decode_core(params, descriptor, jnp.int32(vocab.start_id), max_len)
    return np.asarray(ids)


def greedy_decode_smiles(
    params: CDDDParams, descriptor: jnp.ndarray, vocab: Vocabulary, max_len: int = DEFAULT_MAX_LEN
) -> List[str]:
    ids = greedy_decode(params, descriptor, vocab, max_len=max_len)
    return [vocab.decode(truncate_at_stop(row, vocab.stop_id)) for row in ids]


def _gather_beam(x: jnp.ndarray, beam_idx: jnp.ndarray) -> jnp.ndarray:
    """Reorder axis 1 (the beam axis) of a ``[B, K, ...]`` array according to
    ``beam_idx`` (``[B, K]``, values in ``[0, K)``)."""
    idx = beam_idx.reshape(beam_idx.shape + (1,) * (x.ndim - 2))
    idx = jnp.broadcast_to(idx, beam_idx.shape + x.shape[2:])
    return jnp.take_along_axis(x, idx, axis=1)


@partial(jax.jit, static_argnames=("beam_width", "max_len"))
def _beam_search_decode_core(
    params: CDDDParams,
    descriptor: jnp.ndarray,
    start_id: jnp.ndarray,
    stop_id: jnp.ndarray,
    beam_width: int,
    max_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    B = descriptor.shape[0]
    K = beam_width

    init_states = decoder_initial_states(params, descriptor)
    states = tuple(jnp.broadcast_to(s[:, None, :], (B, K, s.shape[-1])) for s in init_states)

    def flatten(x):
        return x.reshape((B * K,) + x.shape[2:])

    def unflatten(x):
        return x.reshape((B, K) + x.shape[1:])

    prev_ids = jnp.full((B, K), start_id, dtype=jnp.int32)
    # Only beam 0 is "real" at t=0 (all K beams start identical); mask the rest
    # to -inf so the first expansion doesn't produce K copies of the same beam.
    cum_logprob = jnp.where(jnp.arange(K)[None, :] == 0, 0.0, NEG_INF)
    cum_logprob = jnp.broadcast_to(cum_logprob, (B, K)).astype(jnp.float32)
    finished = jnp.zeros((B, K), dtype=bool)
    seqs = jnp.full((B, K, max_len), stop_id, dtype=jnp.int32)

    def step(carry, t):
        states, prev_ids, cum_logprob, finished, seqs = carry
        flat_states = tuple(flatten(s) for s in states)
        new_flat_states, logits = decode_step(params, flat_states, flatten(prev_ids))
        logits = unflatten(logits)
        new_states = tuple(unflatten(s) for s in new_flat_states)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        vocab_size = log_probs.shape[-1]

        # Finished beams may only extend with another stop token, at zero
        # additional cost (keeps them "alive" without further penalizing them).
        forced = jnp.full_like(log_probs, NEG_INF).at[:, :, stop_id].set(0.0)
        log_probs = jnp.where(finished[:, :, None], forced, log_probs)

        candidate_scores = (cum_logprob[:, :, None] + log_probs).reshape(B, K * vocab_size)
        top_scores, top_idx = jax.lax.top_k(candidate_scores, K)
        beam_idx = top_idx // vocab_size
        token_idx = (top_idx % vocab_size).astype(jnp.int32)

        new_states = tuple(_gather_beam(s, beam_idx) for s in new_states)
        new_seqs = _gather_beam(seqs, beam_idx)
        new_finished = jnp.take_along_axis(finished, beam_idx, axis=1) | (token_idx == stop_id)
        new_seqs = new_seqs.at[:, :, t].set(token_idx)

        return (new_states, token_idx, top_scores, new_finished, new_seqs), None

    carry = (states, prev_ids, cum_logprob, finished, seqs)
    carry, _ = jax.lax.scan(step, carry, jnp.arange(max_len))
    _, _, final_cum_logprob, _, final_seqs = carry
    return final_seqs, final_cum_logprob


def beam_search_decode(
    params: CDDDParams,
    descriptor: jnp.ndarray,
    vocab: Vocabulary,
    beam_width: int = 10,
    max_len: int = DEFAULT_MAX_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """``[batch, emb_size]`` descriptor -> beam-searched hypotheses.

    Returns:
        seqs: ``[batch, beam_width, max_len]`` int32, best-first (index 0 is the
            top hypothesis for each batch element).
        cum_logprob: ``[batch, beam_width]`` cumulative log-probabilities.

    ``length_penalty_weight`` is fixed at 0 (raw cumulative log-prob ranking),
    matching the original ``default_model``'s ``BeamSearchDecoder`` config.
    """
    seqs, cum_logprob = _beam_search_decode_core(
        params, descriptor, jnp.int32(vocab.start_id), jnp.int32(vocab.stop_id), beam_width, max_len
    )
    return np.asarray(seqs), np.asarray(cum_logprob)


def beam_search_decode_smiles(
    params: CDDDParams,
    descriptor: jnp.ndarray,
    vocab: Vocabulary,
    beam_width: int = 10,
    max_len: int = DEFAULT_MAX_LEN,
    num_top: int = 1,
) -> List[List[str]]:
    seqs, _ = beam_search_decode(params, descriptor, vocab, beam_width=beam_width, max_len=max_len)
    results = []
    for batch_row in seqs:
        hyps = [vocab.decode(truncate_at_stop(row, vocab.stop_id)) for row in batch_row[:num_top]]
        results.append(hyps)
    return results
