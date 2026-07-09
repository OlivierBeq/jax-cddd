"""High-level inference API for CDDD: SMILES <-> descriptor.

Mirrors the original ``cddd.inference.InferenceModel``'s ``seq_to_emb``/
``emb_to_seq`` methods, as a single JAX-backed object holding the pretrained
weights and vocabulary.

Both methods process arbitrarily large inputs in memory-bounded chunks
internally (see ``chunk_size`` on each) -- callers never need to chunk their
own inputs.

``encode``/``greedy_decode``/``beam_search_decode`` are jit-compiled (see
``modules.py``/``decoding.py``), so their *first* call for a given input shape
pays a one-time tracing+compilation cost (hundreds of ms to a few seconds) --
every later call with that same shape reuses the cached compiled executable
and is fast (single-digit to a few tens of ms). ``CDDDModel`` hides this by
warming up the shapes a single-molecule workload actually hits during
``__init__`` (see ``warmup=``), so a freshly constructed model's first real
``seq_to_emb``/``emb_to_seq`` call is already fast.
"""
from __future__ import annotations

from typing import List, Optional, Union

import jax.numpy as jnp
import numpy as np

from jax_cddd.decoding import (
    DEFAULT_MAX_LEN,
    beam_search_decode,
    beam_search_decode_smiles,
    greedy_decode_smiles,
)
from jax_cddd.modules import encode
from jax_cddd.params import EMB_SIZE, CDDDParams, load_default_model
from jax_cddd.vocab import Vocabulary, encode_batch

# Beam search decoding holds batch x beam_width x max_len worth of state, so it's
# considerably more memory-hungry per-molecule than encoding -- hence the smaller
# default chunk size.
DEFAULT_ENCODE_CHUNK_SIZE = 512
DEFAULT_DECODE_CHUNK_SIZE = 256

# Single-molecule (batch=1) length buckets warmed up eagerly on construction --
# covers the large majority of real-world SMILES (drug-like molecules are
# typically well under 100 tokens); a molecule falling outside these still
# works correctly, it just pays a one-time compile on first use of its bucket.
_WARMUP_ENCODE_BUCKETS = (16, 32, 48, 64)


class CDDDModel:
    """Loads the pretrained default CDDD model and exposes encode/decode.

    Example::

        model = CDDDModel()
        emb = model.seq_to_emb(["CCO", "c1ccccc1"])
        smiles = model.emb_to_seq(emb)
    """

    def __init__(
        self,
        params: Optional[CDDDParams] = None,
        vocab: Optional[Vocabulary] = None,
        warmup: bool = True,
    ):
        self.params = params if params is not None else load_default_model()
        self.vocab = vocab if vocab is not None else Vocabulary.default()
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Eagerly triggers compilation for the shapes a single-molecule
        workload hits, so the first real call a user makes is already fast
        rather than paying the compile cost then. See module docstring."""
        for bucket in _WARMUP_ENCODE_BUCKETS:
            ids = jnp.zeros((1, bucket), dtype=jnp.int32)
            lengths = jnp.asarray([bucket], dtype=jnp.int32)
            encode(self.params, ids, lengths).block_until_ready()

        # beam_search_decode() converts its result to numpy internally, which
        # already blocks until the (async-dispatched) device computation
        # finishes -- no separate block_until_ready() needed here.
        dummy_emb = jnp.zeros((1, EMB_SIZE), dtype=jnp.float32)
        beam_search_decode(self.params, dummy_emb, self.vocab, beam_width=10, max_len=DEFAULT_MAX_LEN)

    def seq_to_emb(
        self, seq: Union[str, List[str]], chunk_size: int = DEFAULT_ENCODE_CHUNK_SIZE
    ) -> np.ndarray:
        """Encode one or more SMILES strings into their CDDD descriptor(s).

        Internally processes the input in chunks of at most ``chunk_size``
        molecules (sorted by length first, to minimize the compute wasted on
        padding short sequences up to one long outlier's length), so this is
        safe to call directly on arbitrarily large collections -- lower
        ``chunk_size`` if you hit GPU memory limits, raise it for more
        throughput on a bigger GPU.

        Returns a ``[emb_size]`` array for a single string, or ``[n, emb_size]``
        for a list.
        """
        single = isinstance(seq, str)
        seq_list = [seq] if single else list(seq)
        n = len(seq_list)

        order = np.argsort([len(s) for s in seq_list])
        sorted_seq = [seq_list[i] for i in order]

        emb = np.empty((n, EMB_SIZE), dtype=np.float32)
        for start in range(0, n, chunk_size):
            chunk = sorted_seq[start : start + chunk_size]
            chunk_idx = order[start : start + chunk_size]
            ids, lengths = encode_batch(chunk, self.vocab, pad_to_bucket=True)
            emb[chunk_idx] = np.asarray(encode(self.params, jnp.asarray(ids), jnp.asarray(lengths)))

        return emb[0] if single else emb

    def emb_to_seq(
        self,
        embedding: np.ndarray,
        beam_width: int = 10,
        num_top: int = 1,
        max_len: int = DEFAULT_MAX_LEN,
        chunk_size: int = DEFAULT_DECODE_CHUNK_SIZE,
    ):
        """Decode one or more CDDD descriptors back into SMILES string(s).

        Internally processes the input in chunks of at most ``chunk_size``
        embeddings, so this is safe to call directly on arbitrarily large
        collections -- lower ``chunk_size`` if you hit GPU memory limits (beam
        search holds ``chunk_size x beam_width x max_len`` worth of state),
        raise it for more throughput on a bigger GPU.

        Args:
            embedding: ``[emb_size]`` or ``[n, emb_size]`` array.
            beam_width: beam search width (``1`` runs plain greedy decoding).
            num_top: number of top hypotheses to return per input.
            max_len: maximum number of decoding steps.
            chunk_size: maximum number of embeddings decoded per internal batch.

        Returns a single string (if given one embedding and ``num_top == 1``), a
        list of strings, or a list of lists of strings (if ``num_top > 1``).
        """
        embedding = np.asarray(embedding)
        single = embedding.ndim == 1
        if single:
            embedding = embedding[None, :]
        n = embedding.shape[0]

        results = []
        for start in range(0, n, chunk_size):
            chunk = embedding[start : start + chunk_size]
            if beam_width == 1:
                top1 = greedy_decode_smiles(self.params, jnp.asarray(chunk), self.vocab, max_len=max_len)
                results.extend([s] for s in top1)
            else:
                results.extend(
                    beam_search_decode_smiles(
                        self.params,
                        jnp.asarray(chunk),
                        self.vocab,
                        beam_width=beam_width,
                        max_len=max_len,
                        num_top=num_top,
                    )
                )

        if num_top == 1:
            results = [hyps[0] for hyps in results]
        return results[0] if single else results
