"""High-level inference API for CDDD: SMILES <-> descriptor.

Mirrors the original ``cddd.inference.InferenceModel``'s ``seq_to_emb``/
``emb_to_seq`` methods, as a single JAX-backed object holding the pretrained
weights and vocabulary.
"""
from __future__ import annotations

from typing import List, Optional, Union

import jax.numpy as jnp
import numpy as np

from jax_cddd.decoding import beam_search_decode_smiles, greedy_decode_smiles
from jax_cddd.modules import encode
from jax_cddd.params import CDDDParams, load_default_model
from jax_cddd.vocab import Vocabulary, encode_batch


class CDDDModel:
    """Loads the pretrained default CDDD model and exposes encode/decode.

    Example::

        model = CDDDModel()
        emb = model.seq_to_emb(["CCO", "c1ccccc1"])
        smiles = model.emb_to_seq(emb)
    """

    def __init__(self, params: Optional[CDDDParams] = None, vocab: Optional[Vocabulary] = None):
        self.params = params if params is not None else load_default_model()
        self.vocab = vocab if vocab is not None else Vocabulary.default()

    def seq_to_emb(self, seq: Union[str, List[str]]) -> np.ndarray:
        """Encode one or more SMILES strings into their CDDD descriptor(s).

        Returns a ``[emb_size]`` array for a single string, or ``[n, emb_size]``
        for a list.
        """
        single = isinstance(seq, str)
        seq_list = [seq] if single else list(seq)
        ids, lengths = encode_batch(seq_list, self.vocab)
        emb = np.asarray(encode(self.params, jnp.asarray(ids), jnp.asarray(lengths)))
        return emb[0] if single else emb

    def emb_to_seq(
        self,
        embedding: np.ndarray,
        beam_width: int = 10,
        num_top: int = 1,
        max_len: int = 1000,
    ):
        """Decode one or more CDDD descriptors back into SMILES string(s).

        Args:
            embedding: ``[emb_size]`` or ``[n, emb_size]`` array.
            beam_width: beam search width (``1`` runs plain greedy decoding).
            num_top: number of top hypotheses to return per input.
            max_len: maximum number of decoding steps.

        Returns a single string (if given one embedding and ``num_top == 1``), a
        list of strings, or a list of lists of strings (if ``num_top > 1``).
        """
        embedding = np.asarray(embedding)
        single = embedding.ndim == 1
        if single:
            embedding = embedding[None, :]

        if beam_width == 1:
            top1 = greedy_decode_smiles(self.params, jnp.asarray(embedding), self.vocab, max_len=max_len)
            results = [[s] for s in top1]
        else:
            results = beam_search_decode_smiles(
                self.params,
                jnp.asarray(embedding),
                self.vocab,
                beam_width=beam_width,
                max_len=max_len,
                num_top=num_top,
            )

        if num_top == 1:
            results = [hyps[0] for hyps in results]
        return results[0] if single else results
