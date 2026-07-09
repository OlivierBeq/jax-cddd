"""Regex-based SMILES character tokenizer and vocabulary, ported from the original
CDDD ``input_pipeline.py``/``models.py`` (``BaseModel.idx_to_char``).

The default model uses a single shared vocabulary for both the encoder input and the
decoder input/output, loaded from ``indices_char.npy`` (a ``dict[str, int]`` mapping
each token to its integer id).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from jax_cddd._download import download_file

REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'

START_TOKEN = "<s>"
STOP_TOKEN = "</s>"

_DEFAULT_VOCAB_FILENAME = "indices_char.npy"

# Can be overridden without a code change via the JAX_CDDD_VOCAB_URL
# environment variable (which takes precedence over this constant).
_DEFAULT_VOCAB_URL: Optional[str] = (
    "https://github.com/OlivierBeq/jax-cddd/releases/download/model_weights/indices_char.npy"
)
_DEFAULT_VOCAB_SHA256: Optional[str] = (
    "99497294eb161b5126fc84ff2f7267400d3299ccc83ffbeb8febc3020b1f7ab7"
)

_VOCAB_URL_ENV_VAR = "JAX_CDDD_VOCAB_URL"


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional token<->id mapping plus the SMILES regex tokenizer."""

    char_to_idx: dict
    idx_to_char: dict

    @property
    def size(self) -> int:
        return len(self.char_to_idx)

    @property
    def start_id(self) -> int:
        return self.char_to_idx[START_TOKEN]

    @property
    def stop_id(self) -> int:
        return self.char_to_idx[STOP_TOKEN]

    @classmethod
    def from_npy(cls, path) -> "Vocabulary":
        # The .npy file stores the raw {index: char} mapping (hence "indices_char");
        # the original code loads it and inverts it to get a {char: index} vocabulary.
        raw_idx_to_char = np.load(path, allow_pickle=True).item()
        idx_to_char = {int(k): str(v) for k, v in raw_idx_to_char.items()}
        char_to_idx = {v: k for k, v in idx_to_char.items()}
        return cls(char_to_idx=char_to_idx, idx_to_char=idx_to_char)

    @classmethod
    def default(cls) -> "Vocabulary":
        return cls.from_npy(download_default_vocab())

    def tokenize(self, smiles: str) -> list:
        return re.findall(REGEX_SML, smiles)

    def encode(self, smiles: str, add_start: bool = True, add_stop: bool = True) -> list:
        """Tokenize a SMILES string into a list of token ids, optionally wrapped with
        the ``<s>``/``</s>`` start/stop tokens (as done for every sequence in the
        original input pipeline)."""
        ids = [self.char_to_idx[token] for token in self.tokenize(smiles)]
        if add_start:
            ids = [self.start_id] + ids
        if add_stop:
            ids = ids + [self.stop_id]
        return ids

    def decode(self, ids) -> str:
        """Convert a sequence of token ids back to a SMILES string, stripping the
        start/stop tokens and any ``-1`` padding sentinel (matches
        ``BaseModel.idx_to_char``)."""
        chars = []
        for i in ids:
            i = int(i)
            if i in (-1, self.start_id, self.stop_id):
                continue
            chars.append(self.idx_to_char[i])
        return "".join(chars)


def default_vocab_path() -> Path:
    return Path(__file__).parent / "data" / _DEFAULT_VOCAB_FILENAME


def download_default_vocab(
    url: Optional[str] = None,
    dest: Optional[Union[str, Path]] = None,
    expected_sha256: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Download the vocabulary (``indices_char.npy``) from the GitHub release
    and place it at ``dest``.

    A no-op if ``dest`` already exists (unless ``force=True``), so this is safe
    to call unconditionally on every ``Vocabulary.default()`` -- which is
    exactly what it does.

    Args:
        url: Overrides the download URL. Defaults to the
            ``JAX_CDDD_VOCAB_URL`` environment variable, then the
            ``_DEFAULT_VOCAB_URL`` constant in this module.
        dest: Overrides the destination path. Defaults to
            ``default_vocab_path()``.
        expected_sha256: Overrides the expected checksum (defaults to
            ``_DEFAULT_VOCAB_SHA256``); verified after download, raising
            ``RuntimeError`` on mismatch (the bad download is removed, not left
            behind).
        force: Re-download even if ``dest`` already exists.

    Returns:
        The path the vocabulary is available at.
    """
    dest = Path(dest) if dest is not None else default_vocab_path()
    resolved_url = url or os.environ.get(_VOCAB_URL_ENV_VAR) or _DEFAULT_VOCAB_URL
    if not dest.exists() and not resolved_url:
        raise RuntimeError(
            f"No vocabulary file found at {dest}, and no download URL is "
            f"configured. Set the {_VOCAB_URL_ENV_VAR} environment variable or "
            "pass url=..."
        )
    return download_file(
        resolved_url,
        dest,
        expected_sha256=expected_sha256 if expected_sha256 is not None else _DEFAULT_VOCAB_SHA256,
        force=force,
        label="CDDD vocabulary",
    )


#: Fixed set of sequence-length buckets used by ``encode_batch(..., pad_to_bucket=True)``.
#: Padding to the smallest bucket that fits, rather than to each batch's exact
#: max length, bounds the number of distinct (batch, time) shapes that
#: jax_cddd.modules.encode (jit-compiled) ever gets called with -- so, e.g.,
#: repeatedly encoding single molecules of different lengths reuses one of a
#: handful of cached compiled executables instead of recompiling per call. Covers
#: SMILES from tiny to very large (drug-like molecules are typically <100 tokens).
LENGTH_BUCKETS = (16, 32, 48, 64, 96, 128, 160, 200, 256, 320, 400, 512)


def bucket_length(n: int, buckets=LENGTH_BUCKETS) -> int:
    """Smallest bucket boundary that is ``>= n``, or ``n`` itself if it exceeds
    every bucket (falls back to an exact, one-off shape for unusually long
    inputs rather than raising)."""
    for b in buckets:
        if n <= b:
            return b
    return n


def encode_batch(smiles_list, vocab: Vocabulary, pad_to_bucket: bool = False):
    """Tokenize and right-pad a batch of SMILES strings.

    Args:
        pad_to_bucket: if True, pad to ``bucket_length(max_len)`` instead of
            the batch's exact max length -- see ``LENGTH_BUCKETS``.

    Returns:
        ids: int32 array [batch, padded_len], right-padded with
            ``vocab.stop_id`` (matching the original input pipeline's padding
            value).
        lengths: int32 array [batch] with each sequence's true length
            (including the ``<s>``/``</s>`` tokens; NOT the padded length).
    """
    encoded = [vocab.encode(s) for s in smiles_list]
    lengths = np.array([len(e) for e in encoded], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    padded_len = bucket_length(max_len) if pad_to_bucket else max_len
    ids = np.full((len(encoded), padded_len), vocab.stop_id, dtype=np.int32)
    for row, seq in enumerate(encoded):
        ids[row, : len(seq)] = seq
    return ids, lengths
