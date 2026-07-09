"""Regex-based SMILES character tokenizer and vocabulary, ported from the original
CDDD ``input_pipeline.py``/``models.py`` (``BaseModel.idx_to_char``).

The default model uses a single shared vocabulary for both the encoder input and the
decoder input/output, loaded from ``indices_char.npy`` (a ``dict[str, int]`` mapping
each token to its integer id).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGEX_SML = r'Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps]'

START_TOKEN = "<s>"
STOP_TOKEN = "</s>"

_DEFAULT_VOCAB_FILENAME = "indices_char.npy"


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
        # Plain path lookup rather than importlib.resources (whose .files() API
        # needs Python >=3.9): jax_cddd is always a local, non-zipped source
        # package, so this is simpler and also keeps this module importable
        # from the older Python required by the legacy-TF1 environment (see
        # scripts/run_original_model.py).
        path = Path(__file__).parent / "data" / _DEFAULT_VOCAB_FILENAME
        return cls.from_npy(path)

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


def encode_batch(smiles_list, vocab: Vocabulary):
    """Tokenize and right-pad a batch of SMILES strings.

    Returns:
        ids: int32 array [batch, max_len], right-padded with ``vocab.stop_id``
            (matching the original input pipeline's padding value).
        lengths: int32 array [batch] with each sequence's true length (including
            the ``<s>``/``</s>`` tokens).
    """
    encoded = [vocab.encode(s) for s in smiles_list]
    lengths = np.array([len(e) for e in encoded], dtype=np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    ids = np.full((len(encoded), max_len), vocab.stop_id, dtype=np.int32)
    for row, seq in enumerate(encoded):
        ids[row, : len(seq)] = seq
    return ids, lengths
