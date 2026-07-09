import numpy as np
import pytest

from jax_cddd.vocab import Vocabulary, encode_batch


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.default()


def test_vocab_size(vocab):
    assert vocab.size == 40


def test_start_stop_ids(vocab):
    assert vocab.char_to_idx["<s>"] == vocab.start_id == 39
    assert vocab.char_to_idx["</s>"] == vocab.stop_id == 0


def test_tokenize_multichar_atoms(vocab):
    assert vocab.tokenize("ClCBr") == ["Cl", "C", "Br"]


def test_tokenize_brackets_and_aromatics(vocab):
    assert vocab.tokenize("c1ccccc1") == ["c", "1", "c", "c", "c", "c", "c", "1"]
    assert vocab.tokenize("[nH]") == ["[", "n", "H", "]"]


def test_encode_wraps_with_start_stop(vocab):
    ids = vocab.encode("CC")
    assert ids[0] == vocab.start_id
    assert ids[-1] == vocab.stop_id
    assert len(ids) == 4  # <s> C C </s>


def test_encode_decode_roundtrip(vocab):
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
    ids = vocab.encode(smiles)
    assert vocab.decode(ids) == smiles


def test_decode_strips_special_tokens(vocab):
    ids = [vocab.start_id, vocab.char_to_idx["C"], vocab.stop_id, -1]
    assert vocab.decode(ids) == "C"


def test_encode_batch_padding_and_lengths(vocab):
    smiles_list = ["C", "CCCC"]
    ids, lengths = encode_batch(smiles_list, vocab)
    assert ids.shape == (2, 6)  # max: <s> C C C C </s> = 6
    assert lengths.tolist() == [3, 6]
    # padding value is the stop id
    assert (ids[0, lengths[0]:] == vocab.stop_id).all()
