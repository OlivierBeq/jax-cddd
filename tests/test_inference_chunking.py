"""CDDDModel chunks large inputs internally (see inference.py) -- these tests
pin down that chunking is transparent: results (modulo negligible float noise
from different batch compositions) and input order are preserved regardless
of chunk_size.
"""
import os

import numpy as np
import pytest

from jax_cddd.inference import CDDDModel
from jax_cddd.params import default_params_path

requires_default_model = pytest.mark.skipif(
    not os.path.exists(default_params_path()),
    reason="default_model_params.npz not found; run scripts/convert_checkpoint.py first",
)

SMILES_LIST = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C",
    "CCN(CC)CC",
    "c1ccncc1",
    "CC(=O)O",
]


@pytest.fixture(scope="module")
def model():
    return CDDDModel()


@requires_default_model
def test_seq_to_emb_chunking_preserves_order_and_values(model):
    emb_unchunked = model.seq_to_emb(SMILES_LIST, chunk_size=1000)
    emb_chunked = model.seq_to_emb(SMILES_LIST, chunk_size=2)

    assert emb_unchunked.shape == emb_chunked.shape == (len(SMILES_LIST), 512)
    np.testing.assert_allclose(emb_unchunked, emb_chunked, atol=1e-4, rtol=1e-4)


@requires_default_model
def test_emb_to_seq_chunking_preserves_order_and_values(model):
    embeddings = model.seq_to_emb(SMILES_LIST)

    recon_unchunked = model.emb_to_seq(embeddings, beam_width=10, chunk_size=1000)
    recon_chunked = model.emb_to_seq(embeddings, beam_width=10, chunk_size=2)

    assert recon_unchunked == recon_chunked


@requires_default_model
def test_emb_to_seq_chunking_with_num_top(model):
    embeddings = model.seq_to_emb(SMILES_LIST[:3])

    top3_unchunked = model.emb_to_seq(embeddings, beam_width=10, num_top=3, chunk_size=1000)
    top3_chunked = model.emb_to_seq(embeddings, beam_width=10, num_top=3, chunk_size=1)

    assert top3_unchunked == top3_chunked


@requires_default_model
def test_single_input_bypasses_chunking_machinery(model):
    emb = model.seq_to_emb("CCO")
    assert emb.shape == (512,)
    smiles = model.emb_to_seq(emb)
    assert isinstance(smiles, str)
    assert smiles == "CCO"


@requires_default_model
def test_chunk_size_smaller_than_one_still_covers_all_inputs(model):
    # chunk_size=1 is the most extreme chunking short of degenerate input.
    emb = model.seq_to_emb(SMILES_LIST, chunk_size=1)
    assert emb.shape == (len(SMILES_LIST), 512)
    recon = model.emb_to_seq(emb, chunk_size=1)
    assert len(recon) == len(SMILES_LIST)
