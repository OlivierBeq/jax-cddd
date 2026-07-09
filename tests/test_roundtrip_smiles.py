import os

import jax.numpy as jnp
import pytest
from rdkit import Chem

from jax_cddd.decoding import beam_search_decode_smiles, greedy_decode_smiles
from jax_cddd.modules import encode
from jax_cddd.params import default_params_path, load_default_model
from jax_cddd.vocab import Vocabulary, encode_batch

requires_default_model = pytest.mark.skipif(
    not os.path.exists(default_params_path()),
    reason="default_model_params.npz not found; run scripts/convert_checkpoint.py first",
)

CURATED_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
    "CC(=O)Nc1ccc(O)cc1",  # paracetamol
    "c1ccncc1",
    "CCN(CC)CC",
]


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.default()


@pytest.fixture(scope="module")
def params():
    return load_default_model()


@requires_default_model
def test_beam_width_1_matches_greedy(params, vocab):
    ids, lengths = encode_batch(CURATED_SMILES, vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))

    greedy_smiles = greedy_decode_smiles(params, descriptor, vocab, max_len=150)
    beam1_smiles = beam_search_decode_smiles(params, descriptor, vocab, beam_width=1, max_len=150)
    beam1_smiles = [hyps[0] for hyps in beam1_smiles]

    assert greedy_smiles == beam1_smiles


@requires_default_model
def test_roundtrip_reconstruction_accuracy(params, vocab):
    ids, lengths = encode_batch(CURATED_SMILES, vocab)
    descriptor = encode(params, jnp.asarray(ids), jnp.asarray(lengths))
    decoded = beam_search_decode_smiles(params, descriptor, vocab, beam_width=10, max_len=150)
    decoded = [hyps[0] for hyps in decoded]

    matches = 0
    for original, reconstructed in zip(CURATED_SMILES, decoded):
        expected_canonical = _canonical(original)
        got_canonical = _canonical(reconstructed)
        if expected_canonical is not None and got_canonical == expected_canonical:
            matches += 1
        print(f"{original!r} -> {reconstructed!r} (canonical match: {got_canonical == expected_canonical})")

    accuracy = matches / len(CURATED_SMILES)
    assert accuracy >= 0.7, f"round-trip accuracy too low: {accuracy:.0%} ({matches}/{len(CURATED_SMILES)})"
