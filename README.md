# jax-cddd 🧪

A JAX port of **CDDD** (Continuous and Data-Driven Descriptors) — a SMILES
autoencoder that produces a fixed-length, continuous molecular descriptor by
translating between equivalent SMILES representations of the same molecule.

> Winter, R., Montanari, F., Noé, F. & Clevert, D.-A. *Learning continuous and
> data-driven molecular descriptors by translating equivalent chemical
> representations*, Chem. Sci., 2019.

This project reimplements the pretrained `default_model` (encoder + decoder) in
JAX/Flax, loading the original TensorFlow 1.x checkpoint weights unmodified. It
is validated against an independent reference recomputation of the original
model to within `1e-4` numerical tolerance — see [Validation](#-validation).

Only inference (embedding and reconstruction) is in scope. Training and the
auxiliary QSAR property head are not ported.

## ✨ What you get

- `seq_to_emb`: SMILES → 512-dim descriptor (the encoder)
- `emb_to_seq`: 512-dim descriptor → reconstructed SMILES (the decoder, greedy
  or beam search)
- A memory-efficient batching path for embedding/reconstructing millions of
  molecules without exhausting GPU memory

## 📦 Installation

Two environments are used: a main runtime environment (JAX/Flax/RDKit) and a
throwaway environment used only once, to convert the original TensorFlow
checkpoint into a JAX-native weight file.

```bash
# Main runtime environment
micromamba create -n jax-cddd python=3.11 pip -c conda-forge -y
micromamba run -n jax-cddd pip install -e .

# One-off conversion environment (CPU TensorFlow only)
micromamba create -n jax-cddd-convert python=3.11 pip -c conda-forge -y
micromamba run -n jax-cddd-convert pip install -e ".[convert]"
```

Then convert the pretrained checkpoint once (needs `default_model.zip`, the
original TF1 checkpoint, placed under `src/jax_cddd/_cddd_ref_/`):

```bash
micromamba run -n jax-cddd-convert python scripts/convert_checkpoint.py
```

This writes `src/jax_cddd/data/default_model_params.npz` (~200 MB, gitignored —
convert once locally, it isn't downloaded or committed).

## 🚀 Quickstart

```python
from jax_cddd.inference import CDDDModel

model = CDDDModel()  # loads the converted default_model weights

# Embed
embedding = model.seq_to_emb("CC(=O)Oc1ccccc1C(=O)O")  # aspirin -> [512] array

# Reconstruct
smiles = model.emb_to_seq(embedding)  # beam search, returns a single string
print(smiles)  # "CC(=O)Oc1ccccc1C(=O)O"
```

`CDDDModel` loads once and can be reused for as many calls as you like — hold
onto the instance rather than re-creating it per molecule.

## 📚 Batches

Both methods accept lists directly and are the right choice for anything from
a handful up to a few thousand molecules at once:

```python
smiles_list = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]

embeddings = model.seq_to_emb(smiles_list)          # [n, 512] array
reconstructed = model.emb_to_seq(embeddings)         # list[str]

# Top-k hypotheses per molecule instead of just the best one
top3 = model.emb_to_seq(embeddings, beam_width=10, num_top=3)  # list[list[str]]
```

Two knobs matter for quality/speed:
- `beam_width` (default `10`): higher explores more reconstruction hypotheses;
  `beam_width=1` degenerates to plain greedy decoding and is fastest.
- `max_len` (default `1000`): decoding step cap; lower it if you know your
  molecules are small, to save time.

## ⚡ Large-scale, memory-efficient batching

A batch pads every sequence up to the length of its longest member, and beam
search allocates `batch × beam_width × max_len` worth of state. Passing an
entire multi-million-molecule dataset as one batch will exhaust GPU memory long
before it finishes. For large datasets, **chunk the work** and, ideally,
**sort by length first** so each chunk pads efficiently instead of wasting
compute on one long outlier per batch:

```python
import numpy as np

def embed_large(model, smiles_list, chunk_size=512):
    """Memory-efficient embedding for large SMILES collections."""
    order = np.argsort([len(s) for s in smiles_list])  # group similar lengths
    sorted_smiles = [smiles_list[i] for i in order]

    embeddings = np.empty((len(smiles_list), 512), dtype=np.float32)
    for start in range(0, len(sorted_smiles), chunk_size):
        chunk = sorted_smiles[start : start + chunk_size]
        chunk_idx = order[start : start + chunk_size]
        embeddings[chunk_idx] = model.seq_to_emb(chunk)
    return embeddings


def reconstruct_large(model, embeddings, chunk_size=256, beam_width=10):
    """Memory-efficient reconstruction; decoding is heavier than encoding, so
    use a smaller chunk_size here than for embed_large."""
    results = []
    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        results.extend(model.emb_to_seq(chunk, beam_width=beam_width))
    return results
```

Practical tips:
- 🔹 **Tune `chunk_size` to your GPU.** Start around `256`–`1024` for
  embedding and half that for reconstruction (beam search is more
  memory-hungry); increase until you're close to, but under, your GPU's
  memory limit.
- 🔹 **Sort by length before chunking.** Padding cost is set by the longest
  sequence in a chunk — grouping similar lengths together avoids paying for
  padding on every short molecule in a chunk that also contains one long one.
- 🔹 **Prefer greedy (`beam_width=1`) for very large reconstruction runs** if
  top-1 accuracy is good enough — beam search is several times more
  expensive in both memory and time.
- 🔹 **Write results incrementally** (e.g. append each chunk to disk) rather
  than accumulating everything in a Python list, if the dataset doesn't fit
  comfortably in host RAM either.
- 🔹 **Cap GPU preallocation on small GPUs** via
  `XLA_PYTHON_CLIENT_MEM_FRACTION` if you share the GPU with other processes
  (the package already requests a conservative `0.3` by default).

## ✅ Validation

The JAX encoder/decoder is checked against reference activations computed
directly from the original checkpoint's raw tensors (an independent NumPy
reimplementation of the TF1 `GRUCell` math, not a copy of this package's code):

```bash
micromamba run -n jax-cddd python -m pytest tests/
```

On 30 diverse molecules: max encoder error `1.2e-6`, max decoder logit error
`1.9e-5` (target was `1e-4`), and exact-argmax agreement on every token. A
round-trip smoke test on real drugs (aspirin, caffeine, ibuprofen, ...)
reconstructs correctly in the large majority of cases, consistent with the
original paper's reported accuracy.

```bash
micromamba run -n jax-cddd python scripts/reconstruct_smoke_test.py
```

## 🗂️ Project layout

```
src/jax_cddd/
    vocab.py        # SMILES tokenizer + vocabulary
    gru.py          # GRU cell matching the original TF1 math exactly
    modules.py       # encoder / decoder forward passes
    decoding.py      # greedy + beam search decoding
    params.py        # weight pytree, .npz (de)serialization, model loading
    inference.py     # CDDDModel: the public API
    _legacy_tf1/     # original TF1 reference code, kept for history
scripts/
    convert_checkpoint.py             # TF1 checkpoint -> JAX weights (one-off)
    validate_against_tf1_reference.py # generates the fidelity test fixtures
    reconstruct_smoke_test.py         # end-to-end embed/reconstruct demo
tests/
```

## 📄 License & citation

The original CDDD implementation is MIT-licensed (© 2018 Jan Robin Winter). If
you use this in published work, please cite the original paper above.
