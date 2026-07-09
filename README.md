# jax-cddd 🧪

A JAX port of **CDDD** (Continuous and Data-Driven Descriptors) — a SMILES
autoencoder that produces a fixed-length, continuous molecular descriptor by
translating between equivalent SMILES representations of the same molecule.

> Winter, R., Montanari, F., Noé, F. & Clevert, D.-A. *Learning continuous and
> data-driven molecular descriptors by translating equivalent chemical
> representations*, Chem. Sci., 2019.

This project reimplements the pretrained `default_model` (encoder + decoder) in
JAX/Flax, loading the original TensorFlow 1.x checkpoint weights unmodified. It
is validated both against an independent from-scratch recomputation of the
model's math *and* against the real, unmodified original code running under a
legacy TensorFlow 1.15, to within `1e-4` numerical tolerance — see
[Validation](#-validation).

Only inference (embedding and reconstruction) is in scope. Training and the
auxiliary QSAR property head are not ported.

## ✨ What you get

- `seq_to_emb`: SMILES → 512-dim descriptor (the encoder)
- `emb_to_seq`: 512-dim descriptor → reconstructed SMILES (the decoder, greedy
  or beam search)
- A memory-efficient batching path for embedding/reconstructing millions of
  molecules without exhausting GPU memory

## 📦 Installation

```bash
pip install jax-cddd
```

`jax-cddd` is [on PyPI](https://pypi.org/project/jax-cddd/). For development
(editable install from a clone), instead:

```bash
micromamba create -n jax-cddd python=3.11 pip -c conda-forge -y
micromamba run -n jax-cddd pip install -e .
```

### Getting the pretrained weights

Two data files — the converted weights
(`src/jax_cddd/data/default_model_params.npz`, ~200 MB) and the vocabulary
(`src/jax_cddd/data/indices_char.npy`) — are downloaded automatically, 
with checksum verification, the first time you instantiate `CDDDModel()`,
if they aren't already present locally — see [Quickstart](#-quickstart) below.
No action needed; they're fetched from this repo's [GitHub release](https://github.com/OlivierBeq/jax-cddd/releases/tag/model_weights).
To point at a different location instead (e.g. a fork's own release), override
via environment variables:

```bash
export JAX_CDDD_WEIGHTS_URL="https://.../default_model_params.npz"
export JAX_CDDD_VOCAB_URL="https://.../indices_char.npy"
```

## 🚀 Quickstart

```python
from jax_cddd.inference import CDDDModel

model = CDDDModel()  # downloads the converted default_model weights on first use

# Embed
embedding = model.seq_to_emb("CC(=O)Oc1ccccc1C(=O)O")  # aspirin -> [512] array

# Reconstruct
smiles = model.emb_to_seq(embedding)  # beam search, returns a single string
print(smiles)  # "CC(=O)Oc1ccccc1C(=O)O"
```

`CDDDModel` loads once and can be reused for as many calls as you like — hold
onto the instance rather than re-creating it per molecule.

## ⚡ Performance

Both `seq_to_emb` and `emb_to_seq` are JIT-compiled under the hood. `CDDDModel()`
eagerly warms up the shapes a single-molecule workload hits during
construction (a few extra seconds, on top of loading the weights), so a
freshly constructed model's first real call is already fast rather than
paying that compile cost then:

- `seq_to_emb`: ~10-30 ms per molecule.
- `emb_to_seq`: ~70-90 ms per molecule (default `beam_width=10`).

Molecules with an unusual shape (SMILES longer than ~64 tokens, or a
non-default `beam_width`/`max_len`) fall outside the warmed-up set and pay a
one-time compile the first time that particular shape is used — every
subsequent call with the same shape is fast again. Pass `CDDDModel(warmup=False)`
to skip warmup and get a near-instant construction instead, if you'd rather
pay the compile cost on first use than at startup.

## 📚 Batches, at any size

Both methods accept lists directly, from a handful of molecules up to entire
multi-million-molecule datasets — large inputs are automatically processed in
memory-bounded chunks under the hood (sorted by length first, to minimize the
compute wasted padding short sequences up to one long outlier's length), so
you never need to chunk your own input:

```python
smiles_list = [...]  # any size

embeddings = model.seq_to_emb(smiles_list)          # [n, 512] array
reconstructed = model.emb_to_seq(embeddings)         # list[str]

# Top-k hypotheses per molecule instead of just the best one
top3 = model.emb_to_seq(embeddings, beam_width=10, num_top=3)  # list[list[str]]
```

Knobs that matter for quality/speed/memory:
- `beam_width` (default `10`): higher explores more reconstruction hypotheses;
  `beam_width=1` degenerates to plain greedy decoding and is fastest.
- `max_len` (default `1000`): decoding step cap; lower it if you know your
  molecules are small, to save time.
- `chunk_size` (default `512` for `seq_to_emb`, `256` for `emb_to_seq`):
  maximum number of molecules/embeddings processed per internal batch. Lower
  it if you hit GPU memory limits (beam search holds
  `chunk_size × beam_width × max_len` worth of state, so it's more
  memory-hungry than encoding); raise it for more throughput on a bigger GPU.

A couple of tips beyond that for very large runs:
- 🔹 **Prefer greedy (`beam_width=1`)** if top-1 accuracy is good enough —
  beam search is several times more expensive in both memory and time.
- 🔹 **Cap GPU preallocation on small GPUs** via
  `XLA_PYTHON_CLIENT_MEM_FRACTION` if you share the GPU with other processes
  (the package already requests a conservative `0.3` by default).

## ✅ Validation

Fidelity is checked two independent ways:

1. **Against the math, from scratch.** `tests/test_against_tf1_reference.py`
   compares the JAX encoder/decoder to reference activations recomputed
   directly from the checkpoint's raw tensors, using a from-scratch NumPy
   reimplementation of the TF1 `GRUCell` formula (not a copy of this package's
   code). On 30 diverse molecules: max encoder error `1.2e-6`, max decoder
   logit error `1.9e-5` (target was `1e-4`), exact-argmax agreement on every
   token.
2. **Against the actual original code.** `tests/test_against_original_code.py`
   compares the JAX port to the *real, unmodified* original implementation —
   the genuine `tf.contrib.rnn.MultiRNNCell` / `tf.contrib.seq2seq.BeamSearchDecoder`
   graph, checkpoint restored via `tf.train.Saver`, executed under a legacy
   TensorFlow 1.15. On a handful of real drugs: max embedding error `1.4e-6`,
   and **every reconstructed SMILES matches exactly**, including the same two
   near-misses the original code also makes (e.g. acetic acid reconstructed
   as its acetate anion).

```bash
micromamba run -n jax-cddd python -m pytest tests/
```

Both fixtures are pre-generated and committed (`tests/fixtures/`), so the test
suite above never needs TensorFlow. Regenerating them requires a legacy
environment (an old Python for the old TF wheel) and the original
`default_model.zip` checkpoint placed under `src/jax_cddd/_cddd_ref_/`:

```bash
micromamba create -n jax-cddd-convert python=3.7 pip -c conda-forge -y
micromamba run -n jax-cddd-convert pip install "tensorflow==1.15.5" "protobuf==3.20.3"

micromamba run -n jax-cddd-convert python scripts/validate_against_tf1_reference.py
micromamba run -n jax-cddd-convert python scripts/run_original_model.py
```

(That environment is never `pip install -e .`'d — this package requires
Python ≥3.10 — its scripts add `src/` to `sys.path` directly to reuse
`jax_cddd`'s dependency-free submodules.)

A round-trip smoke test on real drugs (aspirin, caffeine, ibuprofen, ...) is
also available standalone:

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
    convert_checkpoint.py             # maintainer-only: how the published weights were produced
    validate_against_tf1_reference.py # from-scratch NumPy fidelity fixture
    run_original_model.py             # runs the real original TF1 graph, for fidelity fixture
    reconstruct_smoke_test.py         # end-to-end embed/reconstruct demo
tests/
```

## 📄 License & citation

The original CDDD implementation is MIT-licensed (© 2018 Jan Robin Winter). If
you use this in published work, please cite the original paper above.
