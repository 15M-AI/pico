# Pico

Pico is a standalone, experimental model-training project for Open Mind's
15-million-parameter byte-level transformer. This repository contains the Pico
recipe and cloud/local launch scripts. The shared model implementation, OMF
writer/reader, and runtime live in the separate
[`15M-AI/open-mind`](https://github.com/15M-AI/open-mind) repository.

**All new model checkpoints are written directly in OMF v0.1.** The training
scripts do not use GGUF or convert from another model format.

> **Realistic expectations:** 15M parameters is a very small model. Training
> from scratch at this size cannot be promised to produce a capable assistant.
> Byte-level tokenization is particularly inefficient for natural language,
> and a large lawful corpus is essential. This repo contains no corpus and
> starts no training by itself.

## Model

| Setting | Value |
|---|---:|
| Parameters | 14,983,680 |
| Transformer blocks | 8 |
| Hidden size / attention heads | 384 / 6 |
| Feed-forward width | 1,632 |
| Context | 256 byte tokens |
| Vocabulary | 256 |
| Optimizer | AdamW on CUDA; 8-bit momentum SGD on CPU |
| Regularization | 10% dropout |
| Cloud checkpoint format | OMF v0.1, fp16 |

The default recipe in `config.json` is **20,000 optimizer steps**, batch size
8, sequence length 256, and 4 gradient-accumulation steps: about **163.84
million byte-tokens** if all steps complete. This is a starting budget, not a
quality guarantee. Benchmark a short run on your actual GPU before committing
to the full job.

## Kaggle notebook

### 1. Prepare your Kaggle inputs

- Create a Kaggle notebook with a GPU accelerator enabled.
- Attach a dataset containing a UTF-8 plain-text corpus, for example
  `/kaggle/input/my-pico-data/corpus.txt`.
- Confirm that the data license permits training and retain a source/license
  record in `DATASET_PROVENANCE_TEMPLATE.csv`.
- In notebook settings, allow Internet so the runner can clone the public core
  repository. If Internet is disabled, attach a Kaggle dataset containing a
  compatible Open Mind checkout and set `OPEN_MIND_CORE_DIR` to its path.

Do not upload private or unlicensed text to a public Kaggle dataset. The data
preflight script reads the corpus in chunks, validates strict UTF-8, checks
minimum size, and prints a SHA-256. It does not download data, tokenize in
advance, shuffle documents, or copy the source corpus.

### 2. Get Pico source into the notebook

If Pico is in a Kaggle dataset, use its mounted directory, for example:

```bash
PICO_DIR=/kaggle/input/pico-source
```

Or clone the public repository into the writable working directory:

```bash
git clone https://github.com/15M-AI/pico.git /kaggle/working/Pico
PICO_DIR=/kaggle/working/Pico
```

### 3. Run a short pilot first

Run a short job and inspect the train/validation loss, runtime, VRAM, and
generated samples before starting a longer training run:

```bash
cd "$PICO_DIR"
bash scripts/run_kaggle.sh /kaggle/input/my-pico-data/corpus.txt \
  --steps 100 \
  --checkpoint-every 50 \
  --eval-interval 50 \
  --eval-batches 2
```

### 4. Start the configured run

After reviewing the pilot and dataset provenance:

```bash
bash scripts/run_kaggle.sh /kaggle/input/my-pico-data/corpus.txt
```

The runner clones `15M-AI/open-mind` to `/kaggle/working/open-mind` when needed,
checks that CUDA and PyTorch are available, installs the core packages without
replacing the Kaggle CUDA-enabled PyTorch, and trains to
`/kaggle/working/pico-output/`. Outputs include:

- `pico-0.1.0-v1.omf`: periodically replaced OMF checkpoint.
- `pico-0.1.0.omf`: final OMF model.
- `corpus-manifest.json`: input path, size, encoding, and SHA-256.

Kaggle working storage is temporary. Before the session ends, download the
output directory or publish it as a **private Kaggle dataset/version**. Do not
commit checkpoints to Git. If a session stops after a periodic checkpoint,
restore `pico-0.1.0-v1.omf` to the output directory and rerun the script: it
loads the weights and continues at the recorded step. Also restore the
matching `corpus-manifest.json`; the runner refuses to resume when the saved
corpus SHA-256 is missing or differs. Optimizer state and random-number-
generator state are not stored in OMF, so resumed training is not bitwise
equivalent to an uninterrupted run.

## Local CPU training

Use Python 3.10+, locally installed PyTorch and NumPy, and a UTF-8 corpus. The
updated core repository is already published; the first local run clones it to
`~/.cache/open-mind/core` and installs its local packages:

```bash
cd ~/Pico
bash scripts/run_local.sh /path/to/corpus.txt \
  --steps 100 \
  --checkpoint-every 50 \
  --eval-interval 50 \
  --eval-batches 2
```

To use an existing core checkout, set
`OPEN_MIND_CORE_DIR=/path/to/OpenMind`. The local launcher selects CPU. The
default 15M preset can be slow and memory intensive on a laptop; begin with a
short pilot. For CPU-only training in constrained environments, reduce
`--batch-size`, `--seq-len`, and/or `--gradient-accumulation-steps`. CPU
training automatically selects 8-bit momentum SGD; use `--optimizer adamw`
only when its larger optimizer-state memory is acceptable.

## Resume or adjust a run

The training target is a total optimizer-step count. Set `--steps` to the
desired total and pass an OMF checkpoint to continue from it:

```bash
python train.py \
  --device cuda \
  --dataset /path/to/the/same-corpus.txt \
  --output /path/to/pico-0.1.0.omf \
  --steps 20000 \
  --resume-from /path/to/pico-0.1.0-v1.omf
```

The resume path validates the checkpoint configuration. Preserve the exact
training corpus and settings across a resume. Optimizer state and
random-number-generator state are not included in OMF, so a resumed run
reinitializes those states.

## Files

| Path | Purpose |
|---|---|
| `config.json` | Architecture and training defaults |
| `train.py` | Pico-specific CLI; uses the shared Open Mind trainer |
| `prepare_data.py` | Streaming UTF-8, minimum-size, and SHA-256 validation |
| `scripts/setup_core.sh` | Install core OMF/training packages without changing PyTorch |
| `scripts/run_kaggle.sh` | Kaggle CUDA preflight, OMF training, and artifact hashes |
| `scripts/run_local.sh` | CPU-first local training |
| `MODEL_CARD.md` | Scope, evaluation status, limitations, and release checklist |
| `DATASET_CARD_TEMPLATE.md` | Dataset rights and preparation checklist |
| `DATASET_PROVENANCE_TEMPLATE.csv` | Machine-readable data source/license manifest |
| `legacy/` | Preserved pre-OMF assistant prototype; not used by this recipe |

Run recipe-only checks without training:

```bash
python -m unittest discover -s tests -v
bash -n scripts/*.sh
```

## Licensing and model release

The Pico code is Apache-2.0. The weights have **no license yet**. Decide on a
weight license only after verifying that every dataset's terms permit training
and redistribution of derived weights. Complete the dataset provenance and
model-card release checklists, evaluate a held-out set, and review for
memorization before releasing weights. No training corpus or OMF weights are
included in this repository.
