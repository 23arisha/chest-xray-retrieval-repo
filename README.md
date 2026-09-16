# CXR Retrieval Difficulty

## Repository structure

```
.
├── configs/
│   ├── train_config.py         # Training hyperparameters (Config class, singleton `cfg`)
│   └── datasets.md             # Dataset/checkpoint sources, seeds, evaluation file map
│
├── src/                         # Shared model/data code - single source of truth,
│   │                            # imported by both training/ and evaluation/
│   ├── data/
│   │   ├── preprocessing.py     # Sentence extraction, training-corpus dedup, data-split
│   │   ├── datasets.py          # MimicDataset, MimicDatasetMLM, image transforms
│   │   └── samplers.py          # UniqueImageSampler
│   ├── models/
│   │   ├── image_encoder.py     # SoftAttention, ImageEncoder (grayscale ResNet-50)
│   │   ├── text_encoder.py      # TextEncoder, MLMHead (randomly-init 3-layer Transformer)
│   │   └── attention.py         # func_attention (word-region alignment)
│   ├── losses.py                 # Sentence/word matching + triplet losses
│   ├── engine/trainer.py         # JoImTeR training loop
│   └── utils.py
│
├── training/
│   └── train.py                  # Entry point: data -> model -> JoImTeR trainer -> (optional) .train()
│
├── evaluation/                    # One stage per pipeline step, each independently runnable
│   ├── common.py                       # CheXpert eval-set loading, GT lookups, eval_df export
│   ├── chexbert_utils.py               # CheXbert loading + sentence labeling
│   ├── 00_build_eval_set.py            # Builds the 10,000-image CheXpert evaluation sample
│   ├── 01_build_candidate_pools.py     # Builds the original (129,906) + balanced (6,216) pools
│   ├── 02_run_our_model_retrieval.py   # Retrieval with our trained model (reuses src/)
│   ├── 03_run_cxr_repair_retrieval.py  # Retrieval with the frozen CXR-RePaiR checkpoint
│   ├── 04_run_cxr_redone_retrieval.py  # Retrieval with the frozen CXR-ReDonE checkpoint
│   ├── 05_chexbert_label_retrievals.py # Labels retrieved sentences, merges into tables/ input
│   └── original_notebook_reference.ipynb  # As-run Kaggle notebook, kept for provenance
│
└── tables/
    ├── config.py                     # Finding names, model keys, bootstrap/seed constants
    ├── data_loading.py               # Loads eval_df/results JSON, builds gt/pred matrices
    ├── metrics.py                    # Recall/precision/F1 (per-finding + micro-averaged)
    ├── bootstrap.py                  # Bootstrap CIs, permutation test, BH correction
    ├── generate_tables.py            # Orchestrates and prints Tables 1-4
    └── verify_against_manuscript.py  # Pass/fail check against published manuscript values
```

## Why `src/` exists

The original training script and evaluation notebook each independently
defined the model architecture (`ImageEncoder`, `TextEncoder`, etc.). That
meant two copies of the same classes that could silently drift apart.
`src/` is now the single implementation: `training/train.py` and
`evaluation/02_run_our_model_retrieval.py` both import from it, so the
architecture used to train a checkpoint and the architecture used to
evaluate it are guaranteed identical.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A CUDA-capable GPU is required to train the model and to run the
CXR-RePaiR / CXR-ReDonE inference passes at reasonable speed. Everything
under `tables/` has no GPU dependency and runs on CPU from JSON outputs
alone.

## Data and external checkpoints

This repository contains code only. See `configs/datasets.md` for exact
dataset sources, licenses, and third-party checkpoint filenames
(CXR-RePaiR, CXR-ReDonE, CheXbert).

- **MIMIC-CXR** (training data): https://www.kaggle.com/datasets/zainabhalhoul/mimic-dataset and CC BY-NC 4.0
- **CheXpert** (evaluation data): https://www.kaggle.com/datasets/ashery/chexpert and CC0: Public Domain

## Reproducing the pipeline end to end

```bash
# 1. Train the proposed retrieval model
python -m training.train --mimic_csv /path/to/mimic-data.csv \
    --output_root ./output --train

# 2. Build the 10,000-image CheXpert evaluation sample
python -m evaluation.00_build_eval_set \
    --chexpert_train_csv /path/to/train.csv \
    --image_root /path/to/CheXpert-v1.0-small/train/ \
    --output_dir ./eval_outputs

# 3. Build the original + balanced candidate pools
python -m evaluation.01_build_candidate_pools \
    --mimic_csv /path/to/mimic-data.csv \
    --chexbert_repo_dir /path/to/CheXbert \
    --chexbert_ckpt /path/to/chexbert.pth \
    --output_dir ./pool_outputs

# 4. Run retrieval for each model, for each pool (repeat --candidate_pool /
#    --output per pool)
python -m evaluation.02_run_our_model_retrieval ...
python -m evaluation.03_run_cxr_repair_retrieval ...
python -m evaluation.04_run_cxr_redone_retrieval ...

# 5. Label retrieved sentences with CheXbert and merge into one file per pool
python -m evaluation.05_chexbert_label_retrievals ...

# 6. Reproduce Tables 1-4, then verify against the published values
python -m tables.generate_tables \
    --eval_df eval_outputs/eval_df.json \
    --unbalanced chexbert_results_original.json \
    --balanced chexbert_results_balanced.json

python -m tables.verify_against_manuscript \
    --eval_df eval_outputs/eval_df.json \
    --unbalanced chexbert_results_original.json \
    --balanced chexbert_results_balanced.json
```

Every script's own docstring has the full argument list and exact output
file names; `configs/datasets.md` has a table mapping each stage to what
it produces.

## Random seeds

Seed `42` is used throughout for sentence dedup, train/val/test splitting,
the CheXpert sample draw, and balanced-pool subsampling. Seed `2024`
(`RNG_SEED` in `tables/config.py`) is used for the bootstrap/permutation
procedures that reproduce the manuscript's reported confidence intervals
and p-values.

## Citation

If you use this code, please cite the preprint (details to be updated
upon formal publication via Peer Community In).

## License

Code is released under the MIT License (see `LICENSE`). This covers the
code in this repository only and it does not apply to the third-party
datasets or pretrained checkpoints referenced above, which retain their
own original licenses.
