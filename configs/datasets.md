# Data sources, checkpoints, and seeds

This file documents every external dataset and pretrained checkpoint used
in the manuscript, so results can be reproduced end to end. None of these
files are included in this repository — only the code that consumes them.

## Datasets

| Dataset | Role | Source | License |
|---|---|---|---|
| MIMIC-CXR | Training data (30,432 frontal image/report pairs) | https://www.kaggle.com/datasets/zainabhalhoul/mimic-dataset | CC BY-NC 4.0 |
| CheXpert | Evaluation data (10,000-image random sample, seed=42) | https://www.kaggle.com/datasets/ashery/chexpert | CC0: Public Domain |

Both are secondary uses of publicly released, de-identified datasets that
received institutional approval and patient consent waivers at original
release (see manuscript "Ethical standards" section).

## Pretrained checkpoints

| Model | Checkpoint | Source |
|---|---|---|
| CXR-RePaiR | `clip-imp-pretrained_128_6_after_4.pt` | Official CXR-RePaiR repository (Endo et al.) |
| CXR-ReDonE | `checkpoint_59.pth` (trained on MIMIC-PRO, prior-study references removed) | Official CXR-ReDonE repository (Ramesh, Chi, Rajpurkar, 2022) |
| CheXbert | `chexbert.pth` | Official CheXbert repository |
| Our model | `image_encoder_best.pth` / `text_encoder_best.pth`, produced by `training/train.py` | Trained in this work |

Both CXR-RePaiR and CXR-ReDonE checkpoints were used frozen, without any
additional fine-tuning, in evaluation mode.

## Preprocessing / sentence pools

- Findings-section text was segmented into sentences with spaCy's
  `en_core_web_sm` tokenizer; sentences under 3 words were dropped.
- **Original candidate pool**: 129,906 unique sentences (no dedup cap).
- **Balanced candidate pool**: 6,216 unique sentences, positive sentences
  per CheXpert finding capped at 450 (seed=42 for subsampling above the
  cap); Pleural Effusion had only 530 available positives in the full pool.
- **Training-set sentence dedup** (training pipeline only, distinct from the
  retrieval pools above): each unique sentence capped at 3 image-sentence
  instances (seed=42), yielding 148,032 pairs from 221,070 raw pairs.

## Seeds

| Seed | Used for |
|---|---|
| `42` | Sentence dedup, image-id train/val/test split (90/5/5), CheXpert sample draw, balanced-pool subsampling |
| `2024` | Bootstrap and permutation procedures in `tables/generate_tables.py` (`RNG_SEED`) |

## Evaluation pipeline file map

Each `evaluation/NN_*.py` stage writes files the next stage consumes.
Names below match the `--output`/`--*_path` arguments used across the
scripts (see each script's docstring for the full CLI):

| Stage | Produces |
|---|---|
| `00_build_eval_set.py` | `eval_df.json`, `eval_df_with_paths.json` |
| `01_build_candidate_pools.py` | `candidate_pool_original.json`, `candidate_pool_balanced.json`, `pool_preds_original.json`, `pool_preds_balanced.json` |
| `02_run_our_model_retrieval.py` | `retrieval_ours_<pool>.json` |
| `03_run_cxr_repair_retrieval.py` | `retrieval_repair_<pool>.json` |
| `04_run_cxr_redone_retrieval.py` | `retrieval_redone_<pool>.json` |
| `05_chexbert_label_retrievals.py` | `chexbert_results_original.json`, `chexbert_results_balanced.json` |

Each retrieval script (02-04) is run twice — once per pool — pointing
`--candidate_pool` at either `candidate_pool_original.json` or
`candidate_pool_balanced.json`, and `--output` at a correspondingly named
file. `tables/generate_tables.py` and `tables/verify_against_manuscript.py`
consume `eval_df.json` plus the two `chexbert_results_*.json` files from
stage 05.

## Notes on file paths in the scripts

The training script and evaluation notebook were originally run on Kaggle
and reference `/kaggle/input/...` paths for the datasets and checkpoints
above. The refactored `training/train.py` and `evaluation/*.py` scripts
take these as CLI arguments instead (with the original Kaggle paths kept
only as documented defaults in `training/train.py`); the original,
as-run notebook is kept for reference at
`evaluation/original_notebook_reference.ipynb`.
