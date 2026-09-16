"""
Builds the two retrieval candidate pools used in the manuscript (Sec. 3.3):

  - the ORIGINAL pool: every unique findings-section sentence (>=3 words)
    extracted from MIMIC-CXR, no cap (129,906 sentences in the reported run).
  - the BALANCED pool: original pool sentences labeled with CheXbert, then
    positive sentences per finding capped at `--cap_per_label` (450 in the
    manuscript; seed=42 for subsampling above the cap), yielding 6,216
    sentences in the reported run.

Reuses `extract_subsentences` from `src.data.preprocessing` so sentence
segmentation is guaranteed identical to the training pipeline.

Usage:
    python -m evaluation.01_build_candidate_pools \
        --mimic_csv /path/to/mimic-data.csv \
        --chexbert_repo_dir /path/to/CheXbert \
        --chexbert_ckpt /path/to/chexbert.pth \
        --output_dir ./pool_outputs

Outputs (written to --output_dir):
  - candidate_pool_original.json   : sorted list of unique sentences
  - candidate_pool_balanced.json   : sorted list of balanced-pool sentences
  - pool_preds_original.json       : per-sentence CheXbert labels for the
                                      original pool (index-aligned with
                                      candidate_pool_original.json)
  - pool_preds_balanced.json       : per-sentence CheXbert labels for the
                                      balanced pool, reused by
                                      evaluation/05_chexbert_label_retrievals.py
                                      as a label lookup (avoids re-running
                                      CheXbert on sentences already labeled
                                      here, exactly as the original notebook did).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluation.chexbert_utils import load_chexbert, chexbert_label_sentences
from evaluation.common import CHEXPERT_LABELS
from src.data.preprocessing import extract_subsentences


def build_candidate_pool(mimic_csv):
    """Extracts every unique findings-section sentence (>=3 words) from
    MIMIC-CXR, with NO deduplication cap (unlike the training-corpus
    preprocessing in src.data.preprocessing.deduplicate_sentences, which
    caps each unique sentence at 3 instances)."""
    mimic = pd.read_csv(mimic_csv)
    mimic["findings"] = mimic["findings"].fillna("")
    mimic = mimic[mimic["findings"] != ""].reset_index(drop=True)

    candidate_pool = set()
    for text in tqdm(mimic["findings"], desc="Extracting sentences"):
        for sent in extract_subsentences(text):
            candidate_pool.add(sent)

    candidate_pool = sorted(candidate_pool)
    print(f"Original candidate pool size: {len(candidate_pool)}")
    return candidate_pool


def print_pool_composition(pool_preds, label_cols, pool_name):
    label_counts = {lbl: 0 for lbl in label_cols}
    for pred in pool_preds:
        for lbl in label_cols:
            if pred[lbl] == 1:
                label_counts[lbl] += 1

    total = len(pool_preds)
    print(f"\n=== {pool_name} pool composition: {total} sentences ===\n")
    print(f"{'Label':<28}{'Count':>8}{'% of pool':>12}")
    for lbl, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / total
        print(f"{lbl:<28}{count:>8}{pct:>11.2f}%")


def build_balanced_pool(candidate_pool, pool_preds, label_cols,
                         cap_per_label=450, seed=42):
    """Caps positive sentences per finding at `cap_per_label` (manuscript:
    450, bounded by the smallest class - Pleural Effusion had 530 total in
    the original pool). Sentences positive for multiple findings can be
    selected via more than one finding's cap; duplicates are collapsed."""
    rng = np.random.default_rng(seed)

    label_to_indices = {lbl: [] for lbl in label_cols}
    for i, pred in enumerate(pool_preds):
        for lbl in label_cols:
            if pred[lbl] == 1:
                label_to_indices[lbl].append(i)

    print(f"{'Label':<28}{'Available':>10}{'Sampled':>10}")
    balanced_indices = set()
    for lbl in label_cols:
        avail = label_to_indices[lbl]
        n_take = min(cap_per_label, len(avail))
        if n_take < len(avail):
            chosen = rng.choice(avail, size=n_take, replace=False).tolist()
        else:
            chosen = avail
        balanced_indices.update(chosen)
        flag = "  <-- capped by availability" if n_take < cap_per_label else ""
        print(f"{lbl:<28}{len(avail):>10}{n_take:>10}{flag}")

    balanced_indices = sorted(balanced_indices)
    balanced_pool = [candidate_pool[i] for i in balanced_indices]
    print(f"\nBalanced pool total size (deduped, multi-label overlap collapsed): "
          f"{len(balanced_pool)}")
    return balanced_pool, balanced_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mimic_csv", required=True)
    parser.add_argument("--chexbert_repo_dir", required=True,
                         help="Local clone of https://github.com/stanfordmlgroup/CheXbert")
    parser.add_argument("--chexbert_ckpt", required=True,
                         help="Path to the public chexbert.pth checkpoint")
    parser.add_argument("--output_dir", default="./pool_outputs")
    parser.add_argument("--cap_per_label", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chexbert_batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidate_pool = build_candidate_pool(args.mimic_csv)
    with open(os.path.join(args.output_dir, "candidate_pool_original.json"), "w") as f:
        json.dump(candidate_pool, f)

    cx_mdl, cx_tok = load_chexbert(args.chexbert_repo_dir, args.chexbert_ckpt, device)

    pool_preds = chexbert_label_sentences(
        candidate_pool, cx_mdl, cx_tok, device,
        batch_size=args.chexbert_batch_size, show_progress=True,
    )
    with open(os.path.join(args.output_dir, "pool_preds_original.json"), "w") as f:
        json.dump(pool_preds, f)
    print_pool_composition(pool_preds, CHEXPERT_LABELS, "Original")

    balanced_pool, _ = build_balanced_pool(
        candidate_pool, pool_preds, CHEXPERT_LABELS,
        cap_per_label=args.cap_per_label, seed=args.seed,
    )
    with open(os.path.join(args.output_dir, "candidate_pool_balanced.json"), "w") as f:
        json.dump(balanced_pool, f)

    balanced_preds = chexbert_label_sentences(
        balanced_pool, cx_mdl, cx_tok, device,
        batch_size=args.chexbert_batch_size, show_progress=True,
    )
    with open(os.path.join(args.output_dir, "pool_preds_balanced.json"), "w") as f:
        json.dump(balanced_preds, f)
    print_pool_composition(balanced_preds, CHEXPERT_LABELS, "Balanced")


if __name__ == "__main__":
    main()
