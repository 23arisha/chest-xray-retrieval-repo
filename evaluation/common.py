"""
Shared constants and helpers for the evaluation pipeline: the 14 CheXpert
finding names, the 10,000-image evaluation sample (manuscript Sec. 3.2),
and the ground-truth lookup dicts used by the quick in-pipeline metrics
preview (see evaluation/06_quick_metrics_preview.py).

`tables/generate_tables.py` is the authoritative, manuscript-verified
source of Tables 1-4 and consumes `eval_df.json` as written by
`export_eval_df_json` below (values kept as raw CheXpert -1/0/1 ints, with
-1 = uncertain, excluded downstream).
"""
import json

import numpy as np
import pandas as pd

CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices",
]


def load_chexpert_eval_set(chexpert_train_csv, image_root, seed=42, n_sample=10000):
    """Loads the CheXpert training-set CSV, keeps frontal views only, fills
    unmentioned findings as 0 (uncertain stays -1), assigns a dense `img_id`,
    and draws a fixed random sample of `n_sample` images (manuscript Sec. 3.2).

    `image_root` replaces the CSV's relative "CheXpert-v1.0-small/train/"
    prefix with an absolute path to the actual image files.
    """
    train_df = pd.read_csv(chexpert_train_csv)
    train_df["Path"] = train_df["Path"].str.replace(
        "CheXpert-v1.0-small/train/", image_root, regex=False
    )

    train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"].copy()
    train_df[CHEXPERT_LABELS] = train_df[CHEXPERT_LABELS].fillna(0).astype(int)
    train_df = train_df.reset_index(drop=True)
    train_df["img_id"] = train_df.index

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(train_df.index, size=n_sample, replace=False)
    eval_df = train_df.loc[sorted(sampled_indices)].reset_index(drop=True)
    return eval_df


def build_gt_lookups(eval_df, label_cols=CHEXPERT_LABELS):
    """Returns (gt_positive_by_id, gt_usable_by_id): per-image sets of
    positive findings and of "usable" (non-uncertain) findings, keyed by
    img_id. Uncertain (-1) labels are excluded from both."""
    eval_df_masked = eval_df.copy()
    for col in label_cols:
        eval_df_masked[col] = eval_df_masked[col].replace(-1, pd.NA)

    gt_positive_by_id = {}
    gt_usable_by_id = {}
    for _, r in eval_df_masked.iterrows():
        img_id = r["img_id"]
        gt_positive_by_id[img_id] = {
            c for c in label_cols if pd.notna(r[c]) and r[c] == 1
        }
        gt_usable_by_id[img_id] = {c for c in label_cols if pd.notna(r[c])}
    return gt_positive_by_id, gt_usable_by_id


def export_eval_df_json(eval_df, out_path, label_cols=CHEXPERT_LABELS, include_path=False):
    """Writes eval_df.json in the format expected by tables/generate_tables.py:
    a JSON list of records, each with "img_id" and one raw -1/0/1 int per
    finding (uncertain=-1, excluded downstream; unmentioned already folded
    into 0 by load_chexpert_eval_set).

    If `include_path=True`, each record also carries the image's "Path"
    (absolute path on disk) -- used by the retrieval scripts
    (evaluation/02-04_run_*_retrieval.py) to load images, but NOT part of
    the schema tables/generate_tables.py reads, which only needs img_id +
    finding labels."""
    records = []
    for _, r in eval_df.iterrows():
        rec = {"img_id": int(r["img_id"])}
        for c in label_cols:
            rec[c] = int(r[c])
        if include_path:
            rec["Path"] = r["Path"]
        records.append(rec)
    with open(out_path, "w") as f:
        json.dump(records, f)
    print(f"Wrote {len(records)} records to {out_path}")
