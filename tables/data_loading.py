"""
Loads the evaluation-pipeline outputs (eval_df.json + the two
chexbert_results_*.json files) and builds the numpy ground-truth /
predicted-positive matrices that every table in this package is computed
from.
"""
import json

import numpy as np

from tables.config import FINDINGS


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_gt_matrix(eval_df, findings=FINDINGS):
    """Ground truth matrix: (n_images, n_findings), -1 = uncertain
    (excluded downstream), 0/1 = negative/positive.

    Returns (gt, img_id_to_idx). img_id values are NOT a contiguous
    0..n-1 range (they are original CheXpert training-set row indices),
    so img_id_to_idx maps each img_id to its dense row index.
    """
    n_images = len(eval_df)
    img_ids_sorted = sorted(row["img_id"] for row in eval_df)
    img_id_to_idx = {img_id: i for i, img_id in enumerate(img_ids_sorted)}

    gt = np.zeros((n_images, len(findings)), dtype=int)
    for row in eval_df:
        idx = img_id_to_idx[row["img_id"]]
        for j, finding in enumerate(findings):
            gt[idx, j] = row[finding]
    return gt, img_id_to_idx


def build_pred_matrix(results_dict, model, k, n_images, img_id_to_idx, findings=FINDINGS):
    """Returns a (n_images, n_findings) binary matrix: 1 if any of the
    top-k retrieved sentences for that image/model is CheXbert-labeled
    positive (==1) for that finding."""
    pred = np.zeros((n_images, len(findings)), dtype=int)
    for img_id_str, rec in results_dict.items():
        img_id = int(img_id_str)
        idx = img_id_to_idx[img_id]
        sentences = rec[model][:k]
        for j, finding in enumerate(findings):
            if any(s[finding] == 1 for s in sentences):
                pred[idx, j] = 1
    return pred


def load_all(eval_df_path, unbalanced_path, balanced_path):
    """Convenience loader used by tables/generate_tables.py.

    Returns (eval_df, unbalanced_results, balanced_results, gt, img_id_to_idx, n_images).
    """
    eval_df = load_json(eval_df_path)
    unbalanced_results = load_json(unbalanced_path)
    balanced_results = load_json(balanced_path)

    n_images = len(eval_df)
    gt, img_id_to_idx = build_gt_matrix(eval_df)

    return eval_df, unbalanced_results, balanced_results, gt, img_id_to_idx, n_images
