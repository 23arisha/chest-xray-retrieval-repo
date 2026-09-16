"""
Recall / precision / F1 computation, per-finding or micro-averaged
(pooled TP/FP/FN across all findings and images).

Ground-truth uncertain (-1) is always excluded from the denominator,
consistent with manuscript Sec. 3.2.
"""
import numpy as np


def compute_prf_per_finding(pred_col, gt_col):
    """pred_col, gt_col: 1D arrays of length n_images for ONE finding.
    Excludes uncertain (-1) ground-truth entries."""
    valid = gt_col != -1
    p = pred_col[valid]
    g = gt_col[valid]
    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if (precision and recall and (precision + recall) > 0) else np.nan)
    return tp, fp, fn, tn, recall, precision, f1


def compute_micro_prf(pred_matrix, gt_matrix):
    """Micro-average (pooled TP/FP/FN across ALL findings and images)
    recall/precision/F1. VERIFIED: this - not macro-averaging - is the
    methodology that exactly reproduces the manuscript's Table 1 values
    (confirmed to 4 decimal places for all model x pool x K combinations)."""
    valid = gt_matrix != -1
    p = pred_matrix[valid]
    g = gt_matrix[valid]
    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan
    return recall, precision, f1


def per_finding_recall_vector(pred_matrix, gt_matrix, findings):
    """Returns an array of per-finding recall (length = n_findings)."""
    out = np.zeros(len(findings))
    for j in range(len(findings)):
        _, _, _, _, r, _, _ = compute_prf_per_finding(pred_matrix[:, j], gt_matrix[:, j])
        out[j] = r
    return out


def per_finding_tp_fp_fn(pred_matrix, gt_matrix, findings):
    out = {}
    for j, finding in enumerate(findings):
        tp, fp, fn, tn, r, p, f1 = compute_prf_per_finding(pred_matrix[:, j], gt_matrix[:, j])
        out[finding] = dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=r)
    return out
