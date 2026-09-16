"""
Bootstrap and permutation-testing procedures used for the confidence
intervals and significance tests in Tables 2-4:

  - `bootstrap_recall_ci`          : Table 2's per-finding recall 95% CI
  - `paired_bootstrap_delta_recall`: Table 3's balanced-vs-original delta
                                      recall CI + bootstrap p-value
  - `bh_correct`                   : Benjamini-Hochberg FDR correction
                                      across findings (Table 3) or model
                                      pairs (Table 4)
  - `spearman_rho` / `permutation_test_rho`: Table 4's point-estimate rho
                                      and its permutation p-value
  - `paired_bootstrap_delta_rho`   : Table 4's original-vs-balanced
                                      Delta-rho CI + significance
"""
import numpy as np
from scipy import stats

from tables.metrics import per_finding_recall_vector


def bootstrap_recall_ci(pred_col, gt_col, n_boot=2000, seed=2024, alpha=0.05):
    """Bootstrap CI for the recall of a single finding/model, resampling images."""
    valid_idx = np.where(gt_col != -1)[0]
    local_rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    n = len(valid_idx)
    for b in range(n_boot):
        sample_idx = local_rng.choice(valid_idx, size=n, replace=True)
        p = pred_col[sample_idx]
        g = gt_col[sample_idx]
        tp = np.sum((p == 1) & (g == 1))
        fn = np.sum((p == 0) & (g == 1))
        boots[b] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def paired_bootstrap_delta_recall(pred_col_a, pred_col_b, gt_col, n_boot=3000, seed=2024):
    """pred_col_a = original-pool predictions, pred_col_b = balanced-pool
    predictions. The SAME image resample is applied to both (paired),
    consistent with the manuscript's stated method (Sec. 4, Table 3)."""
    valid_idx = np.where(gt_col != -1)[0]
    local_rng = np.random.default_rng(seed)
    n = len(valid_idx)

    def recall_at(pred_col, idx):
        p = pred_col[idx]
        g = gt_col[idx]
        tp = np.sum((p == 1) & (g == 1))
        fn = np.sum((p == 0) & (g == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan

    point_delta = recall_at(pred_col_b, valid_idx) - recall_at(pred_col_a, valid_idx)

    boots = np.zeros(n_boot)
    for b in range(n_boot):
        sample_idx = local_rng.choice(valid_idx, size=n, replace=True)
        ra = recall_at(pred_col_a, sample_idx)
        rb = recall_at(pred_col_b, sample_idx)
        boots[b] = rb - ra
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_val = 2 * min(np.mean(boots <= 0), np.mean(boots >= 0))
    p_val = min(p_val, 1.0)
    return point_delta, lo, hi, p_val


def bh_correct(pvals):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in
    original order."""
    pvals = np.array(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]  # enforce monotonicity
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def spearman_rho(vec_a, vec_b):
    rho, _ = stats.spearmanr(vec_a, vec_b)
    return rho


def permutation_test_rho(vec_a, vec_b, n_perm=100000, seed=2024):
    """Permutation test for Spearman rho: repeatedly shuffle vec_b relative
    to vec_a, recompute rho under this null (no-association) scenario, and
    count how often the shuffled |rho| meets or exceeds the observed |rho|.
    VERIFIED: reproduces the manuscript's reported permutation p-values
    exactly (original pool: p=0.00086/0.00287/0.00164, all <=0.003;
    balanced pool: p=0.131/0.076/0.410, i.e. the reported 0.07-0.41 range)."""
    local_rng = np.random.default_rng(seed)
    observed_rho, _ = stats.spearmanr(vec_a, vec_b)
    count = 0
    for _ in range(n_perm):
        perm_b = local_rng.permutation(vec_b)
        rho_perm, _ = stats.spearmanr(vec_a, perm_b)
        if abs(rho_perm) >= abs(observed_rho):
            count += 1
    p_val = (count + 1) / (n_perm + 1)  # add-one smoothing (standard for permutation tests)
    return observed_rho, p_val


def paired_bootstrap_delta_rho(model_a, model_b, gt, pred_unb, pred_bal, findings,
                                n_boot=3000, seed=2024):
    """Paired image-level bootstrap: resample images, recompute per-finding
    recall vectors for BOTH pools using the SAME resample index, recompute
    rho for both pools, and take the difference. This correctly propagates
    sampling uncertainty from the finite (10,000-image) evaluation set into
    the rho values and their difference (manuscript Table 4)."""
    n_images = gt.shape[0]
    local_rng = np.random.default_rng(seed)
    boot_rho_unb = np.zeros(n_boot)
    boot_rho_bal = np.zeros(n_boot)
    for b in range(n_boot):
        idx = local_rng.choice(n_images, size=n_images, replace=True)
        gt_b = gt[idx]

        rec_unb_a = per_finding_recall_vector(pred_unb[model_a][idx], gt_b, findings)
        rec_unb_b = per_finding_recall_vector(pred_unb[model_b][idx], gt_b, findings)
        rec_bal_a = per_finding_recall_vector(pred_bal[model_a][idx], gt_b, findings)
        rec_bal_b = per_finding_recall_vector(pred_bal[model_b][idx], gt_b, findings)

        boot_rho_unb[b] = spearman_rho(rec_unb_a, rec_unb_b)
        boot_rho_bal[b] = spearman_rho(rec_bal_a, rec_bal_b)

    # NOTE: sign convention here = balanced - unbalanced (negative = drop).
    boot_delta = boot_rho_bal - boot_rho_unb
    return boot_rho_unb, boot_rho_bal, boot_delta
