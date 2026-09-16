"""
Reproduces Tables 1-4 of:
"Comparative Analysis of Clinical Finding Retrieval Difficulty in Chest
X-Ray Retrieval Models"

Usage:
    python -m tables.generate_tables \
        --eval_df eval_df.json \
        --unbalanced chexbert_results_original.json \
        --balanced chexbert_results_balanced.json

Inputs (see evaluation/00_build_eval_set.py and evaluation/05_chexbert_label_retrievals.py):
  - eval_df.json           : 10,000 CheXpert images, ground-truth labels (-1/0/1) for 14 findings
  - <unbalanced>.json      : original-pool retrieval results (top-10 sentences/image/model, CheXbert-labeled)
  - <balanced>.json        : balanced-pool retrieval results (same structure)

Prints:
  - Table 1: overall recall/precision/F1 at K=1,5,10 for both pools
  - Table 2: per-finding recall under the original pool (with 95% CI)
  - Table 3: change in per-finding recall between pools (bootstrap CI + BH-adjusted p)
  - Table 4: cross-model Spearman rank correlation under both pools + paired
             bootstrap Delta-rho test

For a pass/fail check of these outputs against the published manuscript
values, run `python -m tables.verify_against_manuscript` with the same
input files afterward (or see that module for how to call `run_all`
directly and reuse the results in-process).
"""
import argparse
from itertools import combinations

import numpy as np

from tables.bootstrap import (
    bh_correct,
    bootstrap_recall_ci,
    paired_bootstrap_delta_recall,
    paired_bootstrap_delta_rho,
    permutation_test_rho,
)
from tables.config import FINDINGS, MODEL_DISPLAY, MODELS, N_PERMUTATIONS, RNG_SEED, TOP_K_POOL
from tables.data_loading import build_pred_matrix, load_all
from tables.metrics import compute_micro_prf, per_finding_recall_vector


def compute_table1(unbalanced_results, balanced_results, gt, img_id_to_idx, n_images):
    print("\n" + "=" * 70)
    print("TABLE 1: Overall retrieval performance (recall, precision, F1)")
    print("=" * 70)

    table1_rows = []
    for pool_name, results_dict in [("Original", unbalanced_results), ("Balanced", balanced_results)]:
        for model in MODELS:
            row = {"Model": MODEL_DISPLAY[model], "Pool": pool_name}
            for k in [1, 5, 10]:
                pred = build_pred_matrix(results_dict, model, k, n_images, img_id_to_idx, FINDINGS)
                r, p, f1 = compute_micro_prf(pred, gt)
                row[f"Recall@{k}"] = round(r, 4)
                row[f"Precision@{k}"] = round(p, 4)
                row[f"F1@{k}"] = round(f1, 4)
            table1_rows.append(row)
            print(f"{row['Model']:12s} {row['Pool']:9s} | "
                  f"R@1={row['Recall@1']:.4f} P@1={row['Precision@1']:.4f} F1@1={row['F1@1']:.4f} | "
                  f"R@5={row['Recall@5']:.4f} P@5={row['Precision@5']:.4f} F1@5={row['F1@5']:.4f} | "
                  f"R@10={row['Recall@10']:.4f} P@10={row['Precision@10']:.4f} F1@10={row['F1@10']:.4f}")
    return table1_rows


def compute_table2(pred_unbalanced, gt):
    print("\n" + "=" * 70)
    print("TABLE 2: Finding-level recall under the ORIGINAL pool (with 95% CI)")
    print("=" * 70)

    recall_unbalanced = {}
    for model in MODELS:
        recall_unbalanced[model] = per_finding_recall_vector(pred_unbalanced[model], gt, FINDINGS)

    table2_rows = []
    for j, finding in enumerate(FINDINGS):
        row = {"Finding": finding}
        for model in MODELS:
            r = recall_unbalanced[model][j]
            lo, hi = bootstrap_recall_ci(pred_unbalanced[model][:, j], gt[:, j], n_boot=1000, seed=RNG_SEED)
            row[MODEL_DISPLAY[model]] = f"{r:.3f} [{lo:.3f}\u2013{hi:.3f}]"
        table2_rows.append(row)
        print(f"{finding:28s} | " + " | ".join(f"{MODEL_DISPLAY[m]}: {row[MODEL_DISPLAY[m]]}" for m in MODELS))
    return table2_rows, recall_unbalanced


def compute_table3(pred_unbalanced, pred_balanced, gt):
    print("\n" + "=" * 70)
    print("TABLE 3: Change in per-finding recall (balanced - original)")
    print("=" * 70)

    table3_raw = {model: [] for model in MODELS}
    for model in MODELS:
        pvals, deltas = [], []
        for j, finding in enumerate(FINDINGS):
            delta, lo, hi, p = paired_bootstrap_delta_recall(
                pred_unbalanced[model][:, j], pred_balanced[model][:, j], gt[:, j],
                n_boot=1500, seed=RNG_SEED,
            )
            deltas.append((finding, delta, lo, hi))
            pvals.append(p)
        p_adj = bh_correct(pvals)
        for (finding, delta, lo, hi), padj in zip(deltas, p_adj):
            table3_raw[model].append({"Finding": finding, "Delta": delta, "CI_lo": lo, "CI_hi": hi, "p_adj": padj})

    for j, finding in enumerate(FINDINGS):
        line = f"{finding:28s} | "
        for model in MODELS:
            d = table3_raw[model][j]
            sig = "***" if d["p_adj"] < 0.001 else "**" if d["p_adj"] < 0.01 else "*" if d["p_adj"] < 0.05 else ""
            line += f"{MODEL_DISPLAY[model]}: {d['Delta']:+.3f} [{d['CI_lo']:+.3f},{d['CI_hi']:+.3f}]{sig}  "
        print(line)
    return table3_raw


def compute_table4(pred_unbalanced, pred_balanced, recall_unbalanced, gt):
    print("\n" + "=" * 70)
    print("TABLE 4: Cross-model agreement (Spearman rho) + paired bootstrap Delta-rho")
    print("=" * 70)

    pairs = list(combinations(MODELS, 2))
    recall_balanced = {m: per_finding_recall_vector(pred_balanced[m], gt, FINDINGS) for m in MODELS}

    rho_unbalanced, rho_balanced = {}, {}
    perm_p_unbalanced, perm_p_balanced = {}, {}
    for a, b in pairs:
        rho_u, p_u = permutation_test_rho(recall_unbalanced[a], recall_unbalanced[b],
                                           n_perm=N_PERMUTATIONS, seed=RNG_SEED)
        rho_b, p_b = permutation_test_rho(recall_balanced[a], recall_balanced[b],
                                           n_perm=N_PERMUTATIONS, seed=RNG_SEED)
        rho_unbalanced[(a, b)] = rho_u
        rho_balanced[(a, b)] = rho_b
        perm_p_unbalanced[(a, b)] = p_u
        perm_p_balanced[(a, b)] = p_b
        print(f"  Permutation test {MODEL_DISPLAY[a]} vs {MODEL_DISPLAY[b]}: "
              f"Original rho={rho_u:.3f} perm_p={p_u:.5f}  |  "
              f"Balanced rho={rho_b:.3f} perm_p={p_b:.5f}")

    table4_results = []
    raw_pvals = []
    for a, b in pairs:
        boot_rho_unb, boot_rho_bal, boot_delta = paired_bootstrap_delta_rho(
            a, b, gt, pred_unbalanced, pred_balanced, FINDINGS, seed=RNG_SEED)
        ci_unb = np.percentile(boot_rho_unb, [2.5, 97.5])
        ci_bal = np.percentile(boot_rho_bal, [2.5, 97.5])
        # IMPORTANT (verified against manuscript): the reported Delta-rho is
        # the MEAN of the bootstrap distribution of (rho_original -
        # rho_balanced), not the simple difference of the two point-estimate
        # rho values.
        boot_delta_reported = -boot_delta  # (balanced - original) -> (original - balanced)
        delta_mean = np.mean(boot_delta_reported)
        ci_delta = np.percentile(boot_delta_reported, [2.5, 97.5])
        p_val = 2 * min(np.mean(boot_delta_reported <= 0), np.mean(boot_delta_reported >= 0))
        p_val = min(p_val, 1.0)
        raw_pvals.append(p_val)
        table4_results.append({
            "pair": f"{MODEL_DISPLAY[a]} vs {MODEL_DISPLAY[b]}",
            "rho_unb": rho_unbalanced[(a, b)], "ci_unb": ci_unb,
            "rho_bal": rho_balanced[(a, b)], "ci_bal": ci_bal,
            "delta": delta_mean, "ci_delta": ci_delta,
            "perm_p_unb": perm_p_unbalanced[(a, b)], "perm_p_bal": perm_p_balanced[(a, b)],
        })

    p_adj4 = bh_correct(raw_pvals)
    for res, padj in zip(table4_results, p_adj4):
        print(f"{res['pair']:28s} | rho_orig={res['rho_unb']:.3f} "
              f"[{res['ci_unb'][0]:.3f},{res['ci_unb'][1]:.3f}] permp={res['perm_p_unb']:.5f} | "
              f"rho_bal={res['rho_bal']:.3f} [{res['ci_bal'][0]:.3f},{res['ci_bal'][1]:.3f}] "
              f"permp={res['perm_p_bal']:.5f} | "
              f"delta_rho={res['delta']:.3f} [{res['ci_delta'][0]:.3f},{res['ci_delta'][1]:.3f}] | p_BH={padj:.4g}")
    return table4_results


def run_all(eval_df_path, unbalanced_path, balanced_path):
    """Runs the full Tables 1-4 pipeline and returns every intermediate
    result, for reuse by tables/verify_against_manuscript.py without
    recomputation."""
    print("Loading data...")
    eval_df, unbalanced_results, balanced_results, gt, img_id_to_idx, n_images = load_all(
        eval_df_path, unbalanced_path, balanced_path)
    assert n_images == 10000
    print(f"Loaded {n_images} images, ground truth shape {gt.shape}")

    table1_rows = compute_table1(unbalanced_results, balanced_results, gt, img_id_to_idx, n_images)

    pred_unbalanced = {m: build_pred_matrix(unbalanced_results, m, TOP_K_POOL, n_images, img_id_to_idx, FINDINGS)
                        for m in MODELS}
    pred_balanced = {m: build_pred_matrix(balanced_results, m, TOP_K_POOL, n_images, img_id_to_idx, FINDINGS)
                      for m in MODELS}

    table2_rows, recall_unbalanced = compute_table2(pred_unbalanced, gt)
    table3_raw = compute_table3(pred_unbalanced, pred_balanced, gt)
    table4_results = compute_table4(pred_unbalanced, pred_balanced, recall_unbalanced, gt)

    print("\nDone. Compare the printed values above against Tables 1-4 in the manuscript.")

    return {
        "gt": gt,
        "img_id_to_idx": img_id_to_idx,
        "n_images": n_images,
        "unbalanced_results": unbalanced_results,
        "balanced_results": balanced_results,
        "pred_unbalanced": pred_unbalanced,
        "pred_balanced": pred_balanced,
        "recall_unbalanced": recall_unbalanced,
        "table1_rows": table1_rows,
        "table2_rows": table2_rows,
        "table3_raw": table3_raw,
        "table4_results": table4_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_df", required=True)
    parser.add_argument("--unbalanced", required=True,
                         help="chexbert_results_original.json (original-pool retrieval results)")
    parser.add_argument("--balanced", required=True,
                         help="chexbert_results_balanced.json (balanced-pool retrieval results)")
    args = parser.parse_args()

    run_all(args.eval_df, args.unbalanced, args.balanced)


if __name__ == "__main__":
    main()
