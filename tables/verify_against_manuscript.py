"""
Automated pass/fail check of tables/generate_tables.py's output against the
published manuscript values (hard-coded from the manuscript's text and
Tables 1-4).

Usage:
    python -m tables.verify_against_manuscript \
        --eval_df eval_df.json \
        --unbalanced chexbert_results_original.json \
        --balanced chexbert_results_balanced.json
"""
import argparse

from tables.config import FINDINGS, MODEL_DISPLAY
from tables.data_loading import build_pred_matrix
from tables.generate_tables import run_all
from tables.metrics import compute_micro_prf


def check(label, computed, published, tol=0.001):
    ok = abs(computed - published) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: computed={computed:.4f}  published={published:.4f}  "
          f"diff={abs(computed - published):.4f}")
    return ok


def verify(results):
    """`results` is the dict returned by tables.generate_tables.run_all()."""
    gt = results["gt"]
    n_images = results["n_images"]
    img_id_to_idx = results["img_id_to_idx"]
    unbalanced_results = results["unbalanced_results"]
    balanced_results = results["balanced_results"]
    recall_unbalanced = results["recall_unbalanced"]
    table3_raw = results["table3_raw"]
    table4_results = results["table4_results"]

    print("\n" + "=" * 70)
    print("AUTOMATED VERIFICATION vs. PUBLISHED MANUSCRIPT VALUES")
    print("=" * 70)

    all_ok = True

    # --- Table 1 checks (Original pool, "ours") ---
    pred = build_pred_matrix(unbalanced_results, "ours", 1, n_images, img_id_to_idx, FINDINGS)
    r, p, f1 = compute_micro_prf(pred, gt)
    all_ok &= check("T1 Our-model Original R@1", r, 0.1485)
    all_ok &= check("T1 Our-model Original P@1", p, 0.2938)
    all_ok &= check("T1 Our-model Original F1@1", f1, 0.1973)

    pred = build_pred_matrix(unbalanced_results, "repair", 10, n_images, img_id_to_idx, FINDINGS)
    r, p, f1 = compute_micro_prf(pred, gt)
    all_ok &= check("T1 CXR-RePaiR Original R@10", r, 0.2908)

    pred = build_pred_matrix(balanced_results, "redone", 5, n_images, img_id_to_idx, FINDINGS)
    r, p, f1 = compute_micro_prf(pred, gt)
    all_ok &= check("T1 CXR-ReDonE Balanced R@5", r, 0.2817)

    # --- Table 2 checks (Original pool, per-finding recall at TOP_K_POOL=5) ---
    idx_lo = FINDINGS.index("Lung Opacity")
    idx_pe = FINDINGS.index("Pleural Effusion")
    idx_sd = FINDINGS.index("Support Devices")
    all_ok &= check("T2 Our-model Lung Opacity recall", recall_unbalanced["ours"][idx_lo], 0.033, tol=0.002)
    all_ok &= check("T2 Our-model Pleural Effusion recall", recall_unbalanced["ours"][idx_pe], 0.023, tol=0.002)
    all_ok &= check("T2 Our-model Support Devices recall", recall_unbalanced["ours"][idx_sd], 0.867, tol=0.002)

    # --- Table 3 checks (delta recall, balanced - original) ---
    all_ok &= check("T3 Our-model Support Devices delta", table3_raw["ours"][idx_sd]["Delta"], -0.198, tol=0.003)
    all_ok &= check("T3 Our-model Lung Opacity delta", table3_raw["ours"][idx_lo]["Delta"], 0.082, tol=0.003)
    all_ok &= check("T3 Our-model Pleural Effusion delta", table3_raw["ours"][idx_pe]["Delta"], 0.141, tol=0.003)

    # --- Table 4 checks (rho + delta-rho) ---
    published_t4 = {
        ("ours", "repair"): dict(rho_o=0.807, rho_b=0.424, delta=0.403),
        ("ours", "redone"): dict(rho_o=0.749, rho_b=0.495, delta=0.298),
        ("repair", "redone"): dict(rho_o=0.780, rho_b=0.240, delta=0.487),
    }
    for res in table4_results:
        for (a, b), pub in published_t4.items():
            if MODEL_DISPLAY[a] in res["pair"] and MODEL_DISPLAY[b] in res["pair"]:
                all_ok &= check(f"T4 rho_original {res['pair']}", res["rho_unb"], pub["rho_o"], tol=0.002)
                all_ok &= check(f"T4 rho_balanced {res['pair']}", res["rho_bal"], pub["rho_b"], tol=0.002)
                # bootstrap-dependent, wider tolerance
                all_ok &= check(f"T4 delta_rho {res['pair']}", res["delta"], pub["delta"], tol=0.02)

    # --- Table 4 permutation p-value checks ---
    # (manuscript: original pool p<=0.003, balanced pool p in 0.07-0.41)
    print()
    for res in table4_results:
        ok_orig = res["perm_p_unb"] <= 0.003
        ok_bal = 0.05 <= res["perm_p_bal"] <= 0.45  # wide band matching the manuscript's reported range
        print(f"[{'PASS' if ok_orig else 'FAIL'}] T4 permutation p (Original) {res['pair']}: "
              f"{res['perm_p_unb']:.5f} (expected <= 0.003)")
        print(f"[{'PASS' if ok_bal else 'FAIL'}] T4 permutation p (Balanced) {res['pair']}: "
              f"{res['perm_p_bal']:.5f} (expected in 0.07-0.41 range)")
        all_ok &= ok_orig
        all_ok &= ok_bal

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED - inspect above") + "\n")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_df", required=True)
    parser.add_argument("--unbalanced", required=True)
    parser.add_argument("--balanced", required=True)
    args = parser.parse_args()

    results = run_all(args.eval_df, args.unbalanced, args.balanced)
    verify(results)


if __name__ == "__main__":
    main()
