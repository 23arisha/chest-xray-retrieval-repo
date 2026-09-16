"""
Builds the 10,000-image CheXpert evaluation sample (manuscript Sec. 3.2)
and writes it to disk in the two forms downstream scripts need:

  - eval_df.json            : img_id + the 14 raw CheXpert finding labels
                               (-1/0/1). This is the exact schema
                               tables/generate_tables.py reads.
  - eval_df_with_paths.json : the same, plus each image's absolute file
                               path, for the retrieval scripts
                               (evaluation/02-04_run_*_retrieval.py).

Usage:
    python -m evaluation.00_build_eval_set \
        --chexpert_train_csv /path/to/CheXpert-v1.0-small/train.csv \
        --image_root /path/to/CheXpert-v1.0-small/train/ \
        --output_dir ./eval_outputs
"""
import argparse
import os

from evaluation.common import build_gt_lookups, export_eval_df_json, load_chexpert_eval_set


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chexpert_train_csv", required=True)
    parser.add_argument("--image_root", required=True,
                         help="Absolute path prefix replacing the CSV's "
                              "'CheXpert-v1.0-small/train/' relative prefix.")
    parser.add_argument("--output_dir", default="./eval_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_sample", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    eval_df = load_chexpert_eval_set(
        args.chexpert_train_csv, args.image_root,
        seed=args.seed, n_sample=args.n_sample,
    )

    print(f"Sampled evaluation set size: {len(eval_df)} (seed={args.seed})")
    print("\nPositive cases per finding:")
    from evaluation.common import CHEXPERT_LABELS
    for lbl in CHEXPERT_LABELS:
        print(f"{lbl:<28} {(eval_df[lbl] == 1).sum():,}")

    # Sanity check: GT lookups build without error and cover every image.
    gt_positive_by_id, gt_usable_by_id = build_gt_lookups(eval_df)
    assert len(gt_positive_by_id) == len(eval_df)
    assert len(gt_usable_by_id) == len(eval_df)

    export_eval_df_json(
        eval_df, os.path.join(args.output_dir, "eval_df.json"), include_path=False)
    export_eval_df_json(
        eval_df, os.path.join(args.output_dir, "eval_df_with_paths.json"), include_path=True)


if __name__ == "__main__":
    main()
