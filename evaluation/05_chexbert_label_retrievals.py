"""
Labels each model's top-k retrieved sentences (from evaluation/02-04) with
CheXbert, and merges all three models into the single
`chexbert_results_<pool>.json` file that `tables/generate_tables.py` reads
to produce manuscript Tables 1-4.

For the balanced pool, retrieved sentences are almost always already
present in `pool_preds_balanced.json` (written by
evaluation/01_build_candidate_pools.py when the balanced pool itself was
labeled) - this script reuses those labels via lookup instead of re-running
CheXbert on the same sentences twice, exactly as the original notebook did.
Any retrieved sentence not found in the lookup (should not normally happen,
since retrieval only draws from the candidate pool) falls back to a fresh
CheXbert call.

Usage:
    python -m evaluation.05_chexbert_label_retrievals \
        --chexbert_repo_dir /path/to/CheXbert \
        --chexbert_ckpt /path/to/chexbert.pth \
        --retrieval_ours   retrieval_outputs/retrieval_ours_original.json \
        --retrieval_repair retrieval_outputs/retrieval_repair_original.json \
        --retrieval_redone retrieval_outputs/retrieval_redone_original.json \
        --output chexbert_results_original.json
    # For the balanced pool, additionally pass:
        --candidate_pool pool_outputs/candidate_pool_balanced.json \
        --pool_preds     pool_outputs/pool_preds_balanced.json

Output JSON schema (matches tables/generate_tables.py):
    {img_id_str: {"ours": [...], "repair": [...], "redone": [...]}}
  where each model's value is a list of per-sentence CheXbert label dicts,
  one per top-k retrieved sentence, in retrieval rank order.
"""
import argparse
import json

import torch
from tqdm import tqdm

from evaluation.chexbert_utils import chexbert_label_sentences, load_chexbert

MODEL_KEYS = ("ours", "repair", "redone")


def build_sentence_lookup(candidate_pool, pool_preds):
    return {s: pool_preds[i] for i, s in enumerate(candidate_pool)}


def label_via_lookup_or_chexbert(sentences, lookup, cx_mdl, cx_tok, device):
    """Labels `sentences` using `lookup` where possible; any misses are
    labeled with a single fresh (batched) CheXbert call, preserving order."""
    out = [lookup.get(s) for s in sentences]
    missing_idx = [i for i, v in enumerate(out) if v is None]
    if missing_idx:
        missing_sents = [sentences[i] for i in missing_idx]
        fresh = chexbert_label_sentences(missing_sents, cx_mdl, cx_tok, device)
        for i, pred in zip(missing_idx, fresh):
            out[i] = pred
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chexbert_repo_dir", required=True)
    parser.add_argument("--chexbert_ckpt", required=True)
    parser.add_argument("--retrieval_ours", required=True)
    parser.add_argument("--retrieval_repair", required=True)
    parser.add_argument("--retrieval_redone", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate_pool", default=None,
                         help="If given (with --pool_preds), retrieved sentences are "
                              "labeled via lookup against this already-labeled pool "
                              "instead of a fresh CheXbert call per sentence. Use the "
                              "BALANCED pool's files here; for the original pool this "
                              "is optional but still saves compute.")
    parser.add_argument("--pool_preds", default=None)
    parser.add_argument("--chexbert_batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retrieval_paths = {
        "ours": args.retrieval_ours,
        "repair": args.retrieval_repair,
        "redone": args.retrieval_redone,
    }
    retrieval_results = {}
    for key, path in retrieval_paths.items():
        with open(path) as f:
            retrieval_results[key] = json.load(f)

    all_img_ids = set()
    for key in MODEL_KEYS:
        all_img_ids.update(retrieval_results[key].keys())

    cx_mdl, cx_tok = load_chexbert(args.chexbert_repo_dir, args.chexbert_ckpt, device)

    lookup = {}
    if args.candidate_pool and args.pool_preds:
        with open(args.candidate_pool) as f:
            candidate_pool = json.load(f)
        with open(args.pool_preds) as f:
            pool_preds = json.load(f)
        lookup = build_sentence_lookup(candidate_pool, pool_preds)
        print(f"Loaded label lookup for {len(lookup)} pool sentences.")

    chexbert_results = {img_id: {} for img_id in all_img_ids}

    for key in MODEL_KEYS:
        for img_id, res in tqdm(retrieval_results[key].items(),
                                 desc=f"Labeling {key}", mininterval=2.0):
            sentences = res["sentences"]
            if lookup:
                chexbert_results[img_id][key] = label_via_lookup_or_chexbert(
                    sentences, lookup, cx_mdl, cx_tok, device)
            else:
                chexbert_results[img_id][key] = chexbert_label_sentences(
                    sentences, cx_mdl, cx_tok, device,
                    batch_size=args.chexbert_batch_size)

    with open(args.output, "w") as f:
        json.dump(chexbert_results, f)
    print(f"Wrote CheXbert-labeled retrieval results for {len(chexbert_results)} "
          f"images to {args.output}")


if __name__ == "__main__":
    main()
