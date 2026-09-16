"""
Runs sentence retrieval with the public, frozen CXR-RePaiR checkpoint
(Endo et al.) against a candidate pool, for every image in the CheXpert
evaluation set (manuscript Sec. 3.1.2).

Requires a local clone of https://github.com/rajpurkarlab/CXR-RePaiR (for
its `clip` package) and the public checkpoint
`clip-imp-pretrained_128_6_after_4.pt` (see configs/datasets.md).

Usage:
    python -m evaluation.03_run_cxr_repair_retrieval \
        --cxr_repair_repo_dir /path/to/CXR-RePaiR \
        --checkpoint /path/to/clip-imp-pretrained_128_6_after_4.pt \
        --eval_df eval_outputs/eval_df_with_paths.json \
        --candidate_pool pool_outputs/candidate_pool_original.json \
        --output retrieval_outputs/retrieval_repair_original.json

Output JSON: {img_id (str): {"sentences": [top-k strings], "scores": [floats]}}
"""
import argparse
import json
import sys

import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


def load_repair_model(cxr_repair_repo_dir, checkpoint_path, device):
    if cxr_repair_repo_dir not in sys.path:
        sys.path.insert(0, cxr_repair_repo_dir)
    from clip.model import build_model  # noqa: E402  (path-dependent import)

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model = build_model(state_dict).float().to(device).eval()
    return model


def build_repair_preprocess():
    return Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                   (0.26862954, 0.26130258, 0.27577711)),
    ])


@torch.no_grad()
def encode_texts_repair(texts, model, device, batch_size=256):
    from clip import clip
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding candidate pool (RePaiR)'):
        batch = texts[i:i + batch_size]
        toks = clip.tokenize(batch, context_length=model.context_length).to(device)
        emb = model.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_images_repair(image_paths, model, preprocess, device, batch_size=32):
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack(
            [preprocess(Image.open(p).convert('L')) for p in batch_paths]
        ).to(device)
        emb = model.encode_image(imgs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


def retrieve_topk_repair(img_path, model, preprocess, txt_norm_pool, candidate_pool,
                          device, k=10):
    img_emb = encode_images_repair([img_path], model, preprocess, device, batch_size=1).to(device)
    sims = (img_emb @ txt_norm_pool.t()).squeeze(0)
    topk = torch.topk(sims, k=k)
    sents = [candidate_pool[idx] for idx in topk.indices.tolist()]
    scores = topk.values.tolist()
    return sents, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cxr_repair_repo_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_df", required=True)
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.eval_df) as f:
        eval_records = json.load(f)
    with open(args.candidate_pool) as f:
        candidate_pool = json.load(f)

    model = load_repair_model(args.cxr_repair_repo_dir, args.checkpoint, device)
    preprocess = build_repair_preprocess()
    txt_norm_pool = encode_texts_repair(candidate_pool, model, device).to(device)

    results = {}
    for r in tqdm(eval_records, desc="RePaiR"):
        img_id = r["img_id"]
        sents, scores = retrieve_topk_repair(
            r["Path"], model, preprocess, txt_norm_pool, candidate_pool, device, k=args.top_k)
        results[str(img_id)] = {"sentences": sents, "scores": scores}

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Wrote retrieval results for {len(results)} images to {args.output}")


if __name__ == "__main__":
    main()
