"""
Runs sentence retrieval with "our model" (the JoImTeRNet-based retriever
trained by training/train.py) against a candidate pool, for every image in
the CheXpert evaluation set.

Reuses the exact model classes from `src/` (ImageEncoder, TextEncoder) --
unlike the original notebook, which redefined these classes a second time
for evaluation. Using the same source guarantees the evaluation-time
architecture can never silently drift from the trained checkpoint's
architecture.

Usage:
    python -m evaluation.02_run_our_model_retrieval \
        --eval_df eval_outputs/eval_df.json \
        --candidate_pool pool_outputs/candidate_pool_original.json \
        --image_encoder_ckpt /path/to/image_encoder_best.pth \
        --text_encoder_ckpt /path/to/text_encoder_best.pth \
        --output retrieval_outputs/retrieval_ours_original.json

Output JSON: {img_id (str): {"sentences": [top-k strings], "scores": [floats]}}
"""
import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from src.data.datasets import val_transform
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder

HIDDEN_DIM = 512


def build_bert_config(vocab_size):
    return BertConfig(
        vocab_size=vocab_size, hidden_size=HIDDEN_DIM, num_hidden_layers=3,
        num_attention_heads=8, intermediate_size=2048, hidden_act='gelu',
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=512, layer_norm_eps=1e-12, initializer_range=0.02,
        type_vocab_size=2, pad_token_id=0,
    )


def load_our_model(image_encoder_ckpt, text_encoder_ckpt, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_config = build_bert_config(tokenizer.vocab_size)

    image_encoder = ImageEncoder(output_channels=HIDDEN_DIM).to(device)
    text_encoder = TextEncoder(bert_config, output_channels=HIDDEN_DIM).to(device)

    img_state = torch.load(image_encoder_ckpt, map_location=device)
    image_encoder.load_state_dict(img_state['model'])
    txt_state = torch.load(text_encoder_ckpt, map_location=device)
    text_encoder.load_state_dict(txt_state['model'])
    image_encoder.eval()
    text_encoder.eval()
    print("Loaded checkpoints, epoch:", img_state.get('epoch'), txt_state.get('epoch'))

    return image_encoder, text_encoder, tokenizer


@torch.no_grad()
def encode_candidate_pool(candidate_pool, text_encoder, tokenizer, device, batch_size=256):
    all_sent_feats = []
    for i in tqdm(range(0, len(candidate_pool), batch_size), desc='Encoding candidate pool'):
        batch_sents = candidate_pool[i:i + batch_size]
        enc = tokenizer(batch_sents, padding='max_length', truncation=True,
                         max_length=64, return_tensors='pt').to(device)
        _, sent_feats = text_encoder(enc['input_ids'], enc['attention_mask'], task='itm')
        all_sent_feats.append(sent_feats.cpu())
    all_sent_feats = torch.cat(all_sent_feats, dim=0)
    return F.normalize(all_sent_feats, dim=1).to(device)


@torch.no_grad()
def retrieve_topk(img_path, image_encoder, txt_norm_pool, candidate_pool, device, k=10):
    image = Image.open(img_path).convert('L')
    img_tensor = val_transform(image).unsqueeze(0).to(device)
    _, global_feat, _, _, _, _ = image_encoder(img_tensor)
    img_norm = F.normalize(global_feat, dim=1)
    sims = (img_norm @ txt_norm_pool.t()).squeeze(0)
    topk = torch.topk(sims, k=k)
    sents = [candidate_pool[idx] for idx in topk.indices.tolist()]
    scores = topk.values.tolist()
    return sents, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_df", required=True,
                         help="eval_df_with_paths.json from evaluation/00_build_eval_set.py "
                              "(includes each image's 'Path' field, required here).")
    parser.add_argument("--candidate_pool", required=True)
    parser.add_argument("--image_encoder_ckpt", required=True)
    parser.add_argument("--text_encoder_ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.eval_df) as f:
        eval_records = json.load(f)
    with open(args.candidate_pool) as f:
        candidate_pool = json.load(f)

    image_encoder, text_encoder, tokenizer = load_our_model(
        args.image_encoder_ckpt, args.text_encoder_ckpt, device)
    txt_norm_pool = encode_candidate_pool(candidate_pool, text_encoder, tokenizer, device)

    results = {}
    for r in tqdm(eval_records, desc="Ours"):
        img_id = r["img_id"]
        sents, scores = retrieve_topk(
            r["Path"], image_encoder, txt_norm_pool, candidate_pool, device, k=args.top_k)
        results[str(img_id)] = {"sentences": sents, "scores": scores}

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Wrote retrieval results for {len(results)} images to {args.output}")


if __name__ == "__main__":
    main()
