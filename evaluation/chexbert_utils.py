"""
CheXbert loading and sentence-labeling utilities, used to:
  (a) label the retrieval candidate pool itself (evaluation/01_build_candidate_pools.py), and
  (b) label the top-k sentences retrieved by each model for each evaluation
      image (evaluation/05_chexbert_label_retrievals.py).

Requires a local clone of https://github.com/stanfordmlgroup/CheXbert and
the public `chexbert.pth` checkpoint (see configs/datasets.md).
"""
import importlib.util
import os
from collections import OrderedDict

import torch
from transformers import BertModel, BertConfig, BertTokenizer
from tqdm import tqdm

from evaluation.common import CHEXPERT_LABELS

# CheXbert's 14 condition names, in the exact order its output logits use.
# Identical to CHEXPERT_LABELS here since both follow the CheXpert convention.
CHEXBERT_CONDITIONS = CHEXPERT_LABELS


def _load_bert_labeler_class(chexbert_repo_dir):
    bert_file = os.path.join(chexbert_repo_dir, "src", "models", "bert_labeler.py")
    spec = importlib.util.spec_from_file_location("bert_labeler", bert_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.bert_labeler


def load_chexbert(chexbert_repo_dir, chexbert_ckpt_path, device):
    """Loads the pretrained CheXbert labeler.

    `chexbert_repo_dir` must point to a local clone of the CheXbert repo
    (https://github.com/stanfordmlgroup/CheXbert), which provides the
    `bert_labeler` model class. The pretrained-BERT weight download is
    skipped (fast-init) since `chexbert_ckpt_path` overwrites the whole
    state dict anyway.
    """
    bert_labeler = _load_bert_labeler_class(chexbert_repo_dir)

    cx_tok = BertTokenizer.from_pretrained("bert-base-uncased")

    orig_from_pretrained = BertModel.from_pretrained

    def _fast_from_pretrained(name, *args, **kwargs):
        return BertModel(BertConfig.from_pretrained(name))

    BertModel.from_pretrained = _fast_from_pretrained
    cx_mdl = bert_labeler()
    BertModel.from_pretrained = orig_from_pretrained

    ckpt = torch.load(chexbert_ckpt_path, map_location="cpu")
    state = OrderedDict(
        (k[7:] if k.startswith("module.") else k, v)
        for k, v in ckpt["model_state_dict"].items()
    )
    cx_mdl.load_state_dict(state, strict=True)
    cx_mdl.eval()
    cx_mdl = cx_mdl.to(device)

    return cx_mdl, cx_tok


@torch.no_grad()
def chexbert_label_sentences(sentences, cx_mdl, cx_tok, device,
                              batch_size=32, show_progress=False,
                              condition_names=CHEXBERT_CONDITIONS):
    """Labels each sentence with CheXbert's per-condition prediction.

    Returns a list (one dict per sentence) of {condition_name: predicted_class},
    taking the raw argmax of each condition head as-is. Downstream code
    (evaluation/05_chexbert_label_retrievals.py, tables/generate_tables.py)
    treats a value of 1 as "positive for this finding", matching the
    manuscript's convention.
    """
    all_preds = []
    iterator = range(0, len(sentences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="CheXbert labeling", leave=False)

    for i in iterator:
        batch = sentences[i:i + batch_size]
        enc = cx_tok(batch, padding="max_length", truncation=True,
                      max_length=128, return_tensors="pt")
        logits = cx_mdl(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        batch_preds = [torch.argmax(l, dim=1).cpu().tolist() for l in logits]
        for j in range(len(batch)):
            all_preds.append({
                condition_names[c]: batch_preds[c][j]
                for c in range(len(condition_names))
            })
    return all_preds
