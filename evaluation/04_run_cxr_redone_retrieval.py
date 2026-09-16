"""
Runs sentence retrieval with the public, frozen CXR-ReDonE checkpoint
(Ramesh, Chi, Rajpurkar 2022 - ALBEF retrieval, trained on MIMIC-PRO with
prior-study references removed) against a candidate pool, for every image
in the CheXpert evaluation set (manuscript Sec. 3.1.3).

Requires:
  - a local clone of https://github.com/rajpurkarlab/CXR-ReDonE (for its
    vendored ALBEF code under CXR-ReDonE/ALBEF)
  - the `checkpoint_59.pth` checkpoint (see configs/datasets.md)
  - `timm`, `pyyaml`, `h5py` installed

The ALBEF code in that repo was written against an older `transformers`
version. The compatibility shims below patch three functions
(`apply_chunking_to_forward`, `find_pruneable_heads_and_indices`,
`prune_linear_layer`) back into `transformers.modeling_utils` if missing,
patch a `transformers.file_utils` shim module for the same reason, and
patch `timm`'s `register_model`/`_cfg` if the installed timm version
removed them. These shims are required for the checkpoint architecture to
load correctly under a modern `transformers`/`timm` install and are kept
verbatim from the original evaluation run.

Usage:
    python -m evaluation.04_run_cxr_redone_retrieval \
        --cxr_redone_repo_dir /path/to/CXR-ReDonE \
        --checkpoint /path/to/checkpoint_59.pth \
        --eval_df eval_outputs/eval_df_with_paths.json \
        --candidate_pool pool_outputs/candidate_pool_original.json \
        --output retrieval_outputs/retrieval_redone_original.json

Output JSON: {img_id (str): {"sentences": [top-k strings], "scores": [floats]}}
"""
import argparse
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import BertTokenizer


def install_compatibility_shims(albef_dir):
    """Patches transformers/timm internals the vendored ALBEF code expects
    but which newer library versions removed. See module docstring."""
    import transformers
    import transformers.modeling_utils as _modeling_utils

    if not hasattr(_modeling_utils, "apply_chunking_to_forward"):
        import inspect

        def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
            if chunk_size > 0:
                num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
                chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
                outputs = tuple(forward_fn(*chunk_set) for chunk_set in zip(*chunks))
                return torch.cat(outputs, dim=chunk_dim)
            return forward_fn(*input_tensors)

        _modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward

    if not hasattr(_modeling_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        _modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(_modeling_utils, "prune_linear_layer"):
        def prune_linear_layer(layer, index, dim=0):
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone().detach()
            b = None
            if layer.bias is not None:
                b = (layer.bias.clone().detach() if dim == 1
                     else layer.bias[index].clone().detach())
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = nn.Linear(new_size[1], new_size[0],
                                   bias=layer.bias is not None).to(layer.weight.device)
            new_layer.weight.requires_grad = False
            new_layer.weight.copy_(W.contiguous())
            new_layer.weight.requires_grad = True
            if layer.bias is not None:
                new_layer.bias.requires_grad = False
                new_layer.bias.copy_(b.contiguous())
                new_layer.bias.requires_grad = True
            return new_layer

        _modeling_utils.prune_linear_layer = prune_linear_layer

    try:
        import transformers.file_utils as _futils
    except ModuleNotFoundError:
        _futils = types.ModuleType('transformers.file_utils')
        sys.modules['transformers.file_utils'] = _futils
        transformers.file_utils = _futils

    def _noop_decorator_factory(*a, **k):
        def _wrap(fn):
            return fn
        return _wrap

    for name in ['add_code_sample_docstrings', 'add_start_docstrings',
                 'add_start_docstrings_to_model_forward', 'replace_return_docstrings']:
        setattr(_futils, name, _noop_decorator_factory)
    if not hasattr(_futils, 'ModelOutput'):
        from transformers.utils import ModelOutput as _ModelOutput
        _futils.ModelOutput = _ModelOutput

    try:
        import timm.models.registry as _tregistry
        if not hasattr(_tregistry, 'register_model'):
            _tregistry.register_model = lambda fn: fn
    except ModuleNotFoundError:
        _tregistry = types.ModuleType('timm.models.registry')
        _tregistry.register_model = lambda fn: fn
        sys.modules['timm.models.registry'] = _tregistry

    import timm.models.vision_transformer as _tvit
    if not hasattr(_tvit, '_cfg'):
        def _cfg(url='', **kwargs):
            return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), **kwargs}
        _tvit._cfg = _cfg

    if albef_dir not in sys.path:
        sys.path.insert(0, albef_dir)
    for m in list(sys.modules):  # drop any stale 'models' module from a prior import
        if m == 'models' or m.startswith('models.'):
            del sys.modules[m]


def load_redone_model(cxr_redone_repo_dir, checkpoint_path, device):
    albef_dir = os.path.join(cxr_redone_repo_dir, 'ALBEF')
    install_compatibility_shims(albef_dir)

    from models.model_retrieval import ALBEF as ALBEF_Retrieval
    from models.vit import interpolate_pos_embed
    from models import xbert as _xbert

    _xbert.BertPreTrainedModel.tie_weights = lambda self, *a, **k: None

    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        # newer transformers removed this; xbert never actually passes a
        # non-None head_mask here, so a constant None list is equivalent.
        return [None] * num_hidden_layers

    _xbert.BertPreTrainedModel.get_head_mask = _get_head_mask

    config = yaml.load(open(os.path.join(albef_dir, 'configs', 'Retrieval_flickr.yaml')),
                        Loader=yaml.Loader)
    config['bert_config'] = os.path.join(albef_dir, 'configs', 'config_bert.json')
    config['image_res'] = 256  # matches the CXR-ReDonE cosine-sim retrieval resolution

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Fast-init: skip downloading pretrained bert-base-uncased weights since
    # the checkpoint below overwrites the whole state dict anyway.
    def _fast_bert_from_pretrained(name, *args, **kwargs):
        bert_cfg = kwargs.get('config')
        add_pooling = kwargs.get('add_pooling_layer', True)
        return _xbert.BertModel(config=bert_cfg, add_pooling_layer=add_pooling)

    orig_from_pretrained = _xbert.BertModel.from_pretrained
    _xbert.BertModel.from_pretrained = _fast_bert_from_pretrained
    model = ALBEF_Retrieval(config=config, text_encoder='bert-base-uncased', tokenizer=tokenizer)
    _xbert.BertModel.from_pretrained = orig_from_pretrained

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(
        state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(
        state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    for key in list(state_dict.keys()):
        if 'bert' in key:
            state_dict[key.replace('bert.', '')] = state_dict.pop(key)

    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    model = model.to(device).eval()

    return model, tokenizer


def build_redone_transform():
    # NOTE: raw pixel values 0-255, no ToTensor()/255 scaling - matches the
    # original HDF5-based training pipeline (see manuscript Sec. 3.1.3).
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.Normalize((101.48761, 101.48761, 101.48761),
                              (83.43944, 83.43944, 83.43944)),
    ])


def load_image_for_redone(image_path, redone_transform):
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float64)
    img = np.expand_dims(img, axis=0)
    img = np.repeat(img, 3, axis=0)
    img = torch.from_numpy(img).float()
    return redone_transform(img)


@torch.no_grad()
def encode_texts_redone(texts, model, tokenizer, device, batch_size=256, max_length=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc='Encoding candidate pool (ReDonE)'):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding='max_length', truncation=True,
                         max_length=max_length, return_tensors='pt').to(device)
        text_output = model.text_encoder(enc.input_ids, attention_mask=enc.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state[:, 0, :]
        text_embed = F.normalize(model.text_proj(text_feat), dim=-1)
        embs.append(text_embed.cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_images_redone(image_paths, model, redone_transform, device, batch_size=32):
    embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = torch.stack(
            [load_image_for_redone(p, redone_transform) for p in batch_paths]
        ).to(device)
        image_feat = model.visual_encoder(imgs)
        image_embed = F.normalize(model.vision_proj(image_feat[:, 0, :]), dim=-1)
        embs.append(image_embed.cpu())
    return torch.cat(embs, dim=0)


def retrieve_topk_redone(img_path, model, redone_transform, txt_norm_pool, candidate_pool,
                          device, k=10):
    img_emb = encode_images_redone([img_path], model, redone_transform, device, batch_size=1).to(device)
    sims = (img_emb @ txt_norm_pool.t()).squeeze(0)
    topk = torch.topk(sims, k=k)
    sents = [candidate_pool[idx] for idx in topk.indices.tolist()]
    scores = topk.values.tolist()
    return sents, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cxr_redone_repo_dir", required=True)
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

    model, tokenizer = load_redone_model(args.cxr_redone_repo_dir, args.checkpoint, device)
    redone_transform = build_redone_transform()
    txt_norm_pool = encode_texts_redone(candidate_pool, model, tokenizer, device).to(device)

    results = {}
    for r in tqdm(eval_records, desc="ReDonE"):
        img_id = r["img_id"]
        sents, scores = retrieve_topk_redone(
            r["Path"], model, redone_transform, txt_norm_pool, candidate_pool, device, k=args.top_k)
        results[str(img_id)] = {"sentences": sents, "scores": scores}

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Wrote retrieval results for {len(results)} images to {args.output}")


if __name__ == "__main__":
    main()
