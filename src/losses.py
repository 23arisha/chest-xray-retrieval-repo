"""
Loss functions for JoImTeRNet-style joint image-sentence training:
  - sentence-level bidirectional cross-entropy matching (`sent_loss`)
  - word-region alignment bidirectional cross-entropy matching (`words_loss`)
  - bidirectional triplet losses at both sentence and word level
    (`sent_triplet_loss`, `words_triplet_loss`)

All temperature/margin hyperparameters (GAMMA1-3, sent_margin, word_margin)
are read from the shared `cfg` singleton in `configs.train_config`, matching
the manuscript's reported values (Sec. 3.1.1).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

from configs.train_config import cfg
from src.models.attention import func_attention


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1  = torch.norm(x1, 2, dim)
    w2  = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def cosine_distance(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1  = torch.norm(x1, 2, dim)
    w2  = torch.norm(x2, 2, dim)
    return 1 - (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, eps=1e-8):
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask    = (class_ids == class_ids[i]).astype(np.bool_)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    scores0       = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0         = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0       = scores0 / norm0.clamp(min=eps) * cfg.GAMMA3
    scores0       = scores0.squeeze()

    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)

    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    masks        = []
    att_maps     = []
    similarities = []
    cap_lens     = cap_lens.data.tolist()

    for i in range(batch_size):
        if class_ids is not None:
            mask    = (class_ids == class_ids[i]).astype(np.bool_)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]
        word      = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        word      = word.repeat(batch_size, 1, 1)

        weiContext, attn = func_attention(word, img_features, cfg.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())

        word       = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        word       = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)
        row_sim.mul_(cfg.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)

    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.BoolTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)

    loss0 = nn.CrossEntropyLoss()(similarities,  labels)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps


def triplet_loss_with_cosine_distance(anc, pos, neg, margin=0.5):
    score = cosine_distance(anc, pos) - cosine_distance(anc, neg) + margin
    return TF.relu(score)


def sent_triplet_loss(cnn_code, rnn_code, labels, neg_ids, batch_size):
    i2t_loss = triplet_loss_with_cosine_distance(
        cnn_code[labels], rnn_code[labels], rnn_code[neg_ids],
        margin=cfg.sent_margin
    ).mean()
    t2i_loss = triplet_loss_with_cosine_distance(
        rnn_code[labels], cnn_code[labels], cnn_code[neg_ids],
        margin=cfg.sent_margin
    ).mean()
    return i2t_loss, t2i_loss


def word_similarity(img_features, words_emb, words_num):
    word             = words_emb[0, :, :words_num].unsqueeze(0).contiguous()
    weiContext, attn = func_attention(word, img_features, cfg.GAMMA1)
    att_maps         = attn[0].unsqueeze(0).contiguous()

    word       = word.transpose(1, 2).contiguous()
    weiContext = weiContext.transpose(1, 2).contiguous()
    word       = word.view(1 * words_num, -1)
    weiContext = weiContext.view(1 * words_num, -1)

    row_sim = cosine_similarity(word, weiContext)
    row_sim = row_sim.view(1, words_num)
    row_sim.mul_(cfg.GAMMA2).exp_()
    row_sim = row_sim.sum(dim=1, keepdim=True)
    row_sim = -torch.log(row_sim)

    return row_sim, att_maps


def triplet_loss_with_word_similarity(anc, pos, neg, words_num, flag='img', margin=0.5):
    if flag == 'img':
        pos_score, pos_attn = word_similarity(anc, pos, words_num)
        neg_score, _        = word_similarity(anc, neg, words_num)
    else:
        pos_score, pos_attn = word_similarity(pos, anc, words_num)
        neg_score, _        = word_similarity(neg, anc, words_num)

    score = pos_score - neg_score + margin
    return TF.relu(score), pos_attn


def words_triplet_loss(img_features, words_emb, labels, neg_ids, cap_lens, batch_size):
    i2t_losses = []
    t2i_losses = []
    attn_maps  = []

    for i in range(batch_size):
        img_anchor    = img_features[labels[i:i+1]]
        text_positive = words_emb[labels[i:i+1]]
        text_negative = words_emb[neg_ids[i:i+1]]

        i2t_loss, img_pos_attn = triplet_loss_with_word_similarity(
            img_anchor, text_positive, text_negative,
            cap_lens[i], flag='img', margin=cfg.word_margin
        )
        i2t_losses.append(i2t_loss)
        attn_maps.append(img_pos_attn)

        text_anchor  = words_emb[labels[i:i+1]]
        img_positive = img_features[labels[i:i+1]]
        img_negative = img_features[neg_ids[i:i+1]]

        t2i_loss, _ = triplet_loss_with_word_similarity(
            text_anchor, img_positive, img_negative,
            cap_lens[i], flag='text', margin=cfg.word_margin
        )
        t2i_losses.append(t2i_loss)

    i2t_loss = torch.cat(i2t_losses, 1).mean()
    t2i_loss = torch.cat(t2i_losses, 1).mean()
    return i2t_loss, t2i_loss, attn_maps

