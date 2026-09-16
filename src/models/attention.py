"""
Attention mechanisms for word-region alignment (WRA). `func_attention` is
used directly by `src.losses.words_loss`. `GlobalAttentionGeneral` is a
region-attention module retained from the JoImTeRNet/AttnGAN lineage this
codebase builds on; it is not called by the current training loop but is
kept here for reference and possible extension.
"""
import torch
import torch.nn as nn

from src.models.image_encoder import conv1x1


def func_attention(query, context, gamma1):
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw  = context.size(2), context.size(3)
    sourceL = ih * iw

    context  = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    attn = torch.bmm(contextT, query)
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(-1)(attn)

    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = attn * gamma1
    attn = nn.Softmax(-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)

    attnT           = torch.transpose(attn, 1, 2).contiguous()
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm           = nn.Softmax(-1)
        self.mask         = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        ih, iw     = input.size(2), input.size(3)
        queryL     = ih * iw
        batch_size = context.size(0)
        sourceL    = context.size(2)

        target  = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()

        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)

        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)

        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()

        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn            = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn

