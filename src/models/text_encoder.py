"""
Text encoder: a randomly initialized (NOT pretrained-BERT-weights) 3-layer
Transformer using the BERT-base-uncased WordPiece vocabulary, with dual
heads for masked-language-modeling (MLM) and image-text matching (ITM)
(manuscript Sec. 3.1.1).
"""
import torch.nn as nn
from transformers import BertModel


class MLMHead(nn.Module):
    def __init__(self, bert_config):
        super(MLMHead, self).__init__()
        self.transform = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.act       = nn.GELU()
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.decoder   = nn.Linear(bert_config.hidden_size, bert_config.vocab_size - 1)

    def forward(self, seq_feat):
        return self.decoder(self.LayerNorm(self.act(self.transform(seq_feat))))


class TextEncoder(nn.Module):
    def __init__(self, bert_config, output_channels, pool='cls'):
        super(TextEncoder, self).__init__()
        self.pool    = pool
        self.bert    = BertModel(bert_config, add_pooling_layer=False)
        self.mlm     = MLMHead(bert_config)
        self.sent_fc = nn.Linear(output_channels, output_channels)
        self.word_fc = nn.Linear(output_channels, output_channels)

    def forward(self, x, mask, task='itm'):
        seq_feat = self.bert(input_ids=x, attention_mask=mask)[0]
        if task == 'mlm':
            return self.forward_mlm(seq_feat)
        elif task == 'itm':
            return self.forward_itm(seq_feat, mask)
        else:
            raise ValueError(f'Invalid task: {task}')

    def forward_mlm(self, seq_feat):
        pred = self.mlm(seq_feat)
        return pred.transpose(1, 2)

    def forward_itm(self, seq_feat, mask):
        if self.pool == 'cls':
            sent_feat = self.sent_fc(seq_feat[:, 0])
        else:
            sent_feat = (
                (seq_feat * mask.unsqueeze(-1)).sum(1)
                / mask.sum(1).unsqueeze(-1)
            )
            sent_feat = self.sent_fc(sent_feat)
        word_feat = self.word_fc(seq_feat)
        return word_feat.transpose(1, 2), sent_feat

