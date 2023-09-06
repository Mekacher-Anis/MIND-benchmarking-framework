import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nrms.attention import AdditiveAttention
from models.nrms.mha import MultiheadAttention


class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None, p=0.2) -> None:
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        self.p = p
        if weight is None:
            self.embedding = nn.Embedding(100, 300, device="cuda").cuda()
        else:
            print('Loading weights from pretrained embeddings')
            self.embedding = nn.Embedding.from_pretrained(
                weight.cuda(), freeze=False, padding_idx=0
            ).cuda()
        self.mha = MultiheadAttention(
            input_dim=hparams["embed_size"],
            num_heads=hparams["nhead"],
            embed_dim=hparams["encoder_size"]
        ).cuda()
        self.additive_attn = AdditiveAttention(
            hparams["encoder_size"], hparams["v_size"]
        ).cuda()
        self.dropout = torch.nn.Dropout(self.p).cuda()

    def forward(self, x):
        x = x.cuda()
        x = self.dropout(self.embedding(x)).cuda() # [B, seq_len, embed_size]
        output, _ = self.mha(x) # [B, seq_len, encoder_size]
        output = self.dropout(output).cuda()
        output, _ = self.additive_attn(output) # [B, encoder_size]
        return output.cuda()
