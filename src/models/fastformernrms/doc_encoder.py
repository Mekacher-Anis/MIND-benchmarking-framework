import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nrms.attention import  AdditiveAttention
from models.fastformer import Fastformer


class DocEncoder(nn.Module):
    def __init__(self, hparams, weight=None) -> None:
        super(DocEncoder, self).__init__()
        self.hparams = hparams
        if weight is None:
            self.embedding = nn.Embedding(100, 300, device='cuda').cuda()
        else:
            self.embedding = nn.Embedding.from_pretrained(weight.cuda(), freeze=False, padding_idx=0).cuda()
        self.mha = Fastformer(dim = hparams['embed_size'], heads = hparams['nhead'], max_seq_len = hparams['maxlen']).cuda()
        self.proj = nn.Linear(hparams['embed_size'], hparams['encoder_size']).cuda()
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size']).cuda()
    
    def forward(self, x):
        x = x.cuda()
        mask = torch.ones(x.shape).cuda().bool()
        x = F.dropout(self.embedding(x), 0.2).cuda()
        output = self.mha(x, mask=mask)
        output = F.dropout(output).cuda()
        output = self.proj(output).cuda()
        output, _ = self.additive_attn(output)
        return output.cuda()
