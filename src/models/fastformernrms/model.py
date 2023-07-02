import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fastformernrms.doc_encoder import DocEncoder
from models.nrms.attention import  AdditiveAttention
from models.fastformer import Fastformer


class FastformerNRMS(nn.Module):
    def __init__(self, hparams, weight=None):
        super(FastformerNRMS, self).__init__()
        self.hparams = hparams
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = Fastformer(dim = hparams['encoder_size'], heads = hparams['nhead'], max_seq_len = hparams['maxlen']).cuda()
        self.proj = nn.Linear(hparams['encoder_size'], hparams['encoder_size']).cuda()
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size']).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, clicks, cands, labels=None):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        print('Clicks.shape : ',  clicks.shape)
        print('cands.shape : ',  cands.shape)
        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        seq_len = clicks.shape[2]
        clicks = clicks.reshape(-1, seq_len)
        cands = cands.reshape(-1, seq_len)
        click_embed = self.doc_encoder(clicks)
        cand_embed = self.doc_encoder(cands)
        click_embed = click_embed.reshape(num_user, num_click_docs, -1)
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1)
        # mask = torch.ones(click_embed.shape[1:]).cuda().bool()
        mask = torch.ones(click_embed.shape[:2]).cuda().bool()
        click_output = self.mha(click_embed, mask=mask)
        click_output = F.dropout(click_output, 0.2)

        click_repr = self.proj(click_output)
        click_repr, _ = self.additive_attn(click_output)
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1).cuda() # [B, 1, hid], [B, 10, hid]
        if labels is not None:
            labels = labels.cuda()
            loss = self.criterion(logits, labels)
            return loss, logits
        return torch.sigmoid(logits)
        # return torch.softmax(logits, -1)
