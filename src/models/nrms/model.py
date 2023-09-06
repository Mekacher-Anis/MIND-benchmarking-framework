import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nrms.doc_encoder import DocEncoder
from models.nrms.attention import  AdditiveAttention
from models.nrms.mha import MultiheadAttention


class NRMS(nn.Module):
    def __init__(self, hparams, weight=None, p=.2):
        super(NRMS, self).__init__()
        self.hparams = hparams
        self.p = p
        self.doc_encoder = DocEncoder(hparams, weight=weight)
        self.mha = MultiheadAttention(
            input_dim=hparams["encoder_size"],
            num_heads=hparams["nhead"],
            embed_dim=hparams["encoder_size"]
        ).cuda()
        self.additive_attn = AdditiveAttention(hparams['encoder_size'], hparams['v_size']).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.dropout = torch.nn.Dropout(self.p).cuda()

    def forward(self, clicks, cands, labels=None):
        """forward

        Args:
            clicks (tensor): [num_user, num_click_docs, seq_len]
            cands (tensor): [num_user, num_candidate_docs, seq_len]
        """
        num_click_docs = clicks.shape[1]
        num_cand_docs = cands.shape[1]
        num_user = clicks.shape[0]
        seq_len = clicks.shape[2]
        clicks = clicks.reshape(-1, seq_len) # [B, seq_len]
        cands = cands.reshape(-1, seq_len) # [B, seq_len]
        click_embed = self.doc_encoder(clicks) # [B, encoder_size]
        cand_embed = self.doc_encoder(cands) # [B, encoder_size]
        click_embed = click_embed.reshape(num_user, num_click_docs, -1) # [B, num_click_docs, encoder_size]
        cand_embed = cand_embed.reshape(num_user, num_cand_docs, -1) # [B, num_cand_docs, encoder_size]
        click_output, _ = self.mha(click_embed) # [B, num_click_docs, encoder_size]
        click_output = self.dropout(click_output) # [B, num_click_docs, encoder_size]

        click_repr, _ = self.additive_attn(click_output) # [B, encoder_size]
        logits = torch.bmm(click_repr.unsqueeze(1), cand_embed.permute(0, 2, 1)).squeeze(1).cuda() # [B, num_cand_docs]
        if labels is not None:
            labels = labels.cuda()
            loss = self.criterion(logits, labels)
            return loss, logits
        return logits
