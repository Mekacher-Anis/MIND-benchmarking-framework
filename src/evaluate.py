import pytorch_lightning as pl
from gensim.models import Word2Vec
import torch
from models.nrms import NRMS
from typing import List
from gaisTokenizer import Tokenizer
import argparse
from dataset import TestDataset
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api


class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.w2v: KeyedVectors = api.load(hparams["pretrained_model"])
        if hparams["model"]["dct_size"] == "auto":
            hparams["model"]["dct_size"] = len(self.w2v.key_to_index)
        self.model = NRMS(hparams["model"], torch.tensor(self.w2v.vectors))
        self.w2id = self.w2v.key_to_index
        
        self.hparams = hparams
        self.maxlen = hparams['data']['maxlen']
        self.tokenizer = Tokenizer('k95763565C5F785B50546754545D77505F0325160B58173C17291B3D5E2500135001671C06272B3B06281E1E5E55A9F7EB80C0E58AD1EB50AC')

    def forward(self, viewed, cands, topk):
        """forward

        Args:
            viewed (tensor): [B, viewed_num, maxlen]
            cands (tesnor): [B, cand_num, maxlen]
        Returns:
            val: [B] 0 ~ 1
            idx: [B] 
        """
        logits = self.model(viewed, cands)
        val, idx = logits.topk(topk)
        return idx, val
    
    def predict_one(self, viewed, cands, topk):
        """predict one user

        Args:
            viewed (List[List[str]]): 
            cands (List[List[str]]): 
        Returns:
            topk of cands
        """
        viewed_token = torch.tensor([self.sent2idx(v) for v in viewed]).unsqueeze(0)
        cands_token = torch.tensor([self.sent2idx(c) for c in cands]).unsqueeze(0)
        idx, val = self(viewed_token, cands_token, topk)
        val = val.squeeze().detach().cpu().tolist()

        result = [cands[i] for i in idx.squeeze()]
        return result, val
    
    def sent2idx(self, tokens: List[str]):
        if ']' in tokens:
            tokens = tokens[tokens.index(']'):]
        tokens = [self.w2id[token.strip()]
                  for token in tokens if token.strip() in self.w2id.keys()]
        tokens += [0] * (self.maxlen - len(tokens))
        tokens = tokens[:self.maxlen]
        return tokens
    
    def tokenize(self, sents: str):
        return self.tokenizer.tokenize(sents)


def print_func(r):
    for t in r:
        print(''.join(t))
    
if __name__ == '__main__':    
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./lightning_logs/ranger/v3/epoch=30-auroc=0.89.ckpt')
    args = parser.parse_args()
    

    nrms = Model.load_from_checkpoint(args.model)
    test_dataset = TestDataset('./data', None)
    
    for i in test_dataset:
        impid, viewed, cands = i
        result, val = nrms.predict_one(viewed, cands, len(cands)) 
        break

    
    
