#!python
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
from torchtext.data import get_tokenizer
import json
from tqdm import tqdm
from torch.utils import data
import os
import logging

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.w2v: KeyedVectors = api.load(hparams["pretrained_model"])
        if hparams["model"]["dct_size"] == "auto":
            hparams["model"]["dct_size"] = len(self.w2v.key_to_index)
        self.model = NRMS(hparams["model"], torch.tensor(self.w2v.vectors))
        self.w2id = self.w2v.key_to_index
        self.hparams.update(hparams)
        self.maxlen = hparams['data']['maxlen']

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
    
    def predict_one(self, viewed: torch.Tensor, cands: torch.Tensor, topk: int):
        """predict one user

        Args:
            viewed (List[List[str]]): 
            cands (List[List[str]]): 
        Returns:
            topk of cands
        """
        idx, val = self(viewed, cands, topk)
        # val = val.squeeze().detach().cpu().tolist()

        # result = [cands[i] for i in idx.squeeze()]
        return None, val, idx.detach().cpu().tolist()


    
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./lightning_logs/ranger/v3/epoch=30-auroc=0.89.ckpt')
    parser.add_argument('--abs-path', default="./", type=str)
    args = parser.parse_args()
    device = torch.device('cuda')
    logging.basicConfig(filename=os.path.join(args.abs_path, 'output', 'evaluation.log'), format='%(asctime)s %(message)s', level=logging.DEBUG)
    
    logging.debug('Running main...')
    
    
    with torch.no_grad():
        logging.debug('Loding checkpoint...')
        nrms = Model.load_from_checkpoint(args.model)
        nrms = nrms.to(device)
        logging.debug('Loading dataset...')
        test_ds = TestDataset(os.path.join(args.abs_path, 'data/large/test'), nrms.w2v, dataset_size='large', device=device)

        logging.debug('Starting test...')
        with open(os.path.join(args.abs_path, 'output', 'prediction.txt'), 'w') as f:
            for i in tqdm(test_ds):
                if not i: break
                impid, viewed, cands = i
                logging.debug(f'Doing batch which contains {len(impid)} test samples')
                result, val, orig_idx = nrms.predict_one(viewed, cands, cands.shape[1])
                logging.debug(f'Done with batch')
                for _impid,_cand,_orig_idx in zip(impid, cands, orig_idx):
                    preds = [_orig_idx.index(idx) + 1 for idx in range(_cand.shape[0])]
                    preds_str = json.dumps(preds, separators=(',', ':'));
                    f.write(f'{_impid.item()} {preds_str}\n')