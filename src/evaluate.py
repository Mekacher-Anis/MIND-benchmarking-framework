import pytorch_lightning as pl
import torch
from dataset import TestDataset
from models.fastformernrms.model import FastformerNRMS
import argparse
from gensim.models import KeyedVectors
import gensim.downloader as api
import json
from tqdm import tqdm
from torch.utils import data
import os
import numpy as np
from pytorch_lightning.callbacks import BasePredictionWriter
import datetime

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.w2v: KeyedVectors = api.load(hparams["pretrained_model"])
        if hparams["model"]["dct_size"] == "auto":
            hparams["model"]["dct_size"] = len(self.w2v.key_to_index)
        self.model = FastformerNRMS(hparams["model"], torch.tensor(self.w2v.vectors))
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
        return idx, val, np.argsort(logits.detach().cpu().numpy(), axis=1)
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        impid, viewed, cands, cands_mask = batch
        logits = self.model(viewed, cands)
        res = []
        for _impid, _logits, _cand_mask in zip(impid, logits, cands_mask):
            _logits = torch.masked_select(_logits, _cand_mask).detach().cpu().numpy()
            _preds = (np.argsort(np.argsort(_logits)[::-1]) + 1).tolist()
            res.append({'id': _impid, 'pred': _preds})
        return res
    
    def predict_one(self, viewed: torch.Tensor, cands: torch.Tensor, topk: int):
        """predict one user

        Args:
            viewed (List[List[str]]): 
            cands (List[List[str]]): 
        Returns:
            topk of cands
        """
        # idx, val, sorted_idx = self(viewed, cands, topk)
        # val = val.squeeze().detach().cpu().tolist()
        return self.model(viewed, cands)


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))


    
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./lightning_logs/ranger/v3/epoch=30-auroc=0.89.ckpt')
    parser.add_argument('--abs-path', default="./", type=str)
    args = parser.parse_args()
    device = torch.device('cuda')
    torch.multiprocessing.set_start_method('spawn')
    # logging.basicConfig(filename=os.path.join(args.abs_path, 'output', 'evaluation.log'), format='%(asctime)s %(message)s', level=print)
    
    print('Running main...')
    
    
    with torch.no_grad():
        print('Loading dataset...')
        test_ds = TestDataset(os.path.join(args.abs_path, 'data/large/test'), dataset_size='large', device=device)
        batch_size = 100
        test_dl = data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=6,
            drop_last=False,
            shuffle=False
        )
        
        print('Loding checkpoint...')
        model: Model = Model.load_from_checkpoint(os.path.join(args.abs_path, args.model))
        model = model.to(device)
        model.eval()
        model.model.p = 0.0
        # nrms.model.mha.dropout = 0.0
        model.model.dropout.p = 0.0
        model.model.doc_encoder.dropout.p = 0.0
        # nrms.model.doc_encoder.mha.dropout = 0.0
        
        print('Starting test...')
        # preds = trainer.predict(model, test_dl)
        
        preds = []
        for i in tqdm(test_dl, total=(2370727 // batch_size)+1):
            if not i: break
            impid, viewed, cands, cands_mask = i
            logits = model.predict_one(viewed, cands, cands.shape[1])
            
            for _impid, _logits, _cand_mask in zip(impid, logits, cands_mask):
                _logits = torch.masked_select(_logits, _cand_mask).detach().cpu().numpy()
                _preds = (np.argsort(np.argsort(_logits)[::-1]) + 1).tolist()
                preds.append({'id': _impid, 'preds': _preds})
            del viewed
            del cands
            
        preds = sorted(preds, key=lambda x: int(x['id']))
        with open(os.path.join(args.abs_path, 'output', 'prediction.txt'), 'w') as f:
            for l in preds:
                preds_str = json.dumps(l['preds'], separators=(',', ':'))
                f.write(f"{l['id']} {preds_str}\n")