import pytorch_ranger
import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_ranger import Ranger
from dataset import Dataset, ValDataset
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from metric import ndcg_score, mrr_score
from recommenders.datasets import mind
import os
import torchmetrics
from models.fastformernrms.model import FastformerNRMS


class Model(pl.LightningModule):
    def __init__(self, hparams, abs_path = './'):
        super(Model, self).__init__()
        self.w2v: KeyedVectors = api.load(hparams["pretrained_model"])
        if hparams["model"]["dct_size"] == "auto":
            hparams["model"]["dct_size"] = len(self.w2v.key_to_index)
        self.model = FastformerNRMS(hparams["model"], torch.tensor(self.w2v.vectors))
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False
        self.training_step_outputs = []
        self.abs_path = abs_path

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=1e-5)
        # return pytorch_ranger.Ranger(
        #     self.parameters(), lr=self.hparams["lr"], weight_decay=1e-5
        # )

    def prepare_data(self):
        """prepare_data

        load dataset
        """
        d = self.hparams["data"]
        self.train_ds = Dataset(
            os.path.join(self.abs_path, "data"),
            self.w2v,
            maxlen=self.hparams["data"]["maxlen"],
            pos_num=d["max_hist"],
            neg_k=d["neg_k"],
            dataset_size=d["dataset_size"],
        )
        self.val_ds = ValDataset(
            5,
            os.path.join(self.abs_path, "data"),
            self.w2v,
            maxlen=self.hparams["data"]["maxlen"],
            pos_num=d["max_hist"],
            neg_k=d["neg_k"],
            dataset_size=d["dataset_size"],
        )
        tmp = [t.unsqueeze(0) for t in self.train_ds[0]]
        self.logger.experiment.add_graph(self.model, tmp)
        # num_train = int(len(ds) * 0.85)
        # num_val = len(ds) - num_train
        # self.train_ds, self.val_ds = data.random_split(ds, (num_train, num_val))

    def train_dataloader(self):
        """

        return:
            train_dataloader
        """
        return data.DataLoader(
            self.train_ds,
            batch_size=self.hparams["batch_size"],
            num_workers=2,
            shuffle=True,
        )

    def val_dataloader(self):
        """

        return:
            val_dataloader
        """
        sampler = data.RandomSampler(self.val_ds, num_samples=10000, replacement=True)
        return data.DataLoader(
            self.val_ds,
            sampler=sampler,
            batch_size=self.hparams["batch_size"],
            num_workers=2,
            drop_last=True,
            shuffle=False
        )

    def forward(self):
        """forward
        define as normal pytorch model
        """
        return None

    def training_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        clicks, cands, labels = batch
        loss, score = self.model(clicks, cands, labels)

        # Insert these lines:
        self.manual_backward(loss)
        
        optimizer = self.optimizers()

        optimizer.step()
        optimizer.zero_grad()
        
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {"loss": loss}

    def on_train_epoch_end(self):
        """for each epoch end

        Arguments:
            outputs: list of training_step output
        """
        loss_mean = torch.stack(self.training_step_outputs).mean()
        logs = {"train_loss": loss_mean}
        self.log("train_loss", loss_mean, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.model.eval()
        # self.logger.log_metrics(logs, self.current_epoch)
        return {"progress_bar": logs, "log": logs}
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_output_list = []
        return

    def validation_step(self, batch, batch_idx):
        """for each step(batch)

        Arguments:
            batch {[type]} -- data
            batch_idx {[type]}

        """
        clicks, cands, cands_label = batch
        with torch.no_grad():
            logits = self.model(clicks, cands)
        mrr = 0.0
        auc = 0.0
        ndcg5, ndcg10 = 0.0, 0.0

        for score, label in zip(logits, cands_label):
            # auc += p.auroc(score, label)
            auc += torchmetrics.functional.auroc(score, label, 'binary')
            score = score.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            mrr += mrr_score(label, score)
            ndcg5 += ndcg_score(label, score, 5)
            ndcg10 += ndcg_score(label, score, 10)
        results =  {
            "auroc": (auc / logits.shape[0]).item(),
            "mrr": (mrr / logits.shape[0]).item(),
            "ndcg5": (ndcg5 / logits.shape[0]).item(),
            "ndcg10": (ndcg10 / logits.shape[0]).item(),
        }

        self.val_output_list.append(results)

        return results

    def on_validation_epoch_end(self):
        """
        validation end

        Arguments:
            outputs: list of training_step output
        """
        mrr = torch.tensor([x["mrr"] for x in self.val_output_list])
        auroc = torch.tensor([x["auroc"] for x in self.val_output_list])
        ndcg5 = torch.tensor([x["ndcg5"] for x in self.val_output_list])
        ndcg10 = torch.tensor([x["ndcg10"] for x in self.val_output_list])

        logs = {
            "auroc": auroc.mean(),
            "mrr": mrr.mean(),
            "ndcg@5": ndcg5.mean(),
            "ndcg@10": ndcg10.mean(),
        }
        for k, v in logs.items():
            self.log(k, v, prog_bar=True, logger=True)
        self.model.train()
        return {"progress_bar": logs, "log": logs}


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from config import hparams
    import os

    model = Model(hparams)

    trainer = Trainer(max_epochs=50, accelerator="gpu", devices=1)

    trainer.fit(model)
