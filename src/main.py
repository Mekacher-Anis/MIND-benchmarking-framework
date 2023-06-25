from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch import dtype
from config import hparams
import os
from train import Model
from argparse import ArgumentParser
from recommenders.datasets import mind


parser = ArgumentParser("train args")
parser.add_argument("--gpu", default="0")
parser.add_argument("--epochs", default=50, type=int)
args = parser.parse_args()


model = Model(hparams)
checkpoint_callback = ModelCheckpoint(
    dirpath=f'lightning_logs/{hparams["name"]}/{hparams["version"]}/',
    filename="{epoch}-{auroc:.2f}",
    save_top_k=3,
    verbose=True,
    monitor="auroc",
    mode="max",
    save_last=True,
)

early_stop = EarlyStopping(
    monitor="auroc", min_delta=0.001, patience=5, strict=False, verbose=True, mode="max"
)
logger = TensorBoardLogger(
    save_dir="lightning_logs", name=hparams["name"], version=hparams["version"]
)

trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop, checkpoint_callback],
    logger=logger,
)

trainer.fit(model)
