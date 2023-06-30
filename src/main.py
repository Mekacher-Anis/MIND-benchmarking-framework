from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
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
parser.add_argument('--abs-path', default="./", type=str)
args = parser.parse_args()

os.environ["WANDB__SERVICE_WAIT"] = "300"



model = Model(hparams)
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(args.abs_path, f'lightning_logs/{hparams["name"]}/{hparams["version"]}/'),
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
tb_logger = TensorBoardLogger(
    save_dir=os.path.join(args.abs_path, "lightning_logs"), name=hparams["name"], version=hparams["version"]
)
# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project='MIND-benchmarking-framework')

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = hparams["batch_size"]

trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop, checkpoint_callback],
    logger=[tb_logger, wandb_logger],
)

trainer.fit(model)
