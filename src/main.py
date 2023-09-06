from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
from torch import dtype
import torch
from config import hparams
import os
from train import Model
from argparse import ArgumentParser
from recommenders.datasets import mind
from pytorch_lightning.callbacks import Callback
import datetime

parser = ArgumentParser("train args")
parser.add_argument("--gpu", default="0")
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--abs-path", default="./", type=str)
args = parser.parse_args()

os.environ["WANDB__SERVICE_WAIT"] = "300"

output_dir = os.path.join(
        args.abs_path,
        "lightning_logs/" +
        hparams['description'] +
        '_max_hist_' + str(hparams['data']['max_hist']) +
        '_neg_k_' + str(hparams['data']['neg_k']) +
        '_maxlen_' + str(hparams['data']['maxlen']) +
        '_size_' + hparams['data']['dataset_size'] + '_' +
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

model = Model(hparams, abs_path=args.abs_path)
checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,
    filename="{epoch}-{auroc:.2f}",
    save_top_k=1,
    verbose=True,
    monitor="auroc",
    mode="max",
    save_last=True,
    save_weights_only=True,
)

early_stop = EarlyStopping(
    monitor="auroc", min_delta=0.001, patience=5, strict=False, verbose=True, mode="max"
)
tb_logger = TensorBoardLogger(
    save_dir=output_dir,
    name="",
)
# initialise the wandb logger and name your wandb project
# wandb_logger = WandbLogger(project='MIND-benchmarking-framework')

# # add your batch size to the wandb config
# wandb_logger.experiment.config["batch_size"] = hparams["batch_size"]

trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback],
    val_check_interval=600,
    # check_val_every_n_epoch=1,
    logger=[tb_logger],
)

# model.load_state_dict(
#     torch.load(
#         os.path.join(args.abs_path, f"lightning_logs/20230823-123030/epoch=0-auroc=0.65.ckpt")
#     )["state_dict"]
# )

trainer.fit(model)
