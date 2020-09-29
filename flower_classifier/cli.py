"""
You can run this script by calling `flower_classifier` in your terminal window.
"""

import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from flower_classifier import ROOT_DATA_DIR
from flower_classifier.datasets.oxford_flowers import OxfordFlowersDataModule
from flower_classifier.models.baseline import BaselineResnet
from flower_classifier.models.classifier import FlowerClassifier

app = typer.Typer()


@app.command()
def train(
    learning_rate: float = 0.01,
    batch_size: int = 32,
    max_epochs: int = 1000,
    gpu: bool = False,
    data_dir: str = ROOT_DATA_DIR,
):
    network = BaselineResnet()
    model = FlowerClassifier(network=network, learning_rate=learning_rate)
    data_module = OxfordFlowersDataModule(batch_size=batch_size, data_dir=data_dir)
    logger = WandbLogger(project="flowers", tags=["oxford102"])
    checkpoint_callback = ModelCheckpoint(save_top_k=3, filepath=logger.experiment.dir)
    trainer_args = {
        "gpus": -1 if gpu else None,
        "max_epochs": max_epochs,
        "logger": logger,
        "row_log_interval": 1,
        "checkpoint_callback": checkpoint_callback,
    }
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    app()
