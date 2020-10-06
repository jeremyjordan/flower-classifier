"""
You can run this script by calling `flower_classifier` in your terminal window.
"""
from pprint import pformat

import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from flower_classifier import ROOT_DATA_DIR
from flower_classifier.datasets.oxford_flowers import OxfordFlowersDataModule
from flower_classifier.models.classifier import FlowerClassifier

app = typer.Typer()


@app.command()
def train(
    architecture: str = "resnet34",
    dropout_rate: float = 0.0,
    global_pool: str = "avg",
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 1000,
    gpu: bool = False,
    smoke_test: bool = False,
    log_run: bool = True,
    data_dir: str = ROOT_DATA_DIR,
):
    data_module = OxfordFlowersDataModule(batch_size=batch_size, data_dir=data_dir)
    model = FlowerClassifier(
        architecture=architecture,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        global_pool=global_pool,
        batch_size=data_module.batch_size,
    )
    if log_run:
        project = "test" if smoke_test else "flowers"
        logger = WandbLogger(project=project, tags=["oxford102"])
        checkpoint_callback = ModelCheckpoint(save_top_k=3, monitor="val/loss", filepath=logger.experiment.dir)
    else:
        logger = False
        checkpoint_callback = False

    lr_logger = LearningRateLogger(logging_interval="step")
    trainer_args = {
        "gpus": -1 if gpu else None,
        "max_epochs": 4 if smoke_test else max_epochs,
        "logger": logger,
        "row_log_interval": 1,
        "checkpoint_callback": checkpoint_callback,
        "overfit_batches": 5 if smoke_test else 0,
        "callbacks": [lr_logger],
    }
    typer.echo(f"Trainer args: \n{pformat(trainer_args)}")
    trainer = Trainer(**trainer_args)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    app()
