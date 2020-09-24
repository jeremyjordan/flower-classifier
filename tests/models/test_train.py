import pytest
import pytorch_lightning as pl

from flower_classifier.models.baseline import BaselineResnet
from flower_classifier.models.train import FlowerClassifier


@pytest.fixture(scope="module")
def lightning_module() -> pl.LightningModule:
    network = BaselineResnet()
    model = FlowerClassifier(network, learning_rate=0.01)
    return model


@pytest.fixture(scope="module")
def trainer() -> pl.Trainer:
    trainer = pl.Trainer(fast_dev_run=True)
    return trainer


@pytest.mark.download
def test_training_step(lightning_module, trainer, oxford_dataset):
    trainer.fit(lightning_module, train_dataloader=oxford_dataset, val_dataloaders=oxford_dataset)
