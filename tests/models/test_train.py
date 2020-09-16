import pytest
import pytorch_lightning as pl
import torch

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


@pytest.fixture(scope="module")
def random_dataset() -> torch.utils.data.Dataset:
    from ..datasets import RandomDataset

    dataset = RandomDataset()
    return dataset


@pytest.fixture(scope="module")
def random_dataloader(random_dataset):
    dataloader = torch.utils.data.DataLoader(random_dataset, batch_size=8, shuffle=False)
    return dataloader


def test_training_step(lightning_module, trainer, random_dataloader):
    trainer.fit(lightning_module, train_dataloader=random_dataloader, val_dataloaders=random_dataloader)
