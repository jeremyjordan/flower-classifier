import pytest
import pytorch_lightning as pl

from flower_classifier.models.classifier import FlowerClassifier


@pytest.fixture(scope="module")
def lightning_module() -> pl.LightningModule:
    model = FlowerClassifier(architecture="resnet34")
    return model


@pytest.fixture(scope="module")
def trainer() -> pl.Trainer:
    trainer = pl.Trainer(fast_dev_run=True, logger=False)
    return trainer


@pytest.mark.download
def test_training_step_dataloaders(lightning_module, trainer, oxford_dataloader):
    trainer.fit(lightning_module, train_dataloader=oxford_dataloader, val_dataloaders=oxford_dataloader)


@pytest.mark.download
def test_training_step_datamodule(lightning_module, trainer, oxford_datamodule):
    trainer.fit(lightning_module, datamodule=oxford_datamodule)


def test_training_step_offline(lightning_module, trainer, random_datamodule):
    trainer.fit(lightning_module, datamodule=random_datamodule)
