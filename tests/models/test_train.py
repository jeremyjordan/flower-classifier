import math

import pytest
import pytorch_lightning as pl
from omegaconf import DictConfig

from flower_classifier.models.classifier import FlowerClassifier
from flower_classifier.train import resolve_steps_per_epoch


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


@pytest.mark.download
def test_training_step_csv_dataloader(lightning_module, trainer, oxford_csv_dataloader):
    trainer.fit(lightning_module, train_dataloader=oxford_csv_dataloader, val_dataloaders=oxford_csv_dataloader)


def test_training_step_offline(lightning_module, trainer, random_datamodule):
    trainer.fit(lightning_module, datamodule=random_datamodule)


def test_auto_steps_per_epoch(random_datamodule, batch_size: int = 32):
    cfg = DictConfig(
        {"dataset": {"batch_size": batch_size}, "lr_scheduler": {"scheduler": {"steps_per_epoch": "AUTO"}}}
    )
    random_datamodule.prepare_data()
    len_train = random_datamodule.len_train
    lr_scheduler = resolve_steps_per_epoch(cfg, len_train)
    assert lr_scheduler.scheduler.steps_per_epoch == int(math.ceil(len_train / batch_size))


def test_int_steps_per_epoch(steps_per_epoch=123):
    cfg = DictConfig({"lr_scheduler": {"scheduler": {"steps_per_epoch": steps_per_epoch}}})
    lr_scheduler = resolve_steps_per_epoch(cfg, 1)
    assert cfg.lr_scheduler == lr_scheduler


def test_no_steps_per_epoch(total_steps=20000):
    cfg = DictConfig({"lr_scheduler": {"scheduler": {"total_steps": total_steps}}})
    lr_scheduler = resolve_steps_per_epoch(cfg, 1)
    assert cfg.lr_scheduler == lr_scheduler


def test_only_auto_str(steps_per_epoch="abc"):
    cfg = DictConfig({"lr_scheduler": {"scheduler": {"steps_per_epoch": steps_per_epoch}}})
    with pytest.raises(AssertionError):
        _ = resolve_steps_per_epoch(cfg, 1)


def test_no_lr_schedule():
    cfg = DictConfig({})
    lr_scheduler = resolve_steps_per_epoch(cfg, 1)
    assert lr_scheduler is None
