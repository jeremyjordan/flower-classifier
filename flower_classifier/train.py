"""
You can run this script by calling `flower_classifier` in your terminal window.
"""

import math
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger

from flower_classifier.models.classifier import FlowerClassifier


def resolve_steps_per_epoch(cfg: DictConfig, len_train: int):
    lr_scheduler = getattr(cfg, "lr_scheduler", None)
    scheduler = getattr(lr_scheduler, "scheduler", None)
    steps_per_epoch = getattr(scheduler, "steps_per_epoch", None)
    if steps_per_epoch == "AUTO":
        lr_scheduler.scheduler.steps_per_epoch = int(math.ceil(len_train / cfg.dataset.batch_size))
    elif steps_per_epoch:
        assert isinstance(steps_per_epoch, int), "cfg.lr_scheduler.scheduler.steps_per_epoch must be AUTO or type int"
    return lr_scheduler


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):
    datamodule_args = {}
    if cfg.transforms.train:
        train_transforms = [hydra.utils.instantiate(t) for t in cfg.transforms.train]
        datamodule_args["train_transforms"] = train_transforms
    if cfg.transforms.val:
        val_transforms = [hydra.utils.instantiate(t) for t in cfg.transforms.val]
        datamodule_args["val_transforms"] = val_transforms
    data_module = hydra.utils.instantiate(cfg.dataset, **datamodule_args)
    data_module.prepare_data()
    lr_scheduler = resolve_steps_per_epoch(cfg, len_train=data_module.len_train)

    model = FlowerClassifier(
        **cfg.model,
        optimizer_config=cfg.optimizer,
        lr_scheduler_config=lr_scheduler,
        batch_size=cfg.dataset.batch_size,
    )

    logger = hydra.utils.instantiate(cfg.trainer.logger) or False
    experiment = getattr(logger, "experiment", None)
    logger_dir = getattr(experiment, "dir", "logger")
    checkpoints_dir = os.path.join(logger_dir, "{epoch}")
    checkpoint_callback = hydra.utils.instantiate(cfg.trainer.checkpoint_callback, filepath=checkpoints_dir) or False

    lr_logger = LearningRateLogger(logging_interval="step")
    trainer_args = {
        **cfg.trainer,
        "logger": logger,
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [lr_logger],
    }
    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
