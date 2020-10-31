"""
You can run this script by calling `flower_classifier` in your terminal window.
"""

import os

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger

from flower_classifier.models.classifier import FlowerClassifier


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):
    transforms = [hydra.utils.instantiate(t) for t in cfg.transforms]
    data_module = hydra.utils.instantiate(cfg.dataset, transforms=transforms)
    model = FlowerClassifier(
        **cfg.model,
        optimizer_config=cfg.optimizer,
        lr_scheduler_config=cfg.lr_scheduler,
        batch_size=cfg.dataset.batch_size
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
