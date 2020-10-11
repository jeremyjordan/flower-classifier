"""
You can run this script by calling `flower_classifier` in your terminal window.
"""

import hydra
from pytorch_lightning import Trainer

from flower_classifier.models.classifier import FlowerClassifier


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):
    data_module = hydra.utils.instantiate(cfg.dataset)
    model = FlowerClassifier(**cfg.model)

    logger = hydra.utils.instantiate(cfg.trainer.logger) or False
    checkpoint_callback = hydra.utils.instantiate(cfg.trainer.checkpoint_callback) or False

    trainer_args = {**cfg.trainer, "logger": logger, "checkpoint_callback": checkpoint_callback}
    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
