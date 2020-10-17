import logging

import hydra
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.metrics.functional.classification import confusion_matrix

from flower_classifier.datasets.oxford_flowers import NAMES
from flower_classifier.visualizations import generate_confusion_matrix

logger = logging.getLogger(__name__)


class FlowerClassifier(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        dropout_rate: float = 0.0,
        global_pool: str = "avg",
        learning_rate: float = 1e-3,
        num_classes: int = 102,
        batch_size: int = 64,
        optimizer_config: DictConfig = None,
        lr_scheduler_config: DictConfig = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        # sanity check values
        pool_options = {"avg", "max", "avgmax", "avgmaxc"}
        model_options = timm.list_models(pretrained=True)
        assert global_pool in pool_options, f"global_pool must be one of: {pool_options}"
        assert architecture in model_options, "model architecture not recognized"

        # define training objects
        self.network = timm.create_model(
            architecture, pretrained=True, num_classes=num_classes, drop_rate=dropout_rate, global_pool=global_pool
        )
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_metric = pl.metrics.Accuracy()

    @property
    def example_input_array(self):
        return torch.zeros(1, 3, 256, 256)

    def forward(self, x):
        return self.network(x)

    def _step(self, batch):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        preds = nn.functional.softmax(logits, dim=1).argmax(1)
        acc = self.accuracy_metric(preds, labels)
        return {"loss": loss, "accuracy": acc, "preds": preds, "labels": labels}

    def training_step(self, batch, batch_idx):
        step_result = self._step(batch)
        loss = step_result["loss"]
        acc = step_result["accuracy"]
        metrics = {"train/loss": loss, "train/acc": acc, "epoch": self.current_epoch}
        if self.logger:
            self.logger.log_metrics(metrics, step=self.global_step)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        step_result = self._step(batch)
        return step_result

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        acc = torch.stack([x["accuracy"] for x in validation_step_outputs]).mean()
        metrics = {"val/loss": loss, "val/acc": acc}
        if self.logger:
            self.logger.log_metrics(metrics, step=self.global_step)
            if self.current_epoch > 0 and self.current_epoch % 5 == 0:
                epoch_preds = torch.cat([x["preds"] for x in validation_step_outputs])
                epoch_targets = torch.cat([x["labels"] for x in validation_step_outputs])
                cm = confusion_matrix(epoch_preds, epoch_targets, num_classes=self.hparams.num_classes).cpu().numpy()
                fig = generate_confusion_matrix(cm, class_names=NAMES)  # TODO remove this hardcoding
                if self.logger:
                    self.logger.experiment.log({"confusion_matrix": fig})
        return metrics

    def configure_optimizers(self):
        if self.optimizer_config and self.lr_scheduler_config:
            optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
            scheduler = hydra.utils.instantiate(self.lr_scheduler_config.scheduler, optimizer=optimizer)
            scheduler_dict = OmegaConf.to_container(self.lr_scheduler_config, resolve=True)
            scheduler_dict["scheduler"] = scheduler
            return [optimizer], [scheduler_dict]
        else:
            logger.info("Hydra configuration not set, using default optimizer.")
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            return optimizer
