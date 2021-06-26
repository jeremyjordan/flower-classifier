import logging

import hydra
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchmetrics.functional import confusion_matrix

from flower_classifier.visualizations import generate_confusion_matrix

logger = logging.getLogger(__name__)

# add default for cases where we don't initialize with hydra main
DEFAULT_OPTIMIZER = OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 0.001})


class FlowerClassifier(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        dropout_rate: float = 0.0,
        global_pool: str = "avg",
        num_classes: int = 102,
        batch_size: int = 64,
        optimizer_config: DictConfig = DEFAULT_OPTIMIZER,
        lr_scheduler_config: DictConfig = None,
        pretrained: bool = True,
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
            architecture,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            global_pool=global_pool,
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
        self.log("train/loss", step_result["loss"], on_step=True)
        self.log("train/acc", step_result["accuracy"], on_step=True)
        self.log("epoch", self.current_epoch, on_step=True)
        return {"loss": step_result["loss"]}

    def validation_step(self, batch, batch_idx):
        step_result = self._step(batch)
        self.log("val/loss", step_result["loss"], on_step=False, on_epoch=True, reduce_fx=torch.mean)
        self.log("val/acc", step_result["accuracy"], on_step=False, on_epoch=True, reduce_fx=torch.mean)
        return step_result

    def validation_epoch_end(self, validation_step_outputs):
        if self.logger and self.current_epoch > 0 and self.current_epoch % 5 == 0:
            epoch_preds = torch.cat([x["preds"] for x in validation_step_outputs])
            epoch_targets = torch.cat([x["labels"] for x in validation_step_outputs])
            cm = confusion_matrix(epoch_preds, epoch_targets, num_classes=self.hparams.num_classes).cpu().numpy()
            class_names = getattr(self.train_dataloader().dataset, "classes", None)
            fig = generate_confusion_matrix(cm, class_names=class_names)
            self.logger.experiment.log({"confusion_matrix": fig})

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        if self.lr_scheduler_config is None:
            return optimizer

        scheduler = hydra.utils.instantiate(self.lr_scheduler_config.scheduler, optimizer=optimizer)
        scheduler_dict = OmegaConf.to_container(self.lr_scheduler_config, resolve=True)
        scheduler_dict["scheduler"] = scheduler
        return [optimizer], [scheduler_dict]

    def on_fit_start(self):
        if self.global_rank == 0 and getattr(self.trainer.datamodule, "to_csv", False):
            experiment = getattr(self.logger, "experiment", None)
            logger_dir = getattr(experiment, "dir", "output")
            self.trainer.datamodule.to_csv(logger_dir)

    def on_train_start(self):
        dataset = self.train_dataloader().dataset
        classes = getattr(dataset, "classes", None)
        self.classes = classes

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_prediction_classes"] = self.classes

    def on_load_checkpoint(self, checkpoint):
        self.classes = checkpoint["model_prediction_classes"]
