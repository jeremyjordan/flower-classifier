import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn

from flower_classifier.datasets.oxford_flowers import NAMES
from flower_classifier.visualizations import generate_confusion_matrix


class FlowerClassifier(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        dropout_rate: float = 0.0,
        global_pool: str = "avg",
        learning_rate: float = 1e-3,
        num_classes: int = 102,
        batch_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

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
        self.cm_metric = pl.metrics.ConfusionMatrix()

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
        metrics = {"train/loss": loss, "train/acc": acc}
        self.logger.log_metrics(metrics, step=self.global_step)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        step_result = self._step(batch)
        return step_result

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        acc = torch.stack([x["accuracy"] for x in validation_step_outputs]).mean()
        metrics = {"val/loss": loss, "val/acc": acc}
        self.logger.log_metrics(metrics, step=self.global_step)
        if self.current_epoch > 0 and self.current_epoch % 5 == 0:
            epoch_preds = torch.cat([x["preds"] for x in validation_step_outputs])
            epoch_targets = torch.cat([x["labels"] for x in validation_step_outputs])
            confusion_matrix = self.cm_metric(epoch_preds, epoch_targets).cpu().numpy()
            fig = generate_confusion_matrix(confusion_matrix, class_names=NAMES)  # TODO remove this hardcoding
            if self.logger:
                self.logger.experiment.log({"confusion_matrix": fig})
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=116, epochs=5
        )
        return [optimizer], [scheduler]
