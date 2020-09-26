import pytorch_lightning as pl
import torch
import torch.nn as nn


class FlowerClassifier(pl.LightningModule):
    def __init__(self, network, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters("learning_rate")
        self.network = network
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
        return {"loss": loss, "accuracy": acc}

    def training_step(self, batch, batch_idx):
        step_result = self._step(batch)
        loss = step_result["loss"]
        acc = step_result["accuracy"]
        result = pl.TrainResult(minimize=loss)
        result.log("train/loss", loss)
        result.log("train/acc", acc)
        return result

    def validation_step(self, batch, batch_idx):
        step_result = self._step(batch)
        loss = step_result["loss"]
        acc = step_result["accuracy"]
        result = pl.EvalResult(checkpoint_on=loss)
        result.log("val/loss", loss)
        result.log("val/acc", acc)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
