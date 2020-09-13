import pytorch_lightning as pl
import torch
import torch.nn as nn


class FlowerClassifier(pl.LightningModule):
    def __init__(self, network):
        super().__init__()
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
        result = pl.EvalResult()
        result.log("val/loss", loss)
        result.log("val/acc", acc)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger

    from flower_classifier.datasets.oxford_flowers import (
        OxfordFlowersDataModule,
    )
    from flower_classifier.models.baseline import BaselineResnet

    network = BaselineResnet()
    model = FlowerClassifier(network=network)
    data_module = OxfordFlowersDataModule(batch_size=32)
    logger = WandbLogger(project="test", tags=["test"])
    trainer = Trainer(logger=logger, overfit_pct=0.1, checkpoint_callback=False, row_log_interval=1)
    trainer.fit(model, datamodule=data_module)
