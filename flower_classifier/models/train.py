import pytorch_lightning as pl
import torch
import torch.nn as nn


class FlowerClassifier(pl.LightningModule):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def _step(self, batch):
        imgs, labels = batch
        outputs = self(imgs)
        loss = self.criterion(outputs, labels)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    from flower_classifier.datasets.oxford_flowers import (
        OxfordFlowersDataModule,
    )
    from flower_classifier.models.baseline import BaselineResnet

    trainer = Trainer()
    network = BaselineResnet()
    model = FlowerClassifier(network=network)
    data_module = OxfordFlowersDataModule()
    trainer.fit(model, data_module)
