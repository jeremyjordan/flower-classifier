import logging

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class GradualUnfreezingCallback(Callback):
    """
    This callback will freeze the weights for the portion of the network
    with pre-trained weights. By convention, we will expect that these
    weights will be organized in a `backbone` submodule of the network.

    At the start of training, we will freeze the backbone weights on only
    train the last few linear layers of the network. After a specified
    warmup period, we'll unfreeze the backbone weights and train all weights
    jointly.
    """

    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        if not hasattr(pl_module, "network") or not hasattr(pl_module.network, "backbone"):
            raise ValueError("Cannot find backbone layers.")

        # freeze weights for backbone layers
        logger.info("Freezing backbone weights for network.")
        for param in pl_module.network.backbone.parameters():
            param.requires_grad = False

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.current_epoch == self.warmup_epochs:
            # unfreeze weights so that all layers can train
            logger.info("Unfreezing backbone weights for network.")
            for param in pl_module.network.backbone.parameters():
                param.requires_grad = True
