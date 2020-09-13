from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet34


class BaselineResnet(nn.Module):
    def __init__(self, n_classes=102):
        super().__init__()

        resnet = resnet34(pretrained=True)
        self.backbone = nn.Sequential(OrderedDict(list(resnet.named_children())[:-1]))
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = resnet.fc.in_features
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
