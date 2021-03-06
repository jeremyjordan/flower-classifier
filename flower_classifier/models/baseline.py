from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import resnet34


class BaselineResnet(nn.Module):
    def __init__(self, n_classes=102, pretrained=True):
        super().__init__()
        assert n_classes > 0, "must have at least one class"

        resnet = resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(OrderedDict(list(resnet.named_children())[:-1]))
        in_features = resnet.fc.in_features
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
