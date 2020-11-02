import pandas as pd
import torch
import torchvision
from PIL import Image

from flower_classifier.datasets.oxford_flowers import NAMES as oxford_idx_to_names


class CSVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename: str,
        data_col: str = "filename",
        target_col: str = "label",
        transforms=[],
        class_names=oxford_idx_to_names,
    ):
        self.df = pd.read_csv(filename)
        self.data_col = data_col
        self.target_col = target_col
        self.class_names = class_names
        self.transform = torchvision.transforms.Compose(transforms)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.data_col])
        img = self.transform(img)
        label = row[self.target_col]
        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return self.df.shape[0]

    @property
    def classes(self):
        return self.class_names
