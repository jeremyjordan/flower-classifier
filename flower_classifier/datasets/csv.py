"""
You can generate a CSV dataset by running the `split_dataset()` utility function
in `flower_classifier/datasets/oxford_flowers.py`. This splits the original dataset
into a train and validation split. With separate datasets, you can apply different
transformation pipelines.
"""

import pandas as pd
import pytorch_lightning as pl
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


class CSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str = "/content/drive/My Drive/Flowers/oxford102/train_split.csv",
        val_csv: str = "/content/drive/My Drive/Flowers/oxford102/val_split.csv",
        train_transforms=[],
        val_transforms=[],
        batch_size=16,
        data_col: str = "filename",
        target_col: str = "label",
        class_names=oxford_idx_to_names,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = CSVDataset(
            filename=train_csv,
            data_col=data_col,
            target_col=target_col,
            transforms=train_transforms,
            class_names=class_names,
        )
        self.val_dataset = CSVDataset(
            filename=val_csv,
            data_col=data_col,
            target_col=target_col,
            transforms=val_transforms,
            class_names=class_names,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    @property
    def len_train(self):
        return len(self.train_dataset)

    @property
    def len_valid(self):
        return len(self.val_dataset)
