import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)


class FolderDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, transform=[], target_transform=[]):
        transform = torchvision.transforms.Compose(transform)
        target_transform = torchvision.transforms.Compose(target_transform)
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    def to_csv(self, filename, indices: List[int] = None):
        df = pd.DataFrame(self.samples, columns=["filename", "label"])
        df["class"] = df["label"].apply(lambda x: self.classes[x])
        if indices:
            df = df.iloc[indices]
        df.to_csv(filename, index=False)


class FolderDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, train_transforms=[], val_transforms=[], val_size=0.1, batch_size=16):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = train_transforms
        if val_transforms:
            logger.info("val_transforms not supported for this dataset, using train_transforms for all datasets")

        self.val_size = val_size
        self.batch_size = batch_size

    def prepare_data(self):
        self.dataset = FolderDataset(self.root_dir, transform=self.transforms)
        train_idx, valid_idx = self.get_sampler_indices(self.val_size)
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(valid_idx)

    def get_sampler_indices(self, val_size=0.1, shuffle=True, random_seed=14):
        num_train = len(self.dataset)
        indices = list(range(num_train))
        split = int(np.floor(val_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        return train_idx, valid_idx

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=4
        )

    @property
    def len_train(self):
        return len(self.train_sampler)

    @property
    def len_valid(self):
        return len(self.val_sampler)

    def to_csv(self, output_dir):
        train_csv = Path(output_dir, "train.csv")
        self.dataset.to_csv(train_csv, indices=self.train_sampler.indices)

        val_csv = Path(output_dir, "val.csv")
        self.dataset.to_csv(val_csv, indices=self.val_sampler.indices)
