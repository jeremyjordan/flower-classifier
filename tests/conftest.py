import os

import pytest
import torch
import torchvision

from flower_classifier.datasets.csv import CSVDataset
from flower_classifier.datasets.oxford_flowers import OxfordFlowers102Dataset, OxfordFlowersDataModule, split_dataset
from flower_classifier.datasets.random import RandomDataModule
from tests.datasets import TEST_CACHE_DIR


@pytest.fixture(scope="module")
def oxford_dataset() -> torch.utils.data.Dataset:
    transforms = [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    dataset = OxfordFlowers102Dataset(root_dir=TEST_CACHE_DIR, download=True, transforms=transforms)
    return dataset


@pytest.fixture(scope="module")
def oxford_dataloader(oxford_dataset):
    dataloader = torch.utils.data.DataLoader(oxford_dataset, batch_size=8, shuffle=False)
    return dataloader


@pytest.fixture(scope="module")
def oxford_datamodule():
    transforms = [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    data_module = OxfordFlowersDataModule(data_dir=TEST_CACHE_DIR, batch_size=32, transforms=transforms)
    return data_module


@pytest.fixture(scope="module")
def oxford_csv_dataset() -> torch.utils.data.Dataset:
    split_dataset(root_dir=TEST_CACHE_DIR, target_dir=TEST_CACHE_DIR)
    train_filename = os.path.join(TEST_CACHE_DIR, "train_split.csv")
    transforms = [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    dataset = CSVDataset(filename=train_filename, transforms=transforms)
    return dataset


@pytest.fixture(scope="module")
def oxford_csv_dataloader(oxford_csv_dataset):
    dataloader = torch.utils.data.DataLoader(oxford_csv_dataset, batch_size=8, shuffle=False)
    return dataloader


@pytest.fixture(scope="module")
def random_datamodule():
    data_module = RandomDataModule(batch_size=32)
    return data_module
