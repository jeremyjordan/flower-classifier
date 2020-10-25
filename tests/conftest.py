import pytest
import torch
import torchvision

from flower_classifier.datasets.oxford_flowers import OxfordFlowers102Dataset, OxfordFlowersDataModule
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
def oxford_datamodule(oxford_dataset):
    data_module = OxfordFlowersDataModule(data_dir=TEST_CACHE_DIR, batch_size=32)
    return data_module


@pytest.fixture(scope="module")
def random_datamodule(oxford_dataset):
    data_module = RandomDataModule(batch_size=32)
    return data_module
