import pytest
import torch

from flower_classifier.datasets.oxford_flowers import (
    OxfordFlowers102Dataset,
    OxfordFlowersDataModule,
)
from tests.datasets import TEST_CACHE_DIR


@pytest.fixture(scope="module")
def oxford_dataset() -> torch.utils.data.Dataset:
    dataset = OxfordFlowers102Dataset(root_dir=TEST_CACHE_DIR, download=True)
    return dataset


@pytest.fixture(scope="module")
def oxford_dataloader(oxford_dataset):
    dataloader = torch.utils.data.DataLoader(oxford_dataset, batch_size=8, shuffle=False)
    return dataloader


@pytest.fixture(scope="module")
def oxford_datamodule(oxford_dataset):
    data_module = OxfordFlowersDataModule(data_dir=TEST_CACHE_DIR, batch_size=32)
    return data_module
