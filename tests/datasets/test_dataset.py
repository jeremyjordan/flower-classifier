import pytest
import torch

from flower_classifier.datasets.oxford_flowers import OxfordFlowers102Dataset
from tests.datasets import USER_CACHE_DIR

N_IMAGES = 8189


@pytest.fixture(scope="module")
def oxford_dataset() -> torch.utils.data.Dataset:
    dataset = OxfordFlowers102Dataset(root_dir=USER_CACHE_DIR, download=True)
    return dataset


def test_expected_dataset_len(oxford_dataset):
    assert len(oxford_dataset) == N_IMAGES
