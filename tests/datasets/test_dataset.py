import pytest

N_IMAGES = 8189


@pytest.mark.download
def test_expected_dataset_len(oxford_dataset):
    assert len(oxford_dataset) == N_IMAGES
