import pytest
import torch

from flower_classifier.models.baseline import BaselineResnet

N_CLASSES = 102


@pytest.fixture(scope="module")
def network():
    network = BaselineResnet(n_classes=N_CLASSES)
    return network


@pytest.mark.parametrize(
    "batch_size, img_height, img_width",
    [(1, 256, 256), (8, 100, 300), (1, 33, 33)],
)
def test_expected_input_shape(network: torch.nn.Module, batch_size: int, img_height: int, img_width: int):
    example_input_array = torch.zeros(batch_size, 3, img_height, img_width)
    _ = network(example_input_array)


@pytest.mark.parametrize("img_height, img_width", [(4, 4), (32, 32)])
def test_input_too_small(network: torch.nn.Module, img_height: int, img_width: int):
    example_input_array = torch.zeros(1, 3, img_height, img_width)
    with pytest.raises(ValueError):
        _ = network(example_input_array)


@pytest.mark.parametrize(
    "img_height, img_width",
    [(256, 256), (32, 48)],
)
def test_input_no_batch_dim(network: torch.nn.Module, img_height: int, img_width: int):
    example_input_array = torch.zeros(3, img_height, img_width)
    with pytest.raises(RuntimeError):
        _ = network(example_input_array)


@pytest.mark.parametrize("batch_size, img_height, img_width", [(1, 256, 256), (8, 100, 300), (1, 33, 33)])
def test_expected_output_shape(network: torch.nn.Module, batch_size: int, img_height: int, img_width: int):
    example_input_array = torch.zeros(batch_size, 3, img_height, img_width)
    outputs = network(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == N_CLASSES
