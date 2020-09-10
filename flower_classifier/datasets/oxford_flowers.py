import logging
from pathlib import Path

import torch
import torchvision
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from flower_classifier import ROOT_DATA_DIR

logger = logging.getLogger(__name__)


NAMES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]  # from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/oxford_flowers102.py
IMAGE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"


def download_dataset(root_dir: str):
    root_dir = Path(root_dir)

    # check if files already exist
    images_exist = (root_dir / "102flowers.tgz").exists()
    labels_exist = (root_dir / "imagelabels.mat").exists()

    # download and unpack images
    if not images_exist:
        logger.info("downloading images...")
        torchvision.datasets.utils.download_and_extract_archive(IMAGE_URL, root_dir)

    # download labels
    if not labels_exist:
        logger.info("downloading labels...")
        torchvision.datasets.utils.download_url(LABELS_URL, root_dir)


class OxfordFlowers102Dataset(Dataset):
    """
    Oxford 102 Category Flower Dataset
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

    Note: images are not shuffled by default.
    """

    def __init__(
        self, root_dir: str = ROOT_DATA_DIR, download: bool = False, transforms=[]
    ):

        self.transform = torchvision.transforms.Compose(transforms)
        self.root_dir = Path(root_dir)

        if download:
            download_dataset(self.root_dir)

        labels_filename = self.root_dir / "imagelabels.mat"
        # shift labels from 1-index to 0-index
        self.labels = loadmat(labels_filename)["labels"].flatten() - 1

    def __getitem__(self, index):
        filepath = self.root_dir / "jpg" / f"image_{index+1:05}.jpg"
        img = Image.open(filepath)
        img = self.transform(img)
        label = self.labels[index]
        label = torch.LongTensor([label])
        return img, label

    def __len__(self):
        return len(self.labels)
