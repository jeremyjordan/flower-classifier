from pathlib import Path

import torch

CACHE_DIR = Path(".data_test_cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
USER_CACHE_DIR = str(CACHE_DIR)


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples=32):
        self.n_examples = n_examples
        self.data = torch.rand(n_examples, 3, 32, 32)
        self.labels = torch.randint(0, 2, (n_examples,))

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return self.n_examples
