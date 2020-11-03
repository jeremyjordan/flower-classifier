import pytorch_lightning as pl
import torch


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples=64):
        self.n_examples = n_examples
        self.data = torch.rand(n_examples, 3, 48, 48)
        self.labels = torch.randint(0, 2, (n_examples,))

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return self.n_examples

    @property
    def classes(self):
        return ["0", "1"]


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = RandomDataset(n_examples=64)
        self.val_dataset = RandomDataset(n_examples=32)
        self.len_train = len(self.train_dataset)
        self.len_valid = len(self.val_dataset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
