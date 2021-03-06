import pytorch_lightning as pl
import torch
import torchvision


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_examples=64, transforms=[]):
        self.n_examples = n_examples
        self.data = torch.rand(n_examples, 3, 48, 48)
        self.labels = torch.randint(0, 2, (n_examples,))
        self.transform = torchvision.transforms.Compose(transforms)

    def __getitem__(self, idx):
        x = self.data[idx]
        x = self.transform(x)
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return self.n_examples

    @property
    def classes(self):
        return ["0", "1"]


class RandomDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, train_transforms=[], val_transforms=[]):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = RandomDataset(n_examples=64, transforms=train_transforms)
        self.val_dataset = RandomDataset(n_examples=32, transforms=val_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    @property
    def len_train(self):
        return len(self.train_dataset)

    @property
    def len_valid(self):
        return len(self.val_dataset)
