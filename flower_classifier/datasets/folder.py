import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision


class FolderDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root: str, transform=[], target_transform=[]):
        transform = torchvision.transforms.Compose(transform)
        target_transform = torchvision.transforms.Compose(target_transform)
        super().__init__(root=root, transform=transform, target_transform=target_transform)

    @property
    def classes(self):
        return self.classes

    def to_csv(self, filename):
        df = pd.DataFrame(self.samples, columns=["filename", "label"])
        df["class"] = df["label"].apply(lambda x: self.classes[x])
        df.to_csv(filename, index=False)


class FolderDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, train_transforms=[], val_transforms=[], batch_size=16):

        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = FolderDataset(root=train_dir, transform=train_transforms)
        self.val_dataset = FolderDataset(root=val_dir, transform=train_transforms)

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
