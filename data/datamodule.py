from typing import Optional

import lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from omegaconf import DictConfig

from data.dataset import AbstractGalleryDataset


class AbstractGalleryDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

    def setup(self, stage: Optional[str] = None):
        dataset = AbstractGalleryDataset(self.config.data_dir, transform=self.transform)
        train_samples = int(len(dataset) * self.config.train_size)
        valid_samples = len(dataset) - train_samples
        self.train, self.valid = random_split(dataset, [train_samples, valid_samples])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.config.batch_size, num_workers=self.config.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, batch_size=self.config.batch_size, num_workers=self.config.num_workers)
