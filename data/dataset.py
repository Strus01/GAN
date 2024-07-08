import os

import torch
from torch.utils.data import Dataset
from PIL import Image


class AbstractGalleryDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len([image for image in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, image))])

    def __getitem__(self, index: int) -> Image.Image | torch.Tensor:
        image_path = os.path.join(self.root_dir, os.listdir(self.root_dir)[index])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
