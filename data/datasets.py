import os

import numpy as np
from PIL import Image
from torchvision.datasets import ImageNet
import torch
import random
from pytorch_lightning import seed_everything


"""
Custom dataset classes for loading ImageNet.
"""
class ImagenetDataset(ImageNet):
    def __init__(self, root_dir, split='train', transform=None, seed=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            split (str): Dataset split to use ('train', 'val', etc.). Defaults to 'train'.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(root=root_dir, split=split, transform=transform)

        self.seed = seed
        self.epoch = 0
        seed_everything(self.seed)

        self.image_names = [os.path.basename(path) for path, _ in self.samples]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_targets(self):
        return [sample[1] for sample in self.samples]

    def __getitem__(self, idx):
        # image, label = super().__getitem__(idx) <- This is the original inheritance
        path, label = self.samples[idx]
        img_id = self.image_names[idx]
        image = Image.open(path).convert('RGB')

        if self.seed is not None:
            np.random.seed(self.seed + self.epoch + idx)
            random.seed(int(self.seed + self.epoch + idx))
            torch.manual_seed(self.seed + self.epoch + idx)

        if self.transform:
            image = self.transform(image)
        return img_id, image, label
