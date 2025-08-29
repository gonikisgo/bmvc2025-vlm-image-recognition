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
    def __init__(self, root_dir, split='val', transform=None, specific_classes=None):
        """
        Args:
            root_dir (str): Path to the dataset root directory.
            split (str): Dataset split to use ('train', 'val', etc.). Defaults to 'train'.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(root=root_dir, split=split, transform=transform)
        if specific_classes:
            self._filter_specific_classes(specific_classes)

        self.image_names = [os.path.basename(path) for path, _ in self.samples]


    def _filter_specific_classes(self, specific_classes):
        self.samples = [sample for sample in self.samples if sample[1] in specific_classes]

    def get_targets(self):
        return [sample[1] for sample in self.samples]

    def __getitem__(self, idx):
        # image, label = super().__getitem__(idx) <- This is the original inheritance
        path, label = self.samples[idx]
        img_id = self.image_names[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return img_id, image, label
