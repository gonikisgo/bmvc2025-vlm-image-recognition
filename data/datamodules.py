import sys
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from data.datasets import ImagenetDataset


class ImageNetDataModule(LightningDataModule):
    def __init__(self, cfg, train_transform=None, val_transform=None):
        """
        ImageNet data module for training and validation.

        Args:
            cfg:
            train_transform: which transform to apply to the training images
            val_transform: which transform to apply to the validation images
        """
        super().__init__()
        self.data_dir = cfg.path.data_dir

        self.num_workers = cfg.test.num_workers
        self.batch_size = cfg.test.batch_size

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None, bound=None):
        if stage == 'fit':
            self.train_dataset = ImagenetDataset(
                root_dir=self.data_dir,
                split='train',
                transform=self.train_transform,
                seed=self.seed
            )

            self.val_dataset = ImagenetDataset(
                root_dir=self.data_dir,
                split='val',
                transform=self.val_transform
            )

        if stage == 'test':
            self.train_dataset = ImagenetDataset(
                root_dir=self.data_dir,
                split='train',
                transform=self.val_transform,
                specific_classes=list(range(0, 2))
            )

            self.val_dataset = ImagenetDataset(
                root_dir=self.data_dir,
                split='val',
                transform=self.val_transform,
                specific_classes=list(range(0, 2))
            )

        if stage == 'predict':
            self.val_dataset = ImagenetDataset(
                root_dir=self.data_dir,
                split='val',
                transform=self.val_transform
            )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True)
