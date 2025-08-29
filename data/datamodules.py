import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import seed_everything, LightningDataModule

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


"""
Data module for loading ImageNet images with k-fold cross validation
"""
class DataModuleKFold(LightningDataModule):
    def __init__(self, cfg, train_transform=None, val_transform=None, specific_classes=None):
        super().__init__()
        self.data_dir = cfg.path.data_dir
        self.csv_path = cfg.data.csv_path

        self.num_workers = cfg.train.num_workers
        self.batch_size = cfg.train.batch_size
        self.data_split = cfg.data.split
        self.k_folds = cfg.data.k_folds

        self.seed = cfg.train.seed
        seed_everything(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.specific_classes = specific_classes
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_dataset = None
        self.val_dataset = None
        self.kfold_split = None

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = CustomImagenet(
                img_dir=self.data_dir,
                labels_file=self.csv_path,
                split=self.data_split,
                transform=self.train_transform,
                seed=self.seed
            )
            self.val_dataset = CustomImagenet(
                img_dir=self.data_dir,
                labels_file=self.csv_path,
                split=self.data_split,
                transform=self.val_transform,
            )

            if self.k_folds > 1:
                targets = self.val_dataset.get_targets()
                print(f"Length of targets: {len(targets)}")
                skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
                self.kfold_split = list(skf.split(np.zeros(len(targets)), targets))

    def train_dataloader_fold(self, val_fold_index):
        train_indices, _ = self.kfold_split[val_fold_index]
        train_subset = Subset(self.train_dataset, train_indices)

        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader_fold(self, val_fold_index):
        _, val_indices = self.kfold_split[val_fold_index]
        val_subset = Subset(self.val_dataset, val_indices)

        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
