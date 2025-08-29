import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.dino import DINO
from data.datamodules import ImageNetDataModule

from pytorch_lightning import Trainer
import hydra
import time
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()

    trainer = Trainer()
    model = DINO(cfg)

    val_transform = model.get_image_transform()
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform)
    imagenet_data.setup(stage='test')

    if cfg.test.dataloader == 'val':
        print('Using val dataloader for testing.')
        trainer.test(model, dataloaders=imagenet_data.val_dataloader())
    else:
        print('Using train dataloader for testing.')
        trainer.test(model, dataloaders=imagenet_data.train_dataloader())

    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
