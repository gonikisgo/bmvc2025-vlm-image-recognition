import os
import sys

from pytorch_lightning import Trainer
import hydra
import time
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from eval.models.dino import DINO
from data.datamodules import ImageNetDataModule

@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    specific_classes = cfg.data.cls_list if 'cls_list' in cfg.data else None

    trainer = Trainer()
    model = DINO(cfg)

    val_transform = model.get_image_transform()
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        specific_classes=specific_classes)
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
