import os
import sys

from pytorch_lightning import Trainer
import hydra
import time
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from eval.models.pl_model import TimmClassifier
from data.datamodules import ImageNetDataModule

"""
Script for testing classifier models using PyTorch Lightning and Hydra.
"""
@hydra.main(version_base=None, config_path='../../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    specific_classes = cfg.data.cls_list if 'cls_list' in cfg.data else None

    trainer = Trainer()
    if cfg.test.checkpoint_name != 'none':
        checkpoint_path = f'../../checkpoints/best_overall/{cfg.test.checkpoint_name}'
        model = TimmClassifier.load_from_checkpoint(checkpoint_path, cfg=cfg)
    else:
        model = TimmClassifier(cfg=cfg)

    val_transform = model.get_image_transform(is_training=False)
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        specific_classes=specific_classes)
    imagenet_data.setup(stage='test')

    '''imagenet_data = DataModuleKFold(cfg, val_transform, val_transform)
    imagenet_data.setup(stage='fit')'''

    if cfg.test.dataloader == 'val':
        trainer.test(model, dataloaders=imagenet_data.val_dataloader())
    else:
        trainer.test(model, dataloaders=imagenet_data.train_dataloader())

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
