import os
import sys
import time
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from eval.models.openclip import OpenCLIP, OpenClipEmbedder
from eval.models.clip import CLIP, ClipEmbedder
from eval.models.siglip import SigLIP, SigLipEmbedder
from eval.models.siglip2 import SigLIP2, SigLip2Embedder
from data.datamodules import ImageNetDataModule

"""
Script for testing classifier models using PyTorch Lightning and Hydra.
"""
@hydra.main(version_base=None, config_path='../../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    specific_classes = cfg.data.cls_list if 'cls_list' in cfg.data else None

    trainer = Trainer()
    model = SigLIP2(cfg=cfg)

    val_transform = model.get_image_transform(is_training=False)
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        specific_classes=specific_classes)
    imagenet_data.setup(stage='test', bound=[0, 1000])

    if cfg.test.dataloader == 'val':
        trainer.test(model, dataloaders=imagenet_data.val_dataloader())
    else:
        print('Using train dataloader for testing.')
        trainer.test(model, dataloaders=imagenet_data.train_dataloader())

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()
