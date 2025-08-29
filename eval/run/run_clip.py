import sys
from pathlib import Path

import numpy as np
import torch

import time
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))
from eval.models.openclip import OpenCLIP, OpenClipEmbedder
from eval.models.clip import CLIP, ClipEmbedder
from eval.models.siglip import SigLIP, SigLipEmbedder
from eval.models.siglip2 import SigLIP2, SigLip2Embedder
from eval.models.dino import DINO
from data.datamodules import ImageNetDataModule

"""
Script for testing classifier and embedder models using PyTorch Lightning and Hydra.
Supports both classification and embedding modes.
"""
@hydra.main(version_base=None, config_path='../../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()

    trainer = Trainer()
    
    # Get mode from config (default to classifier if not specified)
    mode = getattr(cfg, 'mode', 'classifier')
    model_name = cfg.model
    
    # Dynamically select model based on configuration and mode
    if mode == 'embedder':
        # Embedder mode - use embedder models
        if model_name == 'SigLIP':
            model = SigLipEmbedder(cfg=cfg)
        elif model_name == 'SigLIP2':
            model = SigLip2Embedder(cfg=cfg)
        elif model_name == 'CLIP':
            model = ClipEmbedder(cfg=cfg)
        elif model_name == 'OpenCLIP':
            model = OpenClipEmbedder(cfg=cfg)
        elif model_name == 'DINOv2':
            model = DINO(cfg=cfg)
        else:
            raise ValueError(f"Unsupported model for embedder mode: {model_name}. Supported models: SigLIP, SigLIP2, CLIP, OpenCLIP, DINO")
    else:
        # Classifier mode - use classifier models
        if model_name == 'SigLIP':
            model = SigLIP(cfg=cfg)
        elif model_name == 'SigLIP2':
            model = SigLIP2(cfg=cfg)
        elif model_name == 'CLIP':
            model = CLIP(cfg=cfg)
        elif model_name == 'OpenCLIP':
            model = OpenCLIP(cfg=cfg)
        else:
            raise ValueError(f"Unsupported model for classifier mode: {model_name}. Supported models: SigLIP, SigLIP2, CLIP, OpenCLIP")

    val_transform = model.get_image_transform(is_training=False)
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform)
    imagenet_data.setup(stage='test')

    if cfg.test.dataloader == 'val':
        trainer.test(model, dataloaders=imagenet_data.val_dataloader())
    else:
        print('Using train dataloader for testing.')
        trainer.test(model, dataloaders=imagenet_data.train_dataloader())

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")
    print(f"Mode: {mode}, Model: {model_name}")

if __name__ == "__main__":
    main()
