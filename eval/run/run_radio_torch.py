import sys
from pathlib import Path
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.radio_embedder_torch import RADIOTorchEmbedder
from data.datamodules import ImageNetDataModule


@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()

    model = RADIOTorchEmbedder(cfg)

    val_transform = model.get_image_transform()
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform)
    imagenet_data.setup(stage='test')

    if cfg.test.dataloader == 'val':
        print('Using validation dataloader for testing.')
        dataloader = imagenet_data.val_dataloader()
    else:
        print('Using training dataloader for testing.')
        dataloader = imagenet_data.train_dataloader()

    results = model.run_test(dataloader)
    model.save_results(results)

    elapsed_time = time.time() - start_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()