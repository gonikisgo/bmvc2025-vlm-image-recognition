import os
import sys
import time
from omegaconf import DictConfig
import hydra

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..', '..')))
from eval.models.radio_embedder_torch import RADIOTorchEmbedder
from data.datamodules import ImageNetDataModule


@hydra.main(version_base=None, config_path='../../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    specific_classes = cfg.data.cls_list if 'cls_list' in cfg.data else None

    model = RADIOTorchEmbedder(cfg)

    val_transform = model.get_image_transform()
    imagenet_data = ImageNetDataModule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        specific_classes=specific_classes)
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