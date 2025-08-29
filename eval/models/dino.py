import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier
from eval.utils import save_embeddings_to_npy


class DINO(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = AutoModel.from_pretrained(self.model2transformers[cfg.test.model]).to('cuda')
        self.processor = AutoImageProcessor.from_pretrained(self.model2transformers[cfg.test.model])
        self.image_transform = self.image_transform_f

    def image_transform_f(self, images):
        return self.processor(images=images, return_tensors="pt", padding=True)

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            outputs = self.model(images['pixel_values'].squeeze(1).to(self.device))
        embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        torch.cuda.empty_cache()
        return embs

    def test_step(self, batch, batch_idx):
        image_names, images, labels = batch
        embs = self.predict_step(images)
        return labels, image_names, embs

    def on_test_end(self):
        gathered_results = [self.test_results]
        save_embeddings_to_npy(
            results=gathered_results,
            model_name=self.model_name,
            dataloader_name=self.cfg.test.dataloader,
            convert_labels_to_int=False,  # DINO doesn't convert labels to int
            project_root=project_root
        )
