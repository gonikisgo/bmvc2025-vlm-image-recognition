import copy
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier
from eval.utils import save_embeddings_to_npy


class RADIOEmbedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = copy.deepcopy(cfg)
        self.model = AutoModel.from_pretrained(self.model2transformers[self.model_name], trust_remote_code=True)
        self.processor = CLIPImageProcessor.from_pretrained(self.model2transformers[self.model_name])
        self.image_transform = self.image_transform_f
        self.model.eval().cuda()
        print(f'Model resolution: {self.cfg.test.resolution}')

    def image_transform_f(self, images):
        resolution = self.cfg.test.resolution
        return self.processor(images=images, return_tensors="pt", do_resize=True, size={"height": resolution, "width": resolution}).pixel_values

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features, _ = self.model(images.squeeze(1)).to(self.device)
        return image_features

    def test_step(self, batch, batch_idx):
        image_names, images, labels = batch
        embs = self.predict_step(images)
        return labels, image_names, embs

    def on_test_end(self):
        gathered_results = [self.test_results]
        
        # For embedder mode, use model name without resolution
        # The radio.py model is only used for embedder mode
        save_embeddings_to_npy(
            results=gathered_results,
            model_name=self.model_name,
            dataloader_name=self.cfg.test.dataloader,
            convert_labels_to_int=True,  # RADIO converts labels to int
            project_root=project_root
        )
