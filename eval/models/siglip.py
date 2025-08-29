import sys
from pathlib import Path

import numpy as np
import torch
from transformers import SiglipProcessor, SiglipModel, AutoModel, AutoProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier
from eval.utils import save_embeddings_to_npy


class SigLIP(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = AutoModel.from_pretrained(self.model2transformers[self.model_name]).to('cuda')
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model2transformers[self.model_name])

        self.image_transform = self.image_transform_f
        self.tokenized_labels = self.__generate_tokenized_labels()
        self.text_embeddings = self._generate_text_embeddings()

    def __generate_tokenized_labels(self):
        n_promts = len(self.openai_templates)
        label_variations = self._generate_label_variations()

        tokenized_labels = [
            self.processor(text=label_variations[i * n_promts:(i + 1) * n_promts], return_tensors="pt", padding="max_length", max_length=64).to('cuda')
            for i in range(1000)
        ]
        return tokenized_labels

    def _generate_text_embeddings(self):
        text_embeddings = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for text_variation in self.tokenized_labels:
                text_features = self.model.get_text_features(**text_variation)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.mean(dim=0)
                text_features /= text_features.norm()
                text_embeddings.append(text_features)
        return torch.stack(text_embeddings)

    def image_transform_f(self, images):
        return self.processor(images=images, return_tensors="pt", padding="max_length")

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def on_test_start(self):
        self.text_embeddings = self.text_embeddings.to(self.device)

    def predict_step(self, images, labels):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.get_image_features(images['pixel_values'].squeeze(1)).to(self.device)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ self.text_embeddings.T).softmax(dim=-1)
            probs, preds = torch.topk(text_probs, k=self.topk, dim=-1)
        torch.cuda.empty_cache()
        return preds, probs


class SigLipEmbedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = AutoModel.from_pretrained(self.model2transformers[self.model_name]).to('cuda')
        self.processor = AutoProcessor.from_pretrained(self.model2transformers[self.model_name])
        self.image_transform = self.image_transform_f
        print(f"Model name: {self.model_name}")

    def image_transform_f(self, images):
        return self.processor(images=images, return_tensors="pt", padding="max_length")

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.get_image_features(images['pixel_values'].squeeze(1)).to(self.device)
        return image_features

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
            convert_labels_to_int=False,  # SigLIP doesn't convert labels to int
            project_root=project_root
        )
