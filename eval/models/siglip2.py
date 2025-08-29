import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributed import all_gather_object
from open_clip import create_model_from_pretrained, get_tokenizer

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier
from eval.utils import save_embeddings_to_npy


class SigLIP2(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.preprocess = create_model_from_pretrained(self.model2transformers[self.model_name])
        self.tokenizer = get_tokenizer(self.model2transformers[self.model_name])
        self.model.to('cuda').eval()

        self.image_transform = self.image_transform_f
        self.tokenized_labels = self.__generate_tokenized_labels()
        self.text_embeddings = self._generate_text_embeddings()

    def __generate_tokenized_labels(self):
        n_promts = len(self.openai_templates)
        label_variations = self._generate_label_variations()

        if self.context == 'all_templates':
            # For all_templates, each class is tokenized separately (8000 total)
            tokenized_labels = [
                self.tokenizer([label_variations[i]], context_length=self.model.context_length).to('cuda')
                for i in range(len(label_variations))
            ]
        else:
            # Original behavior: each class gets all templates
            tokenized_labels = [
                self.tokenizer(label_variations[i * n_promts:(i + 1) * n_promts], context_length=self.model.context_length).to('cuda')
                for i in range(1000)
            ]
        return tokenized_labels

    def _generate_text_embeddings(self):
        text_embeddings = []
        with (torch.no_grad(), torch.amp.autocast('cuda')):
            if self.context == 'all_templates':
                # For all_templates, each class gets one template
                for text_variation in self.tokenized_labels:
                    text_features = self.model.encode_text(text_variation, normalize=True)
                    # Each text_variation is already a single template, so no averaging needed
                    text_features = text_features.squeeze(0)  # Remove batch dimension
                    text_features = text_features / text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            else:
                # Original behavior: average templates
                for text_variation in self.tokenized_labels:
                    text_features = self.model.encode_text(text_variation, normalize=True)
                    text_features = text_features.mean(dim=0)
                    text_features /= text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            
        return embeddings_tensor

    def image_transform_f(self, images):
        return self.preprocess(images)

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def on_test_start(self):
        self.text_embeddings = self.text_embeddings.to(self.device)

    def predict_step(self, images, labels):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images, normalize=True)

            logits = image_features @ self.text_embeddings.T
            logits = logits * self.model.logit_scale.exp() + self.model.logit_bias
            text_probs = torch.sigmoid(logits)

            probs, preds = torch.topk(text_probs, k=self.topk, dim=-1)

        torch.cuda.empty_cache()
        return preds, probs


class SigLip2Embedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.preprocess = create_model_from_pretrained(self.model2transformers[self.model_name])
        self.model.to('cuda').eval()
        self.image_transform = self.image_transform_f

    def image_transform_f(self, images):
        return self.preprocess(images)

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images, normalize=False).to(self.device)
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
            convert_labels_to_int=True,
            project_root=project_root
        )
