import sys
import os

import numpy as np
import torch
from torch.distributed import all_gather_object
from open_clip import create_model_from_pretrained, get_tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from eval.models.pl_model import HFTransformersClassifier


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
        print(f"Generating tokenized labels for {n_promts} promts")
        label_variations = self._generate_label_variations()
        print(f"Label variations: {label_variations[:5]}")

        tokenized_labels = [
            self.tokenizer(label_variations[i * n_promts:(i + 1) * n_promts], context_length=self.model.context_length).to('cuda')
            for i in range(1000)
        ]
        return tokenized_labels

    def _generate_text_embeddings(self):
        text_embeddings = []
        with (torch.no_grad(), torch.amp.autocast('cuda')):
            for text_variation in self.tokenized_labels:
                text_features = self.model.encode_text(text_variation, normalize=True)
                text_features = text_features.mean(dim=0)
                text_features /= text_features.norm()
                text_embeddings.append(text_features)
        return torch.stack(text_embeddings)

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
            suppl = self.create_label_confidence(labels, text_probs)

        torch.cuda.empty_cache()
        return preds, probs, suppl


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
        self.__save_to_npy(gathered_results)

    def __save_to_npy(self, results):
        flattened_results = [item for sublist in results for item in sublist]
        labels, image_names, embeddings = [], [], []

        for output in flattened_results:
            batch_labels = output[0].cpu().numpy() if torch.is_tensor(output[0]) else output[0]
            batch_image_names = output[1]
            batch_embeddings = output[2].cpu().numpy() if torch.is_tensor(output[2]) else output[2]

            labels.extend([int(label) for label in batch_labels])
            image_names.extend(batch_image_names)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        data = {'label': labels, 'image_name': image_names, 'embedding': embeddings}

        np.save(f'{self.exp_name}.npy', data)
        print(f'Data saved to NPY {self.exp_name}.npy file.')
