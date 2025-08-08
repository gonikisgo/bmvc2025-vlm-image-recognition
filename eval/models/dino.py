import sys
import os

import torch
import numpy as np
from transformers import AutoModel, AutoImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from eval.models.pl_model import HFTransformersClassifier


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
        self.__save_to_npy(gathered_results)

    def __save_to_npy(self, results):
        flattened_results = [item for sublist in results for item in sublist]
        labels, image_names, embeddings = [], [], []

        for output in flattened_results:
            batch_labels = output[0].cpu().numpy() if torch.is_tensor(output[0]) else output[0]
            batch_image_names = output[1]
            batch_embeddings = output[2].cpu().numpy() if torch.is_tensor(output[2]) else output[2]

            labels.extend(batch_labels)
            image_names.extend(batch_image_names)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)
        data = {'label': labels, 'image_name': image_names, 'embedding': embeddings}

        np.save(f'{self.exp_name}.npy', data)
        print(f'Data saved to NPY {self.exp_name}.npy file.')
