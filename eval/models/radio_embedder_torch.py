import copy
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from transformers import CLIPImageProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))


class RADIOTorchEmbedder:
    model2torch = {
        'RADIO-G': 'radio_v2.5-g'
    }

    model2transformers = {
        'RADIO-G': 'nvidia/C-RADIOv2-G'
    }

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model()
        self.model.to(self.device).eval()

        self.processor = CLIPImageProcessor.from_pretrained(self.model2transformers[self.cfg.test.model])
        self.image_transform = self.image_transform_f

    def image_transform_f(self, images):
        resolution = self.cfg.test.resolution
        return self.processor(images=images, return_tensors="pt", do_resize=True, size={"height": resolution, "width": resolution}).pixel_values

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def load_model(self):
        print(f'Loading model from version {self.model2torch[self.cfg.test.model]}')
        return torch.hub.load(
            "NVlabs/RADIO",
            'radio_model',
            version=self.model2torch[self.cfg.test.model],
            adaptor_names=self.cfg.test.adaptor,
            return_spatial_features=False,
            force_reload=True
        )

    def predict_step(self, images):
        if images.dim() == 5 and images.size(1) == 1:
            images = images.squeeze(1)  # From [B, 1, C, H, W] to [B, C, H, W]

        with torch.inference_mode(), torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            output = self.model(images)
            embeddings = output[self.cfg.test.adaptor].summary
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def save_results(self, gathered_results):
        labels, image_names, embeddings = [], [], []
        for batch_labels, batch_names, batch_embeddings in gathered_results:
            labels.extend([int(label) for label in batch_labels.cpu().numpy()])
            image_names.extend(batch_names)
            embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

        data = {'label': labels, 'image_name': image_names, 'embedding': embeddings}

        print(len(data['label']), len(data['image_name']), data['embedding'].shape)
        save_dir = os.path.join(self.cfg.test.folder, self.cfg.test.model)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{self.cfg.test.exp_name}.npy')
        np.save(save_path, data)
        print(f'Data saved to NPY {save_path} file.')

    def run_test(self, dataloader):
        results = []
        for batch in tqdm(dataloader, desc="Testing"):
            image_names, images, labels = batch
            embeddings = self.predict_step(images.to(self.device))
            results.append((labels, image_names, embeddings))
        return results
