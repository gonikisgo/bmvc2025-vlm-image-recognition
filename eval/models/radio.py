import copy
import sys
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier


class RADIOEmbedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = copy.deepcopy(cfg)
        print(self.model2transformers[self.model_name])

        hf_repo = self.model2transformers[self.model_name]
        self.model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
        self.processor = CLIPImageProcessor.from_pretrained(hf_repo)

        config = AutoConfig.from_pretrained(hf_repo, trust_remote_code=True)
        config.adaptor_names = ["clip", "sam"]
        self.model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True, config=config)

        self.image_transform = self.image_transform_f
        self.model.eval().cuda()

        print(self.cfg.test.resolution)

        print(self.model.min_resolution_step)
        print(self.model.patch_size)
        print(self.model.adaptors)

    def image_transform_f(self, images):
        resolution = self.cfg.test.resolution
        return self.processor(images=images, return_tensors="pt", do_resize=True, size={"height": resolution, "width": resolution}).pixel_values

    def get_image_transform(self, is_training=False):
        return self.image_transform

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model(images.squeeze(1))["clip"].summary
            print(image_features.shape)
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

        print(len(data['label']), len(data['image_name']), data['embedding'].shape)

        save_dir = os.path.join(self.cfg.test.model)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'{self.cfg.test.exp_name}.npy')
        np.save(save_path, data)
        print(f'Data saved to NPY {save_path} file.')
