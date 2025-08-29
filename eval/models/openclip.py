import sys
from pathlib import Path

import numpy as np
import torch
from open_clip import create_model_and_transforms, get_tokenizer
from torch.distributed import all_gather_object

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.models.pl_model import HFTransformersClassifier


class OpenCLIP(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.processor, self.image_transform = create_model_and_transforms(self.model2transformers[self.model_name])
        self.model = self.model.to('cuda')
        self.tokenizer = get_tokenizer(self.model2transformers[self.model_name])
        self.tokenized_labels = self.tokenizer(self._generate_label_variations()).to('cuda')
        print(len(self.tokenized_labels))
        self.text_embeddings = self._generate_text_embeddings()

    def _generate_text_embeddings(self):
        text_embeddings = []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            print(f"Generating tokenized labels for {len(self.openai_templates)} promts")
            
            if self.context == 'all_templates':
                # For all_templates, each class is tokenized separately (8000 total)
                for i in range(len(self.tokenized_labels)):
                    text_variation = self.tokenized_labels[i:i+1]  # Single class
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.squeeze(0)  # Remove batch dimension
                    text_features = text_features / text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            else:
                # Original behavior: average templates
                for i in range(0, len(self.tokenized_labels), len(self.openai_templates)):
                    text_variation = self.tokenized_labels[i:i + len(self.openai_templates)]
                    text_features = self.model.encode_text(text_variation, normalize=True).to('cuda')
                    text_features = text_features.mean(dim=0)
                    text_features /= text_features.norm()
                    text_embeddings.append(text_features)
                embeddings_tensor = torch.stack(text_embeddings)
            
        return embeddings_tensor

    def on_test_start(self):
        self.text_embeddings = self.text_embeddings.to(self.device)

    def predict_step(self, images, labels):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images).to(self.device)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ self.text_embeddings.T).softmax(dim=-1)
            probs, preds = torch.topk(text_probs, k=self.topk, dim=-1)
        torch.cuda.empty_cache()
        return preds, probs


class OpenClipEmbedder(HFTransformersClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model, self.processor, self.image_transform = create_model_and_transforms(self.model2transformers[self.model_name])

    def predict_step(self, images):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = self.model.encode_image(images).to(self.device)
        return image_features

    def test_step(self, batch, batch_idx):
        image_names, images, labels = batch
        embs = self.predict_step(images)
        return labels, image_names, embs

    def on_test_end(self):
        world_size = torch.distributed.get_world_size()
        is_distributed = world_size > 1

        gathered_results = [self.test_results]
        if is_distributed:
            gathered_results = [None for _ in range(world_size)]
            all_gather_object(gathered_results, self.test_results)

        if self.trainer.global_rank == 0:
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
