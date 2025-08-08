import sys
import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from eval.utils import EmbsLoader, save_predictions_csv


class Mult:
    text_embs = {
        'openai': {'mean7': 'openai_mean_7.npy', 'mean8': 'openai_mean_8.npy'},
        'mod': {'mean7': 'mod_mean_7.npy', 'mean8': 'mod_mean_8.npy'},
        'wordnet': {'0': 'wordnet_0.npy'},

        'templ': {'openai_0': 'openai_0.npy',
                  'mod_0': 'mod_0.npy',
                  'mod_1': 'mod_1.npy',
                  'mod_2': 'mod_2.npy',
                  'mod_3': 'mod_3.npy',
                  'mod_4': 'mod_4.npy',
                  'mod_5': 'mod_5.npy',
                  'mod_6': 'mod_6.npy',
                  'mod_7': 'mod_7.npy'
                  },

        'mapping': {'mod_0': 'mod_0.npy',
                  'mod_1': 'mod_1.npy',
                  'mod_2': 'mod_2.npy',
                  'mod_3': 'mod_3.npy',
                  'mod_4': 'mod_4.npy',
                  'mod_5': 'mod_5.npy',
                  'mod_6': 'mod_6.npy',
                  'mod_7': 'mod_7.npy'
                  }
    }

    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_embeddings = []

        self.labels_option = cfg.test.labels_option
        self.context = cfg.test.context

        self.folder = cfg.test.folder
        self.embs_path = cfg.path.text_embs
        self.text_embeddings = self.load_text_embeddings()
        print(self.text_embeddings.shape)

        embsLoader = EmbsLoader(cfg)

        self.dataloader = cfg.test.dataloader
        if self.dataloader == 'train':
            self.image_embeddings = embsLoader.get_train_embeddings()
            print(len(np.unique(self.image_embeddings['image_name'])))
            print("Loading train embeddings")
        else:
            self.image_embeddings = embsLoader.get_val_embeddings()
            print("Loading val embeddings")

        self.embs_space = cfg.test.emb_space

        self.image_embeddings['embedding'] = np.array(self.image_embeddings['embedding']).astype(np.float32)

        print(f'Number of text embeddings: {self.text_embeddings.shape}')
        print(self.image_embeddings['embedding'].shape)
        self.test_results = []
        self.topk = cfg.test.topk
        self.exp_name = f"{cfg.test.exp_name}_{self.embs_space}_{self.dataloader}_{self.labels_option}_{self.context}"

    def load_text_embeddings(self):
        path = os.path.join(self.embs_path, self.folder, self.text_embs[self.labels_option][self.context])
        print(f'Loading text embeddings from {path}')
        return torch.tensor(np.load(path), device=self.device, dtype=torch.float32)

    def load_mapping_embeddings(self):
        self.text_embeddings = []
        print(len(self.text_embs['mapping'].keys()))
        for key in self.text_embs['mapping'].keys():
            path = os.path.join(self.embs_path, self.folder, self.text_embs['mapping'][key])
            print(f'Loading text embeddings from {path}')
            emb = torch.tensor(np.load(path), device=self.device, dtype=torch.float32)
            self.text_embeddings.append(emb)
        return torch.cat(self.text_embeddings, dim=0)

    def test_step(self, batch_size=50):
        num_samples = len(self.image_embeddings['embedding'])
        for i in range(0, num_samples, batch_size):
            batch_embeddings = torch.tensor(self.image_embeddings['embedding'][i:i + batch_size], device=self.device, dtype=torch.float32)
            batch_labels = self.image_embeddings['label'][i:i + batch_size]
            batch_image_names = self.image_embeddings['image_name'][i:i + batch_size]

            text_features = self.text_embeddings
            preds, probs = self.predict_step(batch_embeddings, batch_labels, text_features)
            outputs = {
                'original_labels': batch_labels,
                'image_names': batch_image_names,
                'preds': preds,
                'probs': probs
            }
            self.on_test_batch_end(outputs)
        self.on_test_end()

    def predict_step(self, image_embeddings, labels, text_features):
        with torch.no_grad(), torch.amp.autocast(self.device.type):
            image_features = image_embeddings

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs, preds = torch.topk(text_probs, k=self.topk, dim=-1)
        torch.cuda.empty_cache()
        return preds, probs

    def on_test_batch_end(self, outputs):
        self.test_results.append(outputs)

    def on_test_end(self):
        gathered_results = [self.test_results]
        flattened_results = [item for sublist in gathered_results for item in sublist]

        data = []
        for output in flattened_results:
            labels, image_names = output['original_labels'], output['image_names']
            preds, probs = output['preds'], output['probs']

            for i in range(len(labels)):
                row = {
                    'original_label': labels[i],
                    'img_id': image_names[i]
                }
                for j in range(self.topk):
                    row[f'top_{j + 1}_pred'] = preds[i][j].item()
                    row[f'top_{j + 1}_prob'] = probs[i][j].item()
                data.append(row)

        print(f"Number of predictions: {len(data)}")
        save_predictions_csv(pd.DataFrame(data), filename=self.exp_name)
        print(f"Saved predictions to {self.exp_name}.csv")


@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()

    dataloaders = ['train']

    text_embs = {
        'openai': {'mean7': 'openai_mean_7.npy', 'mean8': 'openai_mean_8.npy'},
        'mod': {'mean7': 'mod_mean_7.npy', 'mean8': 'mod_mean_8.npy'},
        'wordnet': {'0': 'wordnet_0.npy'},
    }

    for dataloader in dataloaders:
        for labels_option in text_embs.keys():
            for context in text_embs[labels_option].keys():
                print(f"Running with dataloader: {dataloader}, labels_option: {labels_option}, context: {context}")
                cfg.test.dataloader = dataloader
                cfg.test.labels_option = labels_option
                cfg.test.context = context
                mult = Mult(cfg)
                mult.test_step()
                print(f"Finished with dataloader: {dataloader}, labels_option: {labels_option}, context: {context}")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    main()
