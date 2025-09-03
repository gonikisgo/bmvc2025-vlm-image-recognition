import os
import time
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

# Add the project root to the path using Path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from eval.utils import EmbsLoader, save_predictions_csv, eval_on_clean_labels, save_accuracy_results_csv


class Mult:
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        
        self.labels_option = cfg.test.labels_option
        self.context = cfg.test.context

        self.folder = cfg.test.folder  # Should be 'OpenCLIP' for text embeddings
        self.embs_space = cfg.test.emb_space  # e.g., 'RADIO_378'
        
        # Load text embeddings
        self.text_embeddings = self.load_text_embeddings()
        print(f"Text embeddings shape: {self.text_embeddings.shape}")

        # Load image embeddings
        embsLoader = EmbsLoader(cfg)
        self.dataloader = cfg.test.dataloader
        if self.dataloader == 'train':
            self.image_embeddings = embsLoader.get_train_embeddings()
            print(f"Loading train embeddings: {len(np.unique(self.image_embeddings['image_name']))} unique images")
        else:
            self.image_embeddings = embsLoader.get_val_embeddings()
            print("Loading val embeddings")

        self.image_embeddings['embedding'] = np.array(self.image_embeddings['embedding']).astype(np.float32)

        print(f'Image embeddings shape: {self.image_embeddings["embedding"].shape}')
        self.test_results = []
        self.topk = cfg.test.topk
        self.exp_name = f"{cfg.test.exp_name}_{self.embs_space}_{self.dataloader}_{self.labels_option}_{self.context}"

    def load_text_embeddings(self):
        """Load text embeddings from OpenCLIP text embedder output."""
        # Build path to text embeddings file
        text_embeddings_dir = Path(__file__).parent.parent.parent / 'eval' / 'results' / 'text_embeddings' / self.folder
        filename = f'{self.folder}_text_embeddings_{self.context}_{self.labels_option}.npy'
        text_embeddings_path = text_embeddings_dir / filename
        
        print(f'Loading text embeddings from {text_embeddings_path}')
        
        if not text_embeddings_path.exists():
            raise FileNotFoundError(f"Text embeddings file not found: {text_embeddings_path}")
        
        # Load the .npy file which contains a dict with text embeddings
        data = np.load(text_embeddings_path, allow_pickle=True).item()
        text_embeddings = data['text_embeddings']
        
        print(f"Loaded text embeddings with shape: {text_embeddings.shape}")
        print(f"Context: {data['context']}, Labels: {data['labels_option']}")
        
        return torch.tensor(text_embeddings, device=self.device, dtype=torch.float32)

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
        """Process results, save predictions, and run clean label evaluation."""
        gathered_results = [self.test_results]
        flattened_results = [item for sublist in gathered_results for item in sublist]

        # Prepare prediction data
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
        predictions_df = pd.DataFrame(data)
        
        # Save predictions to the VLM results directory with proper naming
        model_with_resolution = self.embs_space  # e.g., 'RADIO_378'
        save_dir = Path(__file__).parent.parent.parent / 'eval' / 'results' / 'vlm' / model_with_resolution
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename following the pattern: model_resolution_classifier_context_labels[_train].csv
        split_suffix = "_train" if self.dataloader == 'train' else ""
        filename = f"{model_with_resolution}_classifier_{self.context}_{self.labels_option}{split_suffix}"
        predictions_path = save_dir / f"{filename}.csv"
        
        # Save predictions
        predictions_df.to_csv(predictions_path, index=False)
        print(f"✓ Predictions saved to: {predictions_path}")
        
        # Run clean label evaluation and save accuracy results
        try:
            # Get path to clean validation labels
            clean_labels_path = self.cfg.path.cleaner_validation
            
            if os.path.exists(clean_labels_path):
                print("\n🔄 Running evaluation on clean labels...")
                
                # Use the save_accuracy_results_csv function which handles both validation and clean validation accuracy
                validation_accuracy, clean_validation_accuracy = save_accuracy_results_csv(
                    predictions_df=predictions_df,
                    clean_labels_path=clean_labels_path,
                    directory=f'results/vlm/{model_with_resolution}',
                    filename=filename,
                    verbose=True
                )
                
                print_success = lambda msg: print(f"\033[92m{msg}\033[0m")
                print_success(f"✓ Validation Accuracy: {validation_accuracy:.2f}%")
                print_success(f"✓ Clean Validation Accuracy: {clean_validation_accuracy:.2f}%")
                print_success(f"✓ Accuracy results saved to: {save_dir / f'{filename}_accuracy.csv'}")
            else:
                print(f"Warning: Clean validation labels file not found at: {clean_labels_path}")
                print("Skipping clean label evaluation.")
                
        except Exception as e:
            print(f"Warning: Error during clean label evaluation: {e}")
            print("Predictions were saved successfully, but accuracy evaluation failed.")

        print(f"\n🎉 Multiplication completed successfully!")
        print(f"✓ Results saved to: {predictions_path}")
        
        return predictions_path

