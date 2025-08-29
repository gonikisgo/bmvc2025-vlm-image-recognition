import os
from pathlib import Path
import pandas as pd
from typing import List, Dict
from omegaconf import DictConfig
import numpy as np
import torch


"""
A class to load embeddings for training and validation dataset splits.
"""


class EmbsLoader:
    """
    A class responsible for loading and managing image embeddings.

    Attributes:
        path_embs (str): The directory path where the embedding files are stored.
    """

    def __init__(self, cfg: DictConfig):
        self.embs_space = cfg.test.emb_space
        embed_dir = {
            'clip': cfg.path.clip_image_embs,
            'openclip': cfg.path.openclip_image_embs,
            'siglip': cfg.path.siglip_image_embs,
            'siglipv2': cfg.path.siglipv2_image_embs,
            'siglipv2-g': cfg.path.siglipv2_g_image_embs,
            'dino': cfg.path.dino_embs,
            'radio-378': cfg.path.radio_378_image_embs,
            'radio-896': cfg.path.radio_896_image_embs,
            'radio-1024': cfg.path.radio_1024_image_embs,
        }
        print(f'Embedding space: {self.embs_space}')
        self.path_embs = embed_dir[self.embs_space]

    def __load_embeddings(self, file_name: str) -> Dict:
        file_path = os.path.join(os.path.join(self.path_embs, file_name))
        return np.load(file_path, allow_pickle=True).item()

    def load_train_embeddings(self, idx: int) -> Dict:
        return self.__load_embeddings(f'imagenet_train_{idx}.npy')

    def __concatenate_embeddings(self, datasets: List[Dict]) -> Dict[str, np.ndarray]:
        concatenated_data = {}
        for key in datasets[0].keys():
            concatenated_data[key] = np.concatenate([d[key] for d in datasets])
        return concatenated_data

    def get_train_embeddings(self) -> Dict:
        if self.embs_space == 'radio-378':
            return self.__load_embeddings(f'imagenet_train.npy')
        else:
            train_ds = [self.load_train_embeddings(idx) for idx in range(1, 5)]
            return self.__concatenate_embeddings(train_ds)

    def get_val_embeddings(self) -> Dict:
        return self.__load_embeddings(f'imagenet_val.npy')

    def get_test_embeddings(self) -> Dict:
        return self.__load_embeddings(f'imagenet_test.npy')


def create_preds_df(data: List[Dict]):
    """
    Converts prediction data to a pandas dataframe with specified top-k predictions and probabilities.

    Args:
        data (List[Dict]): A list of dictionaries containing prediction data.
    """
    if isinstance(data, list):
        if isinstance(data[0], dict):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data must be a list of dictionaries")
    else:
        raise ValueError("Data must be a list")

    return df


def save_predictions_csv(data: pd.DataFrame, directory: str = 'expts', filename: str = 'out'):
    """
    Saves prediction data from pandas dataframe to a .csv file.

    Args:
        data (pd.DataFrame): A dataframe containing prediction data.
        directory (str): The directory to save the CSV file.
        filename (str): Name of the CSV file to be saved in the 'expts' directory.
    """
    # Handle different directory path formats
    if directory.startswith('eval/'):
        # If directory starts with 'eval/', resolve from project root
        project_root = Path(__file__).parent.parent  # Go up from eval/ to project root
        parent_dir = project_root / directory
    elif os.path.isabs(directory):
        # If absolute path, use as-is
        parent_dir = Path(directory)
    else:
        # Relative path from eval/ directory
        parent_dir = Path(__file__).parent / directory
    
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    output_path = parent_dir / f'{filename}.csv'
    data.to_csv(output_path, index=False)


def eval_on_clean_labels(df, clean_labels):
    merged = pd.merge(clean_labels, df, left_on='id', right_on='img_id')
    merged['top_1_pred'] = merged['top_1_pred'].astype(str)

    merged['correct_new'] = merged.apply(
        lambda row: row['top_1_pred'].strip() in str(row['proposed_labels']).strip().split(", ")
        if "," in str(row['proposed_labels'])
        else row['top_1_pred'].strip() == str(row['proposed_labels']).strip(),
        axis=1
    )

    merged['correct_orig'] = merged['top_1_pred'] == merged['original_label_x']

    print("Accuracy on 'clean' labels: ", np.round(merged['correct_new'].mean() * 100, 2))
    return merged


def save_accuracy_results_csv(predictions_df: pd.DataFrame, clean_labels_path: str, directory: str = 'expts', filename: str = 'out', project_root=None):
    """
    Saves accuracy results to CSV files including validation accuracy and clean validation accuracy.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing model predictions
        clean_labels_path (str): Path to the clean validation labels CSV file
        directory (str): Directory to save the CSV files (relative to eval/ folder)
        filename (str): Base filename for the CSV files
        project_root: Path to project root. If None, will be inferred from this file's location
    """
    try:
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        # Load clean validation labels
        clean_labels = pd.read_csv(clean_labels_path)
        
        # Calculate validation accuracy (top-1 accuracy on original labels)
        if 'top_1_pred' in predictions_df.columns and 'original_label' in predictions_df.columns:
            validation_accuracy = (predictions_df['top_1_pred'] == predictions_df['original_label']).mean() * 100
        else:
            validation_accuracy = 0.0
            print("Warning: Could not calculate validation accuracy - missing required columns")
        
        # Calculate clean validation accuracy
        try:
            clean_results = eval_on_clean_labels(predictions_df, clean_labels)
            clean_validation_accuracy = clean_results['correct_new'].mean() * 100
        except Exception as e:
            print(f"Warning: Could not calculate clean validation accuracy: {e}")
            clean_validation_accuracy = 0.0
        
        # Create accuracy results DataFrame
        accuracy_data = {
            'Metric': ['Validation', 'Cleaner Validation'],
            'Accuracy (%)': [round(validation_accuracy, 2), round(clean_validation_accuracy, 2)]
        }
        
        accuracy_df = pd.DataFrame(accuracy_data)
        
        # Save accuracy results CSV - always route from project root to eval folder
        save_dir = project_root / 'eval' / directory
        save_dir.mkdir(parents=True, exist_ok=True)
        accuracy_output_path = save_dir / f'{filename}_accuracy.csv'
        accuracy_df.to_csv(accuracy_output_path, index=False)
        print(f'Accuracy results saved to: {accuracy_output_path}')
        
        return validation_accuracy, clean_validation_accuracy
        
    except Exception as e:
        print(f"Error saving accuracy results: {e}")
        return 0.0, 0.0


def count_parameters_simple(model):
    """Count parameters using simple PyTorch parameter counting."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def save_embeddings_to_npy(results, model_name, dataloader_name, convert_labels_to_int=True, project_root=None):
    """
    Save embeddings results to NPY file. This function can be used by all embedder models
    to avoid code duplication.
    
    Args:
        results: List of results from model testing, can be nested lists
        model_name: Name of the model (used for directory and filename)
        dataloader_name: Name of the dataloader (used for filename)
        convert_labels_to_int: Whether to convert labels to integers (default: True)
        project_root: Path to project root. If None, will be inferred from this file's location
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    # Flatten nested results if needed
    flattened_results = []
    for sublist in results:
        if isinstance(sublist, list):
            flattened_results.extend(sublist)
        else:
            flattened_results.append(sublist)
    
    labels, image_names, embeddings = [], [], []

    for output in flattened_results:
        # Handle tensor conversion only if torch is available
        if torch is not None and torch.is_tensor(output[0]):
            batch_labels = output[0].cpu().numpy()
        else:
            batch_labels = output[0]
            
        batch_image_names = output[1]
        
        if torch is not None and torch.is_tensor(output[2]):
            batch_embeddings = output[2].cpu().numpy()
        else:
            batch_embeddings = output[2]

        if convert_labels_to_int:
            labels.extend([int(label) for label in batch_labels])
        else:
            labels.extend(batch_labels)
        image_names.extend(batch_image_names)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    data = {'label': labels, 'image_name': image_names, 'embedding': embeddings}

    # Print detailed info matching classifier style
    print(f"Processing embeddings: {len(data['label'])} images, {len(data['image_name'])} filenames, embedding shape: {data['embedding'].shape}")

    # Save to /eval/expts/embeddings/{model_name}/ directory structure
    save_dir = project_root / 'eval' / 'expts' / 'embeddings' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'{model_name}_{dataloader_name}.npy'
    np.save(save_path, data)
    print(f'✓ Embeddings data saved to: {save_path}')
    
    return save_path
