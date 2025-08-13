import os
import pandas as pd
from typing import List, Dict
from omegaconf import DictConfig
import numpy as np

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


def save_predictions_csv(data: pd.DataFrame, directory: str = 'results', filename: str = 'out'):
    """
    Saves prediction data from pandas dataframe to a .csv file.

    Args:
        data (pd.DataFrame): A dataframe containing prediction data.
        directory (str): The directory to save the CSV file.
        filename (str): Name of the CSV file to be saved in the 'results' directory.
    """
    parent_dir = os.path.join(os.path.dirname(__file__), directory)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    output_path = os.path.join(parent_dir, f'{filename}.csv')
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
    print("Accuracy on original labels, same subset: ", np.round(merged['correct_orig'].mean() * 100, 2))

    return merged


def count_parameters_simple(model):
    """Count parameters using simple PyTorch parameter counting."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
