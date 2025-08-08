import os
import numpy as np


class ClassDictionary:
    def __init__(self, base_dir: str = None):
        curr_dir = os.path.dirname(__file__)
        self.base_dir = base_dir if base_dir else curr_dir

        self.file_paths = {
            "idx2name": os.path.join(self.base_dir, 'idx_to_name.npy'),
            "custom_name": os.path.join(self.base_dir, '../data/cls_names', 'cls_name_mod.npy'),
            "id2idx": os.path.join(self.base_dir, 'id_to_idx.npy'),
            "idx2id": os.path.join(self.base_dir, 'idx_to_id.npy'),
            "img2idx": os.path.join(self.base_dir, 'img_to_idx.npy')
        }

        self.data = {
            "idx2name": {},
            "custom_name": None,
            "id2idx": {},
            "idx2id": {},
            "img2idx": {}
        }

        self.loaded = {key: False for key in self.file_paths}

    def _load_data(self, key):
        """Loads class data from the respective file into memory."""
        try:
            self.data[key] = np.load(self.file_paths[key], allow_pickle=True).item()
            self.loaded[key] = True
        except FileNotFoundError:
            print(f"Error: File '{self.file_paths[key]}' not found.")
        except ValueError:
            print(f"Error: Invalid data in '{self.file_paths[key]}'.")
        except Exception as e:
            print(f"Unexpected error loading '{key}': {e}")

    def get_class_names(self, idx: int):
        """Retrieve class names associated with a given index."""
        if not self.loaded["idx2name"]:
            self._load_data("idx2name")
        return self.data["idx2name"].get(idx)

    def get_custom_class_name(self, idx: int):
        """Retrieve custom class name associated with a given index."""
        if not self.loaded["custom_name"]:
            self._load_data("custom_name")
        return self.data["custom_name"][idx] if self.data["custom_name"] is not None else None

    def get_class_id(self, class_id: str):
        """Retrieve ImageNet-1k index from a given class ID."""
        if not self.loaded["id2idx"]:
            self._load_data("id2idx")
        return self.data["id2idx"].get(class_id)

    def get_cls_index(self, idx: int):
        """Retrieve class ID from ImageNet-1k index."""
        if not self.loaded["idx2id"]:
            self._load_data("idx2id")
        return self.data["idx2id"].get(idx)

    def get_image_cls(self, img_id: str):
        """Retrieve class index for a given image ID."""
        if not self.loaded["img2idx"]:
            self._load_data("img2idx")
        return self.data["img2idx"].get(img_id)


def create_idx2id_map(cls_names):
    """Create a dictionary mapping indexes to class names."""
    return {idx: cls for idx, cls in enumerate(cls_names)}


def create_id2idx_map(cls_names):
    return {cls: idx for idx, cls in enumerate(cls_names)}
