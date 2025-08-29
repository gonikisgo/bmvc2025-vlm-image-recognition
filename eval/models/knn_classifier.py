import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from eval.utils import create_preds_df, save_predictions_csv, save_accuracy_results_csv, EmbsLoader

"""
A K-Nearest Neighbors (KNN) embedding classifier using cosine similarity.
"""


class KNNClassifier:
    def __init__(self, cfg):
        if self.mode == 'few-shot':
            self.k = min(cfg.test.topk, cfg.test.m_sample)
        else:
            self.k = cfg.test.topk

        print(f'Using {self.k} nearest neighbors')
        self.embs_loader = EmbsLoader(cfg)
        self.is_data_loaded = False
        self.exp_name = cfg.test.exp_name

        self.nn_classifier = None
        self.val_embeddings_d = None
        self.val_embeddings = None
        self.embeddings_d = None
        self.embeddings = None
        self.results = []

        self.lower_bound = cfg.test.lower_bound
        self.upper_bound = cfg.test.upper_bound

        self.seed = cfg.test.seed
        self.m_sample = cfg.test.m_sample
        self.n_iterations = cfg.test.n_iterations

        self.kfold = cfg.test.kfold
        self.folder = int(cfg.test.folder)

    def __fit(self):
        self.nn_classifier = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn_classifier.fit(self.embeddings)
        print(f'Fitted {len(self.embeddings)} train embeddings')

        print('Unique images in train:', len(np.unique(self.embeddings_d['image_name'])))
        print('Unique images in val:', len(np.unique(self.val_embeddings_d['image_name'])))
        print(self.embeddings.shape, self.val_embeddings.shape)

        print(f'Fitted {len(self.embeddings)} train embeddings')

    def __find_duplicates(self):
        distances, indices = self.nn_classifier.kneighbors(self.val_embeddings)
        print(f'Classified {len(self.val_embeddings)} val embeddings')
        similarities = 1 - distances
        indices = indices.tolist()

        i = 0
        for i, (sim, idx) in enumerate(zip(similarities, indices)):
            row = {
                'original_label': self.val_embeddings_d['label'][i],
                'img_id': self.val_embeddings_d['image_name'][i]
            }

            #n = 0
            for j in range(self.k):
                duplicate_img_id = self.embeddings_d['image_name'][idx[j]]
                if duplicate_img_id != row['img_id']: # and n < self.k:
                    row[f'top_{j + 1}_label'] = self.embeddings_d['label'][idx[j]]
                    row[f'top_{j + 1}_pred'] = duplicate_img_id
                    row[f'top_{j + 1}_prob'] = sim[j]
                    #n += 1
                else:
                    print(f'---------------------Found duplicate {duplicate_img_id} in {row["img_id"]}')
            self.results.append(row)
        print(f'last id {i}')

    def __find_duplicates_extend(self):
        k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]

        distances, indices = self.nn_classifier.kneighbors(self.val_embeddings)
        print(f'Classified {len(self.val_embeddings)} val embeddings')

        similarities = 1 - distances
        indices = indices.tolist()

        self.results = []

        i = 0
        for i, (sim, idx) in enumerate(zip(similarities, indices)):
            row = {
                'original_label': self.val_embeddings_d['label'][i],
                'img_id': self.val_embeddings_d['image_name'][i]
            }

            neighbor_labels = []
            for j in range(self.k):
                duplicate_img_id = self.embeddings_d['image_name'][idx[j]]
                if duplicate_img_id != row['img_id']:
                    neighbor_labels.append(self.embeddings_d['label'][idx[j]])
                else:
                    neighbor_labels.append(None)  # Keep list length consistent
                    print(f'---------------------Found duplicate {duplicate_img_id} in {row["img_id"]}')

            for k_ in k_values:
                if k_ > self.k:
                    break
                neighbor_subset = neighbor_labels[:k_]
                sim_subset = sim[:k_]

                valid_pairs = [(label, s) for label, s in zip(neighbor_subset, sim_subset) if label is not None]

                if valid_pairs:
                    labels_only = [label for label, _ in valid_pairs]
                    most_common = Counter(labels_only).most_common(1)[0][0]
                    row[f'k_{k_}_pred'] = most_common

                    # Now compute confidence: average similarity for neighbors with predicted label
                    matching_sims = [s for label, s in valid_pairs if label == most_common]
                    avg_similarity = sum(matching_sims) / len(matching_sims) if matching_sims else None
                    row[f'k_{k_}_conf'] = avg_similarity

                    all_sims = [s for _, s in valid_pairs]
                    avg_all_similarity = sum(all_sims) / len(all_sims) if all_sims else None
                    row[f'k_{k_}_conf_all'] = avg_all_similarity
                else:
                    row[f'k_{k_}_pred'] = None
                    row[f'k_{k_}_conf'] = None
                    row[f'k_{k_}_conf_all'] = None
            self.results.append(row)
        print(f'last id {i}')

    def __sample_train_embs(self):
        # Sample m embeddings per class
        unique_labels = np.unique(self.embeddings_d['label'])
        print(f'Unique labels: {len(unique_labels)}')
        selected_indices = []

        for label in unique_labels:
            label_indices = np.where(self.embeddings_d['label'] == label)[0]
            if len(label_indices) < self.m_sample:
                sampled = label_indices
            else:
                sampled = np.random.choice(label_indices, self.m_sample, replace=False)
            selected_indices.extend(sampled)

        print(selected_indices[:10])

        # Filter training data
        for key in ['label', 'image_name', 'embedding']:
            self.embeddings_d[key] = self.embeddings_d[key][selected_indices]

        print(f"Selected {len(selected_indices)} train embeddings from {len(unique_labels)} classes")

        # Filter validation data
        if self.lower_bound > self.upper_bound:
            raise ValueError(
                f"Invalid bounds: lower_bound ({self.lower_bound}) is greater than upper_bound ({self.upper_bound})")

        if self.upper_bound > len(self.val_embeddings_d['label']):
            self.upper_bound = len(self.val_embeddings_d['label']) + 1

        for key in ['label', 'image_name', 'embedding']:
            self.val_embeddings_d[key] = self.val_embeddings_d[key][self.lower_bound:self.upper_bound]
            print(f'Loaded {len(self.val_embeddings_d[key])} val {key} embeddings')

        self.embeddings = self.embeddings_d['embedding']
        self.val_embeddings = self.val_embeddings_d['embedding']

        self.is_data_loaded = True

    def __load_data(self):
        self.embeddings_d = self.embs_loader.get_train_embeddings()
        self.val_embeddings_d = self.embs_loader.get_val_embeddings()

        self.embeddings_d['embedding'] = np.array(self.embeddings_d['embedding'])
        self.val_embeddings_d['embedding'] = np.array(self.val_embeddings_d['embedding'])

        print(f"Loaded {len(self.val_embeddings_d['label'])} val embeddings")

        if self.mode == 'few-shot':
            self.__sample_train_embs()
        else:
            print('Default sampling: using all train embeddings')
            for key in ['label', 'image_name', 'embedding']:
                if self.lower_bound > self.upper_bound:
                    raise ValueError(
                        f"Invalid bounds: lower_bound ({self.lower_bound}) is greater than upper_bound ({self.upper_bound})")

                if self.upper_bound > len(self.val_embeddings_d[key]):
                    self.upper_bound = len(self.val_embeddings_d[key]) + 1
                self.val_embeddings_d[key] = self.val_embeddings_d[key][self.lower_bound:self.upper_bound]
                print(f'Loaded {len(self.val_embeddings_d[key])} val {key} embeddings')

            self.embeddings = self.embeddings_d['embedding']
            self.val_embeddings = self.val_embeddings_d['embedding']

            self.is_data_loaded = True

    def __save_results(self, itt):
        if len(self.results) != 0:
            if self.mode == 'few-shot':
                path = os.path.join('knn_train', 'few-shot', str(self.m_sample))
                filename = self.exp_name + f'_{self.m_sample}_{itt}'
            elif self.mode == 'kfold':
                path = os.path.join('knn_train', 'kfold', str(self.kfold))
                filename = self.exp_name + f'_{self.kfold}_{itt}'
            else:
                path = os.path.join('knn_train', 'default')
                filename = self.exp_name

            # Save predictions CSV
            save_predictions_csv(create_preds_df(self.results), directory=path, filename=filename)
            print(f'Results saved in knn_train/{filename}.csv')
            
            # Save accuracy results CSV
            try:
                clean_labels_path = self.cfg.path.cleaner_validation
                save_accuracy_results_csv(
                    predictions_df=create_preds_df(self.results),
                    clean_labels_path=clean_labels_path,
                    directory=path,
                    filename=filename
                )
                print(f'Accuracy results saved in knn_train/{filename}_accuracy.csv')
            except Exception as e:
                print(f"Warning: Could not save accuracy results: {e}")
        else:
            print('No results to save')

    def run(self):
        if self.mode == 'few-shot':
            print(f"Sampling {self.m_sample} train embeddings per class")
            for i in range(self.n_iterations):
                print(f"\n--- Iteration {i + 1}/{self.n_iterations} ---")
                np.random.seed(self.seed + i)
                self.__load_data()
                self.__fit()
                self.__find_duplicates_extend()
                self.__save_results(itt=i+1)

        elif self.mode == 'kfold':
            print(f"Using {self.kfold} folds for cross-validation")
            self.__load_data()

            X = self.embeddings_d['embedding']
            y = self.embeddings_d['label']
            image_names = self.embeddings_d['image_name']

            skf = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=self.seed)

            for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y)):
                if fold_idx != self.folder:
                    continue

                print(f"\n--- Fold {fold_idx + 1}/{self.kfold} ---")

                self.embeddings_d_fold = {
                    'embedding': X[train_index],
                    'label': y[train_index],
                    'image_name': image_names[train_index],
                }
                self.val_embeddings_d_fold = {
                    'embedding': X[val_index],
                    'label': y[val_index],
                    'image_name': image_names[val_index],
                }

                self.embeddings_d = self.embeddings_d_fold
                self.val_embeddings_d = self.val_embeddings_d_fold
                self.embeddings = self.embeddings_d['embedding']
                self.val_embeddings = self.val_embeddings_d['embedding']

                overlap = np.intersect1d(self.embeddings_d['image_name'], self.val_embeddings_d['image_name'])
                print(f"Number of overlapping image names: {len(overlap)}")

                self.__fit()
                self.__find_duplicates_extend()
                self.__save_results(itt=fold_idx + 1)

        else:
            self.__load_data()
            self.__fit()
            self.__find_duplicates_extend()
            self.__save_results(itt=0)


@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg):
    start_time = time.time()
    knn = KNNClassifier(cfg)
    knn.run()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    main()
