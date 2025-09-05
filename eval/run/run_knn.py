#!/usr/bin/env python3
import sys
import time
import hydra
from pathlib import Path
from omegaconf import DictConfig

# Add the project root to the path to import eval modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from eval.models.knn_classifier import KNNClassifier

"""
Script for running K-Nearest Neighbors classification on embeddings.
This script uses precomputed embeddings from other models to perform KNN classification.
"""

@hydra.main(version_base=None, config_path='../../conf', config_name='base')
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    
    print(f"Running KNN classifier with {cfg.test.emb_space} embeddings")
    print(f"K = {cfg.test.topk}, Mode = {cfg.test.mode}")
    print(f"Split = {cfg.test.split}")
    
    # Initialize and run k-NN classifier
    knn = KNNClassifier(cfg)
    knn.run()
    
    elapsed_time = time.time() - start_time
    print(f"k-NN classification completed in {elapsed_time:.2f} seconds.")

if __name__ == '__main__':
    main()
