#!/usr/bin/env python3

import sys
import os
import hydra
from pathlib import Path
from omegaconf import DictConfig

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from eval.models.mult import Mult


@hydra.main(config_path="../../conf", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    """Run multiplication of image and text embeddings."""
    
    # Initialize multiplication model
    mult_model = Mult(cfg)
    
    # Run the multiplication and prediction
    print("Running multiplication of image and text embeddings...")
    mult_model.test_step()
    
    print("Multiplication completed successfully!")


if __name__ == "__main__":
    main()
