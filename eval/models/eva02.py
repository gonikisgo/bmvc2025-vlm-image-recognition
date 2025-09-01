import sys
from pathlib import Path
import torch
from torch import nn
import timm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from eval.models.pl_model import TimmClassifier


class EVA02(TimmClassifier):
    """
    EVA02 classifier for ImageNet inference using TIMM library.
    """
    
    # Add EVA02 models to the model mapping
    model2timm = {
        'efficientnet_l2': 'hf_hub:timm/tf_efficientnet_l2.ns_jft_in1k',
        'efficientnet_v2': 'hf_hub:timm/tf_efficientnetv2_xl.in21k_ft_in1k',
        'eva02': 'hf_hub:timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
    }
    
    def __init__(self, cfg, is_training=False, reset_last_layer=False):
        super().__init__(cfg, is_training=is_training)
        
        # Override the model name if not set correctly
        if not hasattr(self, 'model_name') or self.model_name not in self.model2timm:
            self.model_name = 'eva02'
        
        # Recreate model with EVA02 specific configuration
        print(f"Loading EVA02 model: {self.model2timm[self.model_name]}")
        
        self.model = timm.create_model(
            self.model2timm[self.model_name], 
            num_classes=self.num_classes,
            pretrained=True
        )
        
        # Configure model for inference
        self.model.eval()
        
        print(f"Loaded EVA02 model: {self.model2timm[self.model_name]}")
        print(f"Model input size: {self.model.default_cfg['input_size']}")
        print(f"Model loaded successfully. Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def predict_step(self, images, labels):
        """
        Prediction step for EVA02 model.
        
        Args:
            images: Input images tensor
            labels: Ground truth labels
            
        Returns:
            tuple: (predictions, probabilities)
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        with torch.no_grad():
            # Use autocast only if device supports it (CUDA/MPS)
            if self.device.type in ['cuda', 'mps']:
                with torch.amp.autocast(self.device.type):
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            # Apply softmax to get probabilities
            probs_full = outputs.softmax(dim=1)
            
            # Debug: Check if outputs are reasonable
            if torch.isnan(outputs).any():
                print("WARNING: NaN values detected in model outputs!")
            if torch.isinf(outputs).any():
                print("WARNING: Inf values detected in model outputs!")
            
            # Get top-k predictions and their probabilities
            probs, preds = torch.topk(probs_full, k=self.topk, dim=-1)
            
            # Debug: Check if probs and preds have reasonable values
            if probs.max() == 0 or probs.min() < 0:
                print(f"WARNING: Unusual probability values - min: {probs.min():.6f}, max: {probs.max():.6f}")
        
        # Clean up GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return preds, probs
    
    def create_label_confidence(self, labels, outputs):
        """
        Create supplemental information for label confidence.
        
        Args:
            labels: Ground truth labels
            outputs: Model outputs (softmax probabilities)
            
        Returns:
            torch.Tensor: Supplemental confidence information
        """
        batch_size = labels.shape[0]
        # Return empty tensor for now - can be extended for specific analysis
        return torch.zeros(batch_size, 4, device=labels.device)
