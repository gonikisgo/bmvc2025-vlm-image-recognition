#!/usr/bin/env python3
import os
import sys
import subprocess
import torch
import hydra
import argparse
from pathlib import Path
from omegaconf import DictConfig

# Add the project root to the path to import eval modules
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from eval.utils import count_parameters_simple
from eval.models.clip import CLIP
from eval.models.siglip import SigLIP
from eval.models.siglip2 import SigLIP2
from eval.models.openclip import OpenCLIP
from eval.models.radio import RADIOEmbedder
from eval.models.radio_embedder_torch import RADIOTorchEmbedder
from eval.models.eva02 import EVA02

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def create_argument_parser():
    """Create and configure the argument parser for all supported modes."""
    parser = argparse.ArgumentParser(
        description='BMVC 2025 VLM Image Recognition - Experiment Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Main model/command argument
    parser.add_argument('model', 
                       help='Model name or command (SigLIP, SigLIP2, CLIP, OpenCLIP, DINOv2, EfficientNet-L2, EfficientNet-V2, EVA-02, RADIO, knn, few-shot)')
    
    # Mode argument (optional for some commands)
    parser.add_argument('mode', nargs='?', 
                       choices=['classifier', 'embedder', 'all_templates', 'count_params', 'combination', 'kfold'],
                       help='Operation mode')
    
    # Template options
    parser.add_argument('--template', '-t', 
                       choices=['0', '1', '2', '3', '4', '5', '6', '7', 'avg', 'avg_prime'],
                       help='Template to use (0-7, avg, avg_prime)')
    
    # Labels options
    parser.add_argument('--labels', '-l', 
                       choices=['wordnet', 'openai', 'mod'],
                       help='Label type to use (wordnet, openai, mod)')
    
    # Split options
    parser.add_argument('--split', '-s', 
                       choices=['val', 'train'],
                       default='val',
                       help='Dataset split to use (default: val)')
    
    # Dataloader for embedder mode
    parser.add_argument('--dataloader', '-d', 
                       choices=['val', 'train'],
                       default='val',
                       help='Dataloader for embedder mode (default: val)')
    
    # Resolution for RADIO
    parser.add_argument('--resolution', '-r', 
                       choices=['378', '896'],
                       type=str,
                       help='Resolution for RADIO model (378 or 896)')
    
    # k-NN specific arguments
    parser.add_argument('--embedding-space', 
                       help='Embedding space for k-NN/few-shot (e.g., SigLIP2, RADIO_378)')
    
    parser.add_argument('--set', 
                       choices=['val', 'train'],
                       help='Dataset set for k-NN evaluation')
    
    # Few-shot learning
    parser.add_argument('--few-shot', 
                       type=int,
                       help='Number of samples per class for few-shot learning')
    
    # K-fold cross-validation
    parser.add_argument('--kfold', '-k',
                       type=int,
                       help='Number of folds for k-fold cross-validation')
    
    return parser

# ============================================================================
# COLOR UTILITY FUNCTIONS
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_success(message):
    """Print a message in green to indicate success."""
    print(f"{Colors.GREEN}{message}{Colors.RESET}")

def print_error(message):
    """Print a message in red to indicate an error or failure."""
    print(f"{Colors.RED}{message}{Colors.RESET}")

def print_warning(message):
    """Print a message in yellow to indicate a warning."""
    print(f"{Colors.YELLOW}{message}{Colors.RESET}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_model_parameters(model_name):
    """Count parameters for a specific model using its actual config."""
    
    print(f"Counting parameters for {model_name}...")
    print("=" * 50)
    
    # Map model names to config files
    config_map = {
        'CLIP': 'clip',
        'SigLIP': 'siglip',
        'SigLIP2': 'siglip2',
        'OpenCLIP': 'openclip',
        'RADIO': 'radio',  # Use radio config for parameter counting
        'EVA-02': 'eva02'
    }
    
    if model_name not in config_map:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {', '.join(config_map.keys())}")
        return None
    
    try:
        # Initialize Hydra and load the config
        with hydra.initialize(config_path="conf", version_base="1.1"):
            cfg = hydra.compose(config_name="base", overrides=[f"test={config_map[model_name]}"])
        
        print(f"Loading {model_name} with config...")
        
        # Load model
        if model_name == 'CLIP':
            model = CLIP(cfg)
        elif model_name == 'SigLIP':
            model = SigLIP(cfg)
        elif model_name == 'SigLIP2':
            model = SigLIP2(cfg)
        elif model_name == 'OpenCLIP':
            model = OpenCLIP(cfg)
        elif model_name == 'RADIO':
            model = RADIOTorchEmbedder(cfg)  # Use torch version for parameter counting
        elif model_name == 'EVA-02':
            model = EVA02(cfg)
        
        # Count parameters
        # For RADIO, use the underlying model as it doesn't inherit from torch.nn.Module
        if model_name == 'RADIO':
            total_params, trainable_params = count_parameters_simple(model.model)
        else:
            total_params, trainable_params = count_parameters_simple(model)
        
        # Convert to millions for readability
        total_params_m = total_params / 1_000_000
        trainable_params_m = trainable_params / 1_000_000
        
        print(f"{model_name}:")
        print(f"  Total parameters: {total_params:,} ({total_params_m:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params_m:.2f}M)")
        print("=" * 50)
        
        # Clean up GPU memory
        if hasattr(model, 'model'):
            del model.model
        del model
        torch.cuda.empty_cache()
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'total_m': total_params_m,
            'trainable_m': trainable_params_m
        }
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def run_all_templates_processing(model_name, labels_option='mod', dataloader='val'):
    """Run the process_all_templates.py script after the model has completed."""
    print(f"\nRunning all_templates processing for {model_name}...")
    
    cmd = [
        sys.executable, 
        "eval/expts/all_templates/process_all_templates.py",
        model_name,
        labels_option,
        "--dataloader", dataloader
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print_success("All templates processing completed successfully!")
        print("Output:", result.stdout)
    else:
        print_error("All templates processing failed!")
        print_error(f"Error: {result.stderr}")
        print("Output:", result.stdout)
    
    return result.returncode == 0

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_vlm_classifier_params(template, labels_option, split='val'):
    """Validate template, labels_option, and split for VLM classifier mode."""
    if template:
        valid_templates = ['0', '1', '2', '3', '4', '5', '6', '7', 'avg', 'avg_prime', 'all_templates']
        if template not in valid_templates:
            print_error(f"Invalid template: {template}")
            print(f"Available templates: {', '.join(valid_templates)}")
            return False
    
    if labels_option:
        valid_labels_options = ['wordnet', 'openai', 'mod']
        if labels_option not in valid_labels_options:
            print_error(f"Invalid labels_option: {labels_option}")
            print(f"Available labels_options: {', '.join(valid_labels_options)}")
            return False
    
    if split:
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            print_error(f"Invalid split: {split}")
            print(f"Available splits: {', '.join(valid_splits)}")
            return False
    
    return True

def validate_all_templates_params(model):
    """Validate parameters for all_templates mode."""
    if model not in ["SigLIP2", "OpenCLIP"]:
        print_error(f"all_templates mode is only supported for SigLIP2 and OpenCLIP models")
        return False
    return True

def validate_embedder_params(dataloader):
    """Validate dataloader parameter for embedder mode."""
    if dataloader:
        valid_dataloaders = ['train', 'val']
        if dataloader not in valid_dataloaders:
            print_error(f"Invalid dataloader: {dataloader}")
            print(f"Available dataloaders: {', '.join(valid_dataloaders)}")
            return False
    return True

def validate_knn_params(embedding_space, set_param):
    """Validate parameters for k-NN mode."""
    if embedding_space:
        # Map model names to embedding space names used in the system
        valid_embedding_spaces = {
            'SigLIP': 'siglip',
            'SigLIP2': 'siglipv2',  
            'CLIP': 'clip',
            'OpenCLIP': 'openclip',
            'DINOv2': 'dino',
            'RADIO': 'radio-1024'  # Default RADIO embedding space for k-NN
        }
        
        # Allow both model names and direct embedding space names
        all_valid = list(valid_embedding_spaces.keys()) + list(valid_embedding_spaces.values()) + ['radio-896', 'radio-1024', 'siglipv2-g']
        
        if embedding_space not in all_valid:
            print_error(f"Invalid embedding_space: {embedding_space}")
            print(f"Available embedding spaces: {', '.join(all_valid)}")
            return False
    
    if set_param:
        valid_sets = ['train', 'val']
        if set_param not in valid_sets:
            print_error(f"Invalid set: {set_param}")
            print(f"Available sets: {', '.join(valid_sets)}")
            return False
    
    return True

def validate_kfold_params(embedding_space, n_folds, split):
    """Validate parameters for k-fold mode."""
    if embedding_space:
        # Map model names to embedding space names used in the system
        valid_embedding_spaces = {
            'SigLIP': 'SigLIP',
            'SigLIP2': 'SigLIP2',  
            'CLIP': 'CLIP',
            'OpenCLIP': 'OpenCLIP',
            'DINOv2': 'DINOv2',
            'RADIO': 'RADIO'
        }
        
        # Allow both model names and direct embedding space names
        all_valid = list(valid_embedding_spaces.keys()) + list(valid_embedding_spaces.values())
        
        if embedding_space not in all_valid:
            print_error(f"Invalid embedding_space: {embedding_space}")
            print(f"Available embedding spaces: {', '.join(all_valid)}")
            return False
    
    if n_folds:
        try:
            n_folds_int = int(n_folds)
            if n_folds_int < 2:
                print_error(f"Number of folds must be at least 2, got: {n_folds}")
                return False
        except ValueError:
            print_error(f"Number of folds must be an integer, got: {n_folds}")
            return False
    
    if split:
        valid_splits = ['train', 'val']
        if split not in valid_splits:
            print_error(f"Invalid split: {split}")
            print(f"Available splits: {', '.join(valid_splits)}")
            return False
    
    return True

# ============================================================================
# VLM CLASSIFICATION HANDLING
# ============================================================================

def handle_vlm_classifier(model, template, labels_option, config, run_script, split='val'):
    """Handle VLM models in classifier mode."""
    print(f"Running {model} in classifier mode")
    
    # Create filename suffix based on split
    split_suffix = "_train" if split == 'train' else ""
    context = template or '0'
    labels = labels_option or 'mod'
    
    print(f"Split: {split}")
    print(f"Results will be saved to: results/vlm/{model}/{model}_classifier_{context}_{labels}{split_suffix}")
    
    # Validate parameters
    if not validate_vlm_classifier_params(template, labels_option, split):
        return False
    
    # Build command
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    
    # Always set mode for run_clip.py
    cmd.append(f"test.mode=classifier")
    
    # Add template parameter
    if template:
        cmd.append(f"test.context='{template}'")
    
    # Add labels_option parameter
    if labels_option:
        cmd.append(f"test.labels_option={labels_option}")
    
    # Add split parameter
    cmd.append(f"test.dataloader={split}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ {model} classifier completed successfully!")
        print(f"✓ Predictions saved to: results/vlm/{model}/{model}_classifier_{context}_{labels}{split_suffix}.csv")
        print(f"✓ Accuracy results saved to: results/vlm/{model}/{model}_classifier_{context}_{labels}{split_suffix}_accuracy.csv")
    else:
        print_error(f"\n✗ {model} classifier failed!")
    
    return result.returncode == 0

# ============================================================================
# RADIO CLASSIFICATION HANDLING
# ============================================================================

def handle_radio_classifier(model, template, labels_option, config, run_script, split, resolution):
    """Handle RADIO model in classifier mode - runs RADIO, then OpenCLIP text embedder, then multiplication."""
    # Validate that all required parameters are provided
    if template is None:
        print_error("RADIO classifier requires template parameter")
        print("Available templates: 0-7, avg, avg_prime")
        return False
    if labels_option is None:
        print_error("RADIO classifier requires labels_option parameter")
        print("Available labels_options: wordnet, openai, mod")
        return False
    
    print(f"Running {model} in classifier mode with resolution {resolution}")
    print("This will execute three steps:")
    print("1. Run RADIO model for image embeddings")
    print("2. Run OpenCLIP text embedder with provided template and labels")
    print("3. Run multiplication of embeddings and evaluate on clean labels")
    print("=" * 70)
    
    # Create filename suffix based on split
    split_suffix = "_train" if split == 'train' else ""
    context = template
    labels = labels_option
    
    # Create model name with resolution for file paths
    model_with_resolution = f"{model}_{resolution}"
    
    print(f"Split: {split}")
    print(f"Template: {context}")
    print(f"Labels option: {labels}")
    print(f"Resolution: {resolution}")
    
    # Validate parameters
    if not validate_vlm_classifier_params(template, labels_option, split):
        return False
    
    # Validate resolution
    if resolution not in [378, 896]:
        print_error(f"Invalid resolution for RADIO classifier: {resolution}")
        print("Available resolutions: 378, 896")
        return False
    
    success = True
    
    # Check if RADIO embeddings already exist
    embeddings_file = f"eval/results/embeddings/{model_with_resolution}/{model_with_resolution}_{split if split == 'train' else 'val'}.npy"
    
    if os.path.exists(embeddings_file):
        print(f"\n✅ RADIO embeddings already exist: {embeddings_file}")
        print("Skipping Step 1 (RADIO model for image embeddings)")
        print_success("✓ RADIO embeddings found, proceeding to Step 2")
    else:
        # Step 1: Run RADIO model for image embeddings
        print(f"\n🔄 Step 1: Running RADIO model for image embeddings...")
        print(f"Results will be saved to: eval/results/embeddings/{model_with_resolution}/")
        
        # Build RADIO command
        radio_cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
        radio_cmd.append(f"test.mode=classifier")
        radio_cmd.append(f"test.resolution={resolution}")
        radio_cmd.append(f"test.dataloader={split}")
        
        print(f"Running RADIO command: {' '.join(radio_cmd)}")
        radio_result = subprocess.run(radio_cmd)
        
        if radio_result.returncode == 0:
            print_success("✓ RADIO model completed successfully!")
            print(f"✓ RADIO embeddings saved to: eval/results/embeddings/{model_with_resolution}/{model_with_resolution}_{split if split == 'train' else 'val'}.npy")
        else:
            print_error("✗ RADIO model failed!")
            success = False
    
    # Step 2: Run OpenCLIP text embedder with provided template and labels
    if success:
        text_embeddings_file = f"eval/results/text_embeddings/OpenCLIP/OpenCLIP_text_embeddings_{context}_{labels}.npy"
        
        if os.path.exists(text_embeddings_file):
            print(f"\n✅ OpenCLIP text embeddings already exist: {text_embeddings_file}")
            print("Skipping Step 2 (OpenCLIP text embedder)")
            print_success("✓ Text embeddings found, proceeding to Step 3")
        else:
            print(f"\n🔄 Step 2: Running OpenCLIP text embedder...")
            print(f"Template: {context}, Labels: {labels}")
            print(f"Results will be saved to: eval/results/text_embeddings/OpenCLIP/")
            
            # Build OpenCLIP command
            openclip_cmd = [sys.executable, "eval/run/run_clip.py", "test=openclip"]
            openclip_cmd.append("test.mode=text_embedder")
            
            # Add template parameter (always present after validation)
            openclip_cmd.append(f"test.context='{context}'")
            
            # Add labels_option parameter (always present after validation)
            openclip_cmd.append(f"test.labels_option={labels}")
            
            print(f"Running OpenCLIP command: {' '.join(openclip_cmd)}")
            openclip_result = subprocess.run(openclip_cmd)
            
            if openclip_result.returncode == 0:
                print_success("✓ OpenCLIP text embedder completed successfully!")
                print(f"✓ OpenCLIP text embeddings saved to: {text_embeddings_file}")
            else:
                print_error("✗ OpenCLIP text embedder failed!")
                success = False
    
    # Step 3: Run multiplication of embeddings and evaluate on clean labels
    if success:
        print(f"\n🔄 Step 3: Running multiplication of embeddings...")
        print(f"Multiplying RADIO image embeddings with OpenCLIP text embeddings")
        print(f"Results will be saved to: eval/results/vlm/{model_with_resolution}/{model_with_resolution}_classifier_{context}_{labels}{split_suffix}.csv")
        
        # Build multiplication command
        mult_cmd = [sys.executable, "eval/run/run_mult.py", "test=mult"]
        
        # Set embedding space to RADIO model with resolution
        mult_cmd.append(f"test.emb_space={model_with_resolution}")
        
        # Add template parameter 
        mult_cmd.append(f"test.context={context}")
        
        # Add labels_option parameter
        mult_cmd.append(f"test.labels_option={labels}")
        
        # Add dataloader parameter
        mult_cmd.append(f"test.dataloader={split}")
        
        print(f"Running multiplication command: {' '.join(mult_cmd)}")
        mult_result = subprocess.run(mult_cmd)
        
        if mult_result.returncode == 0:
            print_success("✓ Multiplication and evaluation completed successfully!")
            print(f"✓ Final results saved to: eval/results/vlm/{model_with_resolution}/{model_with_resolution}_classifier_{context}_{labels}{split_suffix}.csv")
            print(f"✓ Accuracy results saved to: eval/results/vlm/{model_with_resolution}/{model_with_resolution}_classifier_{context}_{labels}{split_suffix}_accuracy.csv")
        else:
            print_error("✗ Multiplication and evaluation failed!")
            success = False
    
    # Final status
    print("\n" + "=" * 70)
    if success:
        print_success("🎉 RADIO classifier pipeline completed successfully!")
        print("All three steps completed:")
        print("✓ RADIO image embeddings generated")
        print("✓ OpenCLIP text embeddings generated") 
        print("✓ Multiplication performed and clean label evaluation completed")
    else:
        print_error("❌ RADIO classifier pipeline failed!")
        print("One or more steps encountered errors. Check the output above for details.")
    
    return success

# ============================================================================
# VLM EMBEDDER HANDLING
# ============================================================================

def handle_vlm_embedder(model, config, run_script, dataloader=None):
    """Handle VLM models in embedder mode."""
    dataloader = dataloader or 'val'  # Default to 'val' if not specified
    
    print(f"Running {model} in embedder mode with {dataloader} dataloader")
    print(f"Results will be saved to: eval/results/embeddings/{model}/{model}_{dataloader}.npy")
    
    # Build command
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    
    # Always set mode for run_clip.py
    cmd.append(f"test.mode=embedder")
    
    # Add dataloader parameter
    cmd.append(f"test.dataloader={dataloader}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ {model} embedder completed successfully!")
        print(f"✓ Embeddings saved to: eval/results/embeddings/{model}/{model}_{dataloader}.npy")
    else:
        print_error(f"\n✗ {model} embedder failed!")
    
    return result.returncode == 0

# ============================================================================
# EFFICIENTNET CLASSIFICATION HANDLING
# ============================================================================

def handle_efficientnet_classifier(model, config, run_script, template, labels_option, split='val'):
    """Handle EfficientNet models (classifier only)."""
    print(f"Running {model} in classifier mode")
    
    # Add split suffix for filename display
    split_suffix = "_train" if split == 'train' else ""
    print(f"Results will be saved to: results/supervised_models/{model}/{model}{split_suffix}")
    
    # EfficientNet models don't support templates or labels_option
    if template:
        print_warning(f"EfficientNet models don't support templates, ignoring template: {template}")
    if labels_option:
        print_warning(f"EfficientNet models don't support labels_option, ignoring: {labels_option}")
    
    # Build command with dataloader parameter
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    cmd.append(f"test.dataloader={split}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ {model} classifier completed successfully!")
        print(f"✓ Predictions saved to: results/supervised_models/{model}/{model}{split_suffix}.csv")
        print(f"✓ Accuracy results saved to: results/supervised_models/{model}/{model}{split_suffix}_accuracy.csv")
    else:
        print_error(f"\n✗ {model} classifier failed!")
    
    return result.returncode == 0

# ============================================================================
# ALL TEMPLATES EVALUATION HANDLING
# ============================================================================

def handle_all_templates_evaluation(model, labels_option, config, run_script):
    """Handle all_templates evaluation mode."""
    print(f"Running {model} in all_templates mode")
    
    # Validate parameters
    if not validate_all_templates_params(model):
        return False
    
    # Set default labels_option if not provided
    if not labels_option:
        labels_option = 'mod'
    
    print(f"Running {model} in all_templates mode with {labels_option} labels")
    print(f"Results will be saved to: results/all_templates/{model}/{model}_classifier_all_templates_{labels_option}")
    
    # Use all_templates config
    config = f"{config}_all_templates"
    
    # Build command
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    
    # Always set mode for run_clip.py (all_templates uses classifier mode)
    cmd.append(f"test.mode=classifier")
    
    # Add labels_option parameter
    cmd.append(f"test.labels_option={labels_option}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # Run post-processing if successful
    if result.returncode == 0:
        print_success(f"\n✓ Model {model} completed successfully in all_templates mode.")
        print(f"✓ Results saved to: results/all_templates/{model}/{model}_classifier_all_templates_{labels_option}")
        print("Now running post-processing to generate CSV with predictions...")
        
        dataloader = 'val'  # Default, could be made configurable
        success = run_all_templates_processing(model, labels_option, dataloader)
        
        if success:
            print_success(f"\n✓ All templates processing completed successfully for {model}!")
            print("✓ The CSV with predictions has been saved and is ready for further processing.")
        else:
            print_warning(f"\n✗ Warning: All templates processing failed for {model}.")
            print("The model results may still be available, but post-processing failed.")
    else:
        print_error(f"\n✗ Model {model} failed in all_templates mode. Skipping post-processing.")
    
    return result.returncode == 0

# ============================================================================
# k-NN CLASSIFICATION HANDLING
# ============================================================================

def handle_few_shot_knn(embedding_space, m_sample, split='val'):
    """Handle few-shot k-NN classification using precomputed embeddings."""
    embedding_space = embedding_space or 'DINOv2'  # Default for few-shot
    
    print(f"Running few-shot k-NN with {embedding_space} embeddings")
    print(f"Reference set size (m): {m_sample}")
    print(f"Split: {split}")
    
    # Check if embedding files exist
    embeddings_base_dir = "eval/results/embeddings"
    train_embedding_file = f"{embeddings_base_dir}/{embedding_space}/{embedding_space}_train.npy"
    val_embedding_file = f"{embeddings_base_dir}/{embedding_space}/{embedding_space}_val.npy"
    
    if not os.path.exists(train_embedding_file):
        print_error(f"Train embedding file not found: {train_embedding_file}")
        print(f"Please run: python exp_launcher.py {embedding_space} embedder train")
        return False
        
    if not os.path.exists(val_embedding_file):
        print_error(f"Val embedding file not found: {val_embedding_file}")
        print(f"Please run: python exp_launcher.py {embedding_space} embedder")
        return False
    
    # Calculate number of iterations based on 2500/m
    n_iterations = max(1, 2500 // m_sample)
    
    print(f"Will run {n_iterations} iterations (2500/{m_sample})")
    print(f"Results will be saved to: eval/results/knn/{embedding_space}/few-shot/{m_sample}/")
    print(f"Summary will be saved to: eval/results/knn/{embedding_space}/few-shot/{m_sample}_shot_accuracy_summary.csv")
    
    # Build command
    cmd = [sys.executable, "eval/run/run_knn.py", "test=knn"]
    
    # Set few-shot mode
    cmd.append("test.mode=few-shot")
    
    # Override the embedding space
    cmd.append(f"test.emb_space={embedding_space}")
    
    # Add split parameter  
    cmd.append(f"test.split={split}")
    
    # Add m_sample parameter
    cmd.append(f"test.m_sample={m_sample}")
    
    # Add number of iterations
    cmd.append(f"test.n_iterations={n_iterations}")
    
    # Update experiment name
    exp_name = f"knn_{embedding_space}_few_shot"
    cmd.append(f"test.exp_name={exp_name}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ Few-shot k-NN classifier completed successfully!")
        print(f"✓ {n_iterations} prediction files saved to: eval/results/knn/{embedding_space}/few-shot/{m_sample}/")
        print(f"✓ Accuracy summary with confidence intervals saved to: eval/results/knn/{embedding_space}/few-shot/{m_sample}_shot_accuracy_summary.csv")
    else:
        print_error(f"\n✗ Few-shot k-NN classifier failed!")
    
    return result.returncode == 0


def handle_knn_classifier(embedding_space, set_param):
    """Handle k-NN classification using precomputed embeddings."""
    # Set default values
    embedding_space = embedding_space or 'SigLIP2'
    set_param = set_param or 'val'
    
    print(f"Running k-NN classifier with {embedding_space} embeddings")
    print(f"Set: {set_param}")

    
    # Check if embedding files exist
    embeddings_base_dir = "eval/results/embeddings"
    train_embedding_file = f"{embeddings_base_dir}/{embedding_space}/{embedding_space}_train.npy"
    val_embedding_file = f"{embeddings_base_dir}/{embedding_space}/{embedding_space}_val.npy"

    if not os.path.exists(train_embedding_file):
        print_error(f"Train embedding file not found: {train_embedding_file}")
        print(f"Please run: python exp_launcher.py {embedding_space} embedder train")
        return False
        
    if not os.path.exists(val_embedding_file):
        print_error(f"Val embedding file not found: {val_embedding_file}")
        print(f"Please run: python exp_launcher.py {embedding_space} embedder")
        return False

    print(f"Results will be saved to: eval/results/knn/{embedding_space}/knn_{embedding_space}_{set_param}.csv")

    # Validate parameters
    if not validate_knn_params(embedding_space, set_param):
        return False
    
    # Build command
    cmd = [sys.executable, "eval/run/run_knn.py", "test=knn"]
    
    # Override the embedding space to use the actual model name
    cmd.append(f"test.emb_space={embedding_space}")
    
    # Add set parameter  
    cmd.append(f"test.split={set_param}")
    
    # Update experiment name to include parameters
    exp_name = f"knn_{embedding_space}_{set_param}"
    cmd.append(f"test.exp_name={exp_name}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ k-NN classifier completed successfully!")
        print(f"✓ Predictions saved to: eval/results/knn/{embedding_space}/{exp_name}.csv")
        print(f"✓ Accuracy results saved to: eval/results/knn/{embedding_space}/{exp_name}_accuracy.csv")
    else:
        print_error(f"\n✗ k-NN classifier failed!")
    
    return result.returncode == 0

# ============================================================================
# K-FOLD VALIDATION HANDLING
# ============================================================================

def handle_kfold_validation(embedding_space, n_folds, split):
    """Handle k-fold cross-validation using precomputed embeddings."""
    # Set default values
    embedding_space = embedding_space or 'SigLIP2'
    n_folds = int(n_folds) if n_folds else 5
    split = split or 'train'  # Default to train split for k-fold
    
    print(f"Running k-fold cross-validation with {embedding_space} embeddings")
    print(f"Number of folds: {n_folds}")
    print(f"Split: {split}")
    
    actual_model_name = embedding_space
    
    # Check if embedding files exist
    embeddings_base_dir = "eval/results/embeddings"
    embedding_file = f"{embeddings_base_dir}/{actual_model_name}/{actual_model_name}_{split}.npy"
    
    if not os.path.exists(embedding_file):
        print_error(f"Embedding file not found: {embedding_file}")
        print(f"Please run: python exp_launcher.py {actual_model_name} embedder train")
        return False
    
    # Validate parameters
    if not validate_kfold_params(embedding_space, n_folds, split):
        return False
    
    print(f"Results will be saved to: eval/results/kfold/{n_folds}fold_{split}/")
    print(f"k-NN classifier will handle all {n_folds} folds internally...")
    
    # Build command - k-NN classifier handles all folds internally
    cmd = [sys.executable, "eval/run/run_knn.py", "test=knn"]
    
    # Set k-fold mode
    cmd.append("test.mode=kfold")

    # Override the embedding space
    cmd.append(f"test.emb_space={actual_model_name}")
    
    # Add split parameter
    cmd.append(f"test.split={split}")
    
    # Add k-fold parameters
    cmd.append(f"test.kfold={n_folds}")
    # Set folder=-1 to run all folds (modified k-NN classifier behavior)
    cmd.append("test.folder=-1")

    # Update experiment name
    exp_name = f"kfold_{actual_model_name}"
    cmd.append(f"test.exp_name={exp_name}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n🎉 K-fold cross-validation completed successfully!")
        print(f"✓ All {n_folds} folds completed successfully")
        print(f"✓ Results saved to: eval/results/kfold/{n_folds}fold_{split}/")
        print(f"✓ Individual fold results: {actual_model_name}_{n_folds}fold_[1-{n_folds}].csv")
    else:
        print_error(f"\n❌ K-fold cross-validation failed!")
        print("Check the output above for error details.")
    
    return result.returncode == 0

# ============================================================================
# SPECIAL MODEL HANDLING
# ============================================================================

def handle_radio_model(mode, config, run_script, resolution=None):
    """Handle RADIO model with appropriate config and script selection."""
    if mode == 'classifier' or mode == 'count_params':
        # For classifier mode, use radio_embedder_torch with resolution support
        config = 'radio'
        run_script = 'run_radio_torch.py'
        
        # Validate resolution
        if resolution not in [378, 896]:
            print_error(f"Invalid resolution for RADIO classifier: {resolution}")
            print("Available resolutions: 378, 896")
            return None, None, None
        
        print(f"Using RADIO classifier with resolution: {resolution}")
        return config, run_script, resolution
    else:  # embedder mode
        config = 'radio_embedder'
        run_script = 'run_radio.py'
        return config, run_script, None

def handle_dino_model(mode, template, labels_option, dataloader):
    """Handle DINO model (embedder only)."""
    if mode != "embedder":
        print(f"DINOv2 model only supports embedder mode, ignoring mode: {mode}")
    mode = "embedder"  # Force embedder mode for DINOv2
    
    if template and mode != "embedder":  # Only warn if template was provided but not in embedder mode
        print(f"DINOv2 model doesn't support templates, ignoring template: {template}")
        template = None
    if labels_option:
        print(f"DINOv2 model doesn't support labels_option, ignoring: {labels_option}")
        labels_option = None
    
    # For DINOv2 in embedder mode, template was repurposed as dataloader, so keep it
    return mode, template, labels_option, dataloader

# ============================================================================
# COMBINATION MODEL HANDLING
# ============================================================================

def handle_combination_model():
    """Handle combination model pipeline - runs VL precision, k-fold precision, then combination model."""
    print("Running combination model pipeline...")
    print("This will execute three scripts in sequence:")
    print("1. eval/expts/vlm/calc_vl_cls_precision.py")
    print("2. eval/expts/kfold/calc_v_cls_precision.py (requires SigLIP2 10-fold k-fold results)")
    print("3. eval/expts/vlm/SigLIP2/combination_model.py")
    print("=" * 70)
    
    success = True
    
    # Step 1: Run VL classification precision calculation (from eval/expts/vlm directory)
    print("\n🔄 Step 1: Running VL classification precision calculation...")
    cmd1 = [sys.executable, "calc_vl_cls_precision.py"]
    print(f"Running command: {' '.join(cmd1)} (from eval/expts/vlm/)")
    result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd="eval/expts/vlm")
    
    if result1.returncode == 0:
        print_success("✓ VL classification precision calculation completed successfully!")
        if result1.stdout.strip():
            print("Output:", result1.stdout)
    else:
        print_error("✗ VL classification precision calculation failed!")
        print_error(f"Error: {result1.stderr}")
        if result1.stdout.strip():
            print("Output:", result1.stdout)
        success = False
    
    # Check for 10-fold k-fold results before Step 2
    if success:
        print("\n🔄 Checking for SigLIP2 10-fold k-fold results...")
        
        # Check if 10-fold results exist for SigLIP2
        kfold_dir = "eval/results/kfold/10fold_train"
        required_files = [f"SigLIP2_10fold_{i}.csv" for i in range(1, 11)]
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(kfold_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print_error("✗ Missing required 10-fold k-fold results for SigLIP2!")
            print_error(f"Missing files: {len(missing_files)}/10")
            print("Missing files:")
            for file in missing_files[:5]:  # Show first 5 missing files
                print(f"  - {file}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more files")
            
            print("\n" + "🔧" + " To generate the required 10-fold k-fold results, run:")
            print_warning("  python exp_launcher.py SigLIP2 kfold 10 train")
            print("\nThis will generate all 10 fold files needed for the combination model.")
            print("Then re-run the combination model:")
            print_warning("  python exp_launcher.py SigLIP2 combination")
            return False
        else:
            print_success("✓ All required 10-fold k-fold results found for SigLIP2!")
    
    # Step 2: Run k-fold precision calculation (from eval/expts/kfold directory)
    if success:
        print("\n🔄 Step 2: Running k-fold precision calculation...")
        cmd2 = [sys.executable, "calc_v_cls_precision.py"]
        print(f"Running command: {' '.join(cmd2)} (from eval/expts/kfold/)")
        result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd="eval/expts/kfold")
        
        if result2.returncode == 0:
            print_success("✓ K-fold precision calculation completed successfully!")
            if result2.stdout.strip():
                print("Output:", result2.stdout)
        else:
            print_error("✗ K-fold precision calculation failed!")
            print_error(f"Error: {result2.stderr}")
            if result2.stdout.strip():
                print("Output:", result2.stdout)
            success = False
    
    # Step 3: Run combination model (from eval/expts/vlm/SigLIP2 directory)
    if success:
        print("\n🔄 Step 3: Running combination model...")
        cmd3 = [sys.executable, "combination_model.py"]
        print(f"Running command: {' '.join(cmd3)} (from eval/expts/vlm/SigLIP2/)")
        result3 = subprocess.run(cmd3, capture_output=True, text=True, cwd="eval/expts/vlm/SigLIP2")
        
        if result3.returncode == 0:
            print_success("✓ Combination model completed successfully!")
            if result3.stdout.strip():
                # Print output but limit to reasonable length
                output_lines = result3.stdout.strip().split('\n')
                if len(output_lines) > 20:
                    print("Output (first 20 lines):")
                    for line in output_lines[:20]:
                        print("  ", line)
                    print(f"  ... ({len(output_lines) - 20} more lines)")
                else:
                    print("Output:")
                    for line in output_lines:
                        print("  ", line)
        else:
            print_error("✗ Combination model failed!")
            print_error(f"Error: {result3.stderr}")
            if result3.stdout.strip():
                print("Output:", result3.stdout)
            success = False
    
    # Final status
    print("\n" + "=" * 70)
    if success:
        print_success("🎉 Combination model pipeline completed successfully!")
        print("All three steps executed without errors.")
    else:
        print_error("❌ Combination model pipeline failed!")
        print("One or more steps encountered errors. Check the output above for details.")
    
    return success

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def show_usage():
    print("=" * 80)
    print("🚀 BMVC 2025 VLM Image Recognition - Experiment Launcher")
    print("\n" + "=" * 80)
    print("🤖 SUPPORTED MODELS & MODES")
    print("=" * 80)
    
    print("\n📦 MODELS:")
    print("  • SigLIP, SigLIP2    - Vision-language models (all modes)")
    print("  • CLIP, OpenCLIP     - Vision-language models (all modes)")
    print("  • DINOv2             - Vision model (embedder only)")
    print("  • EfficientNet-L2/V2 - Supervised models (classifier only)")
    print("  • EVA-02             - Vision model (classifier only)")
    print("  • RADIO              - Vision model")
    
    print("\n⚙️  MODES:")
    print("  📊 classifier     - Zero-shot classification using vision-language models")
    print("  🗂️  embedder       - Generate embeddings for downstream tasks")
    print("  📈 all_templates  - Test all prompt templates at once")
    print("  🎯 knn            - K-nearest neighbors using pre-computed embeddings")
    print("  🎲 few-shot       - Few-shot dataset evaluation")
    print("  📊 kfold          - K-fold cross-validation for robust evaluation")
    print("  🔀 combination    - Meta-model combining multiple predictions")
    print("  🔢 count_params   - Count model parameters")

    print("\n🎛️  PARAMETERS:")
    print("  Templates (--template or -t):")
    print("    • 0-7         - Individual template indices")
    print("    • avg         - Average of all templates")
    print("    • avg_prime   - Modified average template")
    print("  Labels (--labels or -l):")
    print("    • wordnet     - WordNet-based class names")
    print("    • openai      - OpenAI-style class names") 
    print("    • mod         - Modified class names")
    print("  Splits/Data (--split/-s, --dataloader/-d):")
    print("    • val         - Validation set (default)")
    print("    • train       - Training set")
    print("  RADIO Resolution (--resolution or -r):")
    print("    • 378         - 378x378 resolution")
    print("    • 896         - 896x896 resolution")

    print("\n" + "=" * 80)
    print("🏗️  RECOMMENDED WORKFLOW & PREREQUISITES")
    print("=" * 80)
    print("  Generate embeddings FIRST - they are reused across multiple experiments:")
    print("      python exp_launcher.py <MODEL> embedder --dataloader=val")
    print("      python exp_launcher.py <MODEL> embedder --dataloader=train")
    print("      💾 Saved to: eval/results/embeddings/{MODEL}/{MODEL}_{split}.npy")

    print("\n  📊 VLM Results are needed for:")
    print("      • Benchmarking zero-shot performance against embedding-based methods")
    print("      • All templates evaluation (compares performance across different prompts)")
    print("      💡 VLM experiments run independently (no embeddings required)")

    print("\n📋 EXPERIMENT-SPECIFIC PREREQUISITES:")
    
    print("\n  🎯 k-NN Classification, Few-shot, K-fold Cross-validation:")
    print("      REQUIRES: Pre-computed embeddings (see workflow above)")
    
    print("\n  🔀 Combination Model (SigLIP2 only):")
    print("      REQUIRES ALL of the following:")
    print("      • K-fold results: python exp_launcher.py SigLIP2 kfold --kfold=10 --split=train")
    print("      • k-NN results: python exp_launcher.py knn --embedding-space=SigLIP2 --set=val")
    print("      • VLM results: python exp_launcher.py SigLIP2 classifier -t avg_prime -l mod -s val")
    print("      🚀 Then run: python exp_launcher.py SigLIP2 combination")

    print("\n🔧 RADIO PIPELINE:")
    print("  RADIO classifier runs a complete 3-step process:")
    print("  " + "─" * 60)
    print("  Step 1: 🖼️  Generate RADIO image embeddings (specified resolution)")
    print("  Step 2: 📝 Generate OpenCLIP text embeddings (specified template/labels)")
    print("  Step 3: ✖️  Element-wise multiplication + clean label evaluation")
    print("  " + "─" * 60)

    print("\n💾 OUTPUT STRUCTURE:")
    print("  • VLM Classifiers: eval/results/vlm/{model}/{model}_classifier_{template}_{labels}[_train].csv")
    print("  • RADIO Results: eval/results/vlm/RADIO_{res}/RADIO_{res}_classifier_{template}_{labels}.csv")
    print("  • Embeddings: eval/results/embeddings/{model}/{model}_{split}.npy")
    print("  • k-NN Results: eval/results/knn/{model}/knn_{model}_{set}.csv")
    print("  • K-fold Results: eval/results/kfold/{n}fold_{split}/{model}_{n}fold_[1-{n}].csv")
    
    print("\n" + "=" * 80)
    print("� USAGE EXAMPLES")
    print("=" * 80)

    print("\n🔸 STEP 1: Generate Embeddings (Recommended First)")
    print("    python exp_launcher.py SigLIP2 embedder --dataloader=val")
    print("    python exp_launcher.py SigLIP2 embedder --dataloader=train")
    print("    python exp_launcher.py DINOv2 embedder -d train")
    print("    # → Creates reusable embeddings for multiple experiments")

    print("\n🔸 VLM Zero-shot Classification")
    print("    python exp_launcher.py CLIP classifier --template=0 --labels=wordnet")
    print("    python exp_launcher.py SigLIP2 classifier -t avg -l openai -s train")
    print("    # → Direct vision-language model inference")
    
    print("\n🔸 RADIO")
    print("    python exp_launcher.py RADIO classifier -t avg -l wordnet -s val -r 896")
    print("    # → Automatically runs: embedder → text embeddings → multiplication → evaluation")

    print("\n🔸 Embedding-based Experiments (REQUIRES: 1 step completed)")
    print("    python exp_launcher.py knn --embedding-space=SigLIP2 --set=val")
    print("    python exp_launcher.py few-shot --embedding-space=DINOv2 --few-shot=10")
    print("    python exp_launcher.py SigLIP2 kfold --kfold=10 --split=train")

    print("\n🔸 Advanced Experiments")
    print("    python exp_launcher.py SigLIP2 all_templates --labels=mod")
    print("    # → Tests all prompt templates")
    print()
    print("    # Combination model (REQUIRES: k-fold + k-NN + VLM avg_prime mod results)")
    print("    python exp_launcher.py SigLIP2 combination")

    print("\n� Utility Commands")
    print("    python exp_launcher.py CLIP count_params")
    print("    # → Model parameter count")


def main():
    parser = create_argument_parser()
    
    # Check if no arguments provided or help requested
    if len(sys.argv) == 1:
        print_error("No arguments provided!")
        print("Use -h or --help for instructions")
        return
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ['help', '-help', '--help', '-h']:
        show_usage()
        return
    
    args = parser.parse_args()
    
    model = args.model
    mode = args.mode
    
    # Handle k-NN mode specially
    if model.lower() == 'knn':
        if not args.embedding_space:
            print_error("k-NN mode requires --embedding-space parameter")
            print("Example: python exp_launcher.py knn --embedding-space=SigLIP2")
            return
        
        success = handle_knn_classifier(args.embedding_space, args.set)
        if not success:
            print_error("Failed to run k-NN classifier")
        return
    
    # Handle few-shot mode specially
    if model.lower() == 'few-shot':
        if not args.embedding_space:
            print_error("Few-shot mode requires --embedding-space parameter")
            print("Example: python exp_launcher.py few-shot --embedding-space=DINOv2 --few-shot=10")
            return
        
        if not args.few_shot:
            print_error("Few-shot mode requires --few-shot parameter")
            print("Example: python exp_launcher.py few-shot --embedding-space=DINOv2 --few-shot=10")
            return
        
        # Few-shot only supports val split for now
        split = 'val'
        
        success = handle_few_shot_knn(args.embedding_space, args.few_shot, split)
        if not success:
            print_error("Failed to run few-shot k-NN classifier")
        return
    
    # Handle k-fold mode specially
    if mode and mode.lower() == 'kfold':
        if not args.kfold:
            print_error("K-fold mode requires --kfold parameter")
            print("Example: python exp_launcher.py SigLIP2 kfold --kfold=10 --split=train")
            return
        
        embedding_space = model  # First argument is embedding space for k-fold
        
        success = handle_kfold_validation(embedding_space, args.kfold, args.split)
        if not success:
            print_error("Failed to run k-fold cross-validation")
        return
    
    # Require mode for other commands
    if not mode:
        print_error(f"Mode is required for model: {model}")
        print("Available modes: classifier, embedder, all_templates, count_params, combination")
        return
    
    # Handle combination mode specially - requires SigLIP2 as model
    if mode.lower() == 'combination':
        if model != 'SigLIP2':
            print_error(f"Combination mode only supports SigLIP2 model, got: {model}")
            print("Please use: python exp_launcher.py SigLIP2 combination")
            return
        
        success = handle_combination_model()
        if not success:
            print_error("Failed to run combination model pipeline")
        return
    
    # Handle count_params mode
    if mode == "count_params":
        count_model_parameters(model)
        return
    
    # Map model names to config files and run scripts
    config_map = {
        'SigLIP': ('siglip', 'run_clip.py'),
        'SigLIP2': ('siglip2', 'run_clip.py'),
        'CLIP': ('clip', 'run_clip.py'), 
        'OpenCLIP': ('openclip', 'run_clip.py'),
        'DINOv2': ('dino', 'run_clip.py'),
        'EfficientNet-L2': ('effl2', 'run_efficientnet.py'),
        'EfficientNetV2': ('effv2', 'run_efficientnet.py'),
        'EVA-02': ('eva02', 'run_eva02.py'),
        'RADIO': ('radio', 'run_radio.py')  # Default config, will be overridden based on mode
    }
    
    if model not in config_map:
        print(f"Unknown model: {model}")
        print("Available: SigLIP, SigLIP2, CLIP, OpenCLIP, DINOv2, EfficientNet-L2, EfficientNet-V2, EVA-02, RADIO")
        return
    
    config, run_script = config_map[model]
    
    # Validate mode
    if mode not in ["classifier", "embedder", "all_templates"]:
        print(f"Unknown mode: {mode}")
        print("Available modes: classifier, embedder, all_templates, combination, knn, few-shot, kfold")
        return
    
    # Handle different model types and modes
    success = False
    
    # Validate mode-specific constraints before any model-specific handling
    if mode == "all_templates":
        success = handle_all_templates_evaluation(model, args.labels, config, run_script)
        return  # Early return after handling all_templates
    
    # Special handling for DINOv2 (embedder only) - only after all_templates check
    if model == 'DINOv2':
        mode, template, labels_option, dataloader = handle_dino_model(mode, args.template, args.labels, args.dataloader)
    
    # Special handling for RADIO (different configs and scripts for different modes)
    if model == 'RADIO':
        resolution = int(args.resolution) if args.resolution else None
        result = handle_radio_model(mode, config, run_script, resolution)
        if result[0] is None:  # Error in handle_radio_model
            return
        config, run_script, validated_resolution = result
        resolution = validated_resolution  # Use validated resolution
    
    # Validate parameters based on mode
    if mode == "embedder":
        if not validate_embedder_params(args.dataloader):
            return
    
    # Route to appropriate handler based on model type and mode
    if model.startswith('EfficientNet') or model == 'EVA-02':
        # EfficientNet models and EVA-02 only support classifier mode
        if mode != "classifier":
            print(f"{model} models only support classifier mode, ignoring mode: {mode}")
            return
        success = handle_efficientnet_classifier(model, config, run_script, args.template, args.labels, args.split)
    elif mode == "classifier":
        # VLM classification
        if model == 'RADIO':
            # Use specialized RADIO classifier handler with resolution support
            # Validate that all required parameters are provided for RADIO
            if not args.split:
                print_error("RADIO classifier requires --split parameter")
                print("Usage: python exp_launcher.py RADIO classifier --template=avg --labels=wordnet --split=val --resolution=896")
                return
            if not args.resolution:
                print_error("RADIO classifier requires --resolution parameter")
                print("Usage: python exp_launcher.py RADIO classifier --template=avg --labels=wordnet --split=val --resolution=896")
                return
            success = handle_radio_classifier(model, args.template, args.labels, config, run_script, args.split, int(args.resolution))
        elif model in ['SigLIP', 'SigLIP2', 'CLIP', 'OpenCLIP']:
            # Use provided split or default to 'val' for VLM classifiers
            split = args.split if args.split else 'val'
            success = handle_vlm_classifier(model, args.template, args.labels, config, run_script, split)
        else:
            print(f"Classifier mode not supported for {model}")
    elif mode == "embedder":
        # VLM embedder
        if model in ['SigLIP', 'SigLIP2', 'CLIP', 'OpenCLIP', 'DINOv2', 'RADIO']:
            success = handle_vlm_embedder(model, config, run_script, args.dataloader)
        else:
            print(f"Embedder mode not supported for {model}")
    
    if not success:
        print(f"Failed to run {model} in {mode} mode")

if __name__ == "__main__":
    main()
