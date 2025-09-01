#!/usr/bin/env python3
import sys
import subprocess
import torch
from pathlib import Path

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
    import hydra
    from omegaconf import DictConfig
    
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

def validate_vlm_classifier_params(template, labels_option):
    """Validate template and labels_option for VLM classifier mode."""
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
    
    return True

def validate_all_templates_params(model, labels_option):
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

# ============================================================================
# VLM CLASSIFICATION HANDLING
# ============================================================================

def handle_vlm_classifier(model, template, labels_option, config, run_script):
    """Handle VLM models in classifier mode."""
    print(f"Running {model} in classifier mode")
    print(f"Results will be saved to: expts/vlm/{model}/{model}_classifier_{template or '0'}_{labels_option or 'mod'}")
    
    # Validate parameters
    if not validate_vlm_classifier_params(template, labels_option):
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
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        context = template or '0'
        labels = labels_option or 'mod'
        print_success(f"\n✓ {model} classifier completed successfully!")
        print(f"✓ Predictions saved to: expts/vlm/{model}/{model}_classifier_{context}_{labels}.csv")
        print(f"✓ Accuracy results saved to: expts/vlm/{model}/{model}_classifier_{context}_{labels}_accuracy.csv")
    else:
        print_error(f"\n✗ {model} classifier failed!")
    
    return result.returncode == 0

# ============================================================================
# VLM EMBEDDER HANDLING
# ============================================================================

def handle_vlm_embedder(model, config, run_script, dataloader=None):
    """Handle VLM models in embedder mode."""
    dataloader = dataloader or 'val'  # Default to 'val' if not specified
    
    print(f"Running {model} in embedder mode with {dataloader} dataloader")
    print(f"Results will be saved to: eval/expts/embeddings/{model}/{model}_{dataloader}.npy")
    
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
        print(f"✓ Embeddings saved to: eval/expts/embeddings/{model}/{model}_{dataloader}.npy")
    else:
        print_error(f"\n✗ {model} embedder failed!")
    
    return result.returncode == 0

# ============================================================================
# EFFICIENTNET CLASSIFICATION HANDLING
# ============================================================================

def handle_efficientnet_classifier(model, config, run_script, template, labels_option):
    """Handle EfficientNet models (classifier only)."""
    print(f"Running {model} in classifier mode")
    print(f"Results will be saved to: expts/supervised_models/{model}/{model}")
    
    # EfficientNet models don't support templates or labels_option
    if template:
        print_warning(f"EfficientNet models don't support templates, ignoring template: {template}")
    if labels_option:
        print_warning(f"EfficientNet models don't support labels_option, ignoring: {labels_option}")
    
    # Build command
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print_success(f"\n✓ {model} classifier completed successfully!")
        print(f"✓ Predictions saved to: expts/supervised_models/{model}/{model}.csv")
        print(f"✓ Accuracy results saved to: expts/supervised_models/{model}/{model}_accuracy.csv")
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
    if not validate_all_templates_params(model, labels_option):
        return False
    
    # Set default labels_option if not provided
    if not labels_option:
        labels_option = 'mod'
    
    print(f"Running {model} in all_templates mode with {labels_option} labels")
    print(f"Results will be saved to: expts/vlm/{model}/{model}_classifier_all_templates_{labels_option}")
    
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
        print(f"✓ Results saved to: expts/vlm/{model}/{model}_classifier_all_templates_{labels_option}")
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
# SPECIAL MODEL HANDLING
# ============================================================================

def handle_radio_model(mode, config, run_script):
    """Handle RADIO model with appropriate config and script selection."""
    if mode == 'classifier' or mode == 'count_params':
        config = 'radio'
        run_script = 'run_radio_torch.py'
    else:  # embedder mode
        config = 'radio_embedder'
        run_script = 'run_radio.py'
    return config, run_script

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
# MAIN FUNCTION
# ============================================================================

def show_usage():
    """Display usage information."""
    print("Usage: python exp_launcher.py <model_name> [mode] [template/dataloader] [labels_option]")
    print("Models: SigLIP, SigLIP2, CLIP, OpenCLIP, DINO, EfficientNet-L2, EfficientNet-V2, RADIO")
    print("Modes: classifier (default), embedder (not available for EfficientNet, DINO only), count_params, all_templates")
    print("Templates: 0-7, avg, avg_prime (only for classifier mode of VLM models)")
    print("Dataloaders: train, val (only for embedder mode, defaults to 'val')")
    print("Labels Options: wordnet, openai, mod (only for VLM models, ignored for EfficientNet and RADIO)")
    print("\nOutput Structure:")
    print("  VLM Classifiers: Results saved to expts/vlm/{model_name}/{model_name}_classifier_{context}_{labels_option}.csv")
    print("  EfficientNet Classifiers: Results saved to expts/supervised_models/{model_name}/{model_name}.csv")
    print("  Embedders: Results saved to eval/expts/embeddings/{model_name}/{model_name}_{dataloader}.npy")
    print("\nExamples:")
    print("  # VLM Classification:")
    print("  python exp_launcher.py CLIP classifier 0 wordnet  # Run CLIP classifier with template 0 and WordNet labels")
    print("  # → Saves to: expts/vlm/CLIP/CLIP_classifier_0_wordnet.csv")
    print("  python exp_launcher.py SigLIP classifier avg openai  # Run SigLIP classifier with avg template and OpenAI labels")
    print("  # → Saves to: expts/vlm/SigLIP/SigLIP_classifier_avg_openai.csv")
    print("  # VLM Embedders:")
    print("  python exp_launcher.py SigLIP embedder            # Run SigLIP embedder (defaults to val)")
    print("  # → Saves to: eval/expts/embeddings/SigLIP/SigLIP_val.npy")
    print("  python exp_launcher.py SigLIP embedder train      # Run SigLIP embedder with train dataloader")
    print("  # → Saves to: eval/expts/embeddings/SigLIP/SigLIP_train.npy")
    print("  python exp_launcher.py DINO                      # Run DINO embedder (mode enforced, defaults to val)")
    print("  # → Saves to: eval/expts/embeddings/DINOv2/DINOv2_val.npy")
    print("  python exp_launcher.py DINO embedder train       # Run DINO embedder with train dataloader")
    print("  # → Saves to: eval/expts/embeddings/DINOv2/DINOv2_train.npy")
    print("  # EfficientNet Classification:")
    print("  python exp_launcher.py EfficientNet-L2            # Run EfficientNet-L2 classifier")
    print("  # → Saves to: expts/supervised_models/EfficientNet-L2/EfficientNet-L2.csv")
    print("  python exp_launcher.py EVA-02                      # Run EVA-02 classifier")
    print("  # → Saves to: expts/supervised_models/EVA-02/EVA-02.csv")
    print("  # All Templates Evaluation:")
    print("  python exp_launcher.py SigLIP2 all_templates mod  # Run SigLIP2 in all_templates mode with mod labels")
    print("  # → Saves to: expts/vlm/SigLIP2/SigLIP2_classifier_all_templates_mod.csv")
    print("  python exp_launcher.py OpenCLIP all_templates wordnet  # Run OpenCLIP in all_templates mode with WordNet labels")
    print("  # → Saves to: expts/vlm/OpenCLIP/OpenCLIP_classifier_all_templates_wordnet.csv")
    print("  # Special cases:")
    print("  python exp_launcher.py RADIO classifier           # Run RADIO classifier (uses radio config & run_radio_torch.py)")
    print("  # → Saves to: expts/vlm/RADIO/RADIO_classifier_0_mod.csv")
    print("  python exp_launcher.py RADIO embedder             # Run RADIO embedder (uses radio_embedder config & run_radio.py, defaults to val)")
    print("  # → Saves to: eval/expts/embeddings/RADIO/RADIO_val.npy")
    print("  python exp_launcher.py RADIO embedder train       # Run RADIO embedder with train dataloader")
    print("  # → Saves to: eval/expts/embeddings/RADIO/RADIO_train.npy")
    print("  python exp_launcher.py RADIO count_params         # Count parameters for RADIO model (uses radio config & run_radio_torch.py)")
    print("  python exp_launcher.py CLIP count_params          # Count parameters for CLIP model")

def main():
    if len(sys.argv) < 2:
        show_usage()
        return

    model = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "classifier"
    
    # Handle count_params mode
    if mode == "count_params":
        count_model_parameters(model)
        return
    
    # Parse parameters based on mode
    # For classifier and all_templates: template and labels_option
    # For embedder: dataloader (using template position)
    template = sys.argv[3] if len(sys.argv) > 3 else None
    labels_option = sys.argv[4] if len(sys.argv) > 4 else None
    dataloader = None
    
    # For embedder mode, repurpose template parameter as dataloader
    if mode == "embedder":
        dataloader = template  # Use template position for dataloader
        template = None  # Embedders don't use templates
        labels_option = None  # Embedders don't use labels_option
    
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
        print("Available: SigLIP, SigLIP2, CLIP, OpenCLIP, DINO, EfficientNet-L2, EfficientNet-V2, EVA-02, RADIO")
        return
    
    config, run_script = config_map[model]
    
    # Validate mode
    if mode not in ["classifier", "embedder", "all_templates"]:
        print(f"Unknown mode: {mode}")
        print("Available modes: classifier, embedder, all_templates")
        return
    
    # Handle different model types and modes
    success = False
    
    # Validate mode-specific constraints before any model-specific handling
    if mode == "all_templates":
        if not validate_all_templates_params(model, labels_option):
            return
        success = handle_all_templates_evaluation(model, labels_option, config, run_script)
        return  # Early return after handling all_templates
    
    # Special handling for DINOv2 (embedder only) - only after all_templates check
    if model == 'DINOv2':
        mode, template, labels_option, dataloader = handle_dino_model(mode, template, labels_option, dataloader)
    
    # Special handling for RADIO (different configs and scripts for different modes)
    if model == 'RADIO':
        config, run_script = handle_radio_model(mode, config, run_script)
        # RADIO doesn't support labels_option
        if labels_option:
            print(f"RADIO doesn't support labels_option, ignoring: {labels_option}")
            labels_option = None
    
    # Validate parameters based on mode
    if mode == "embedder":
        if not validate_embedder_params(dataloader):
            return
    
    # Route to appropriate handler based on model type and mode
    if model.startswith('EfficientNet') or model == 'EVA-02':
        # EfficientNet models and EVA-02 only support classifier mode
        if mode != "classifier":
            print(f"{model} models only support classifier mode, ignoring mode: {mode}")
        success = handle_efficientnet_classifier(model, config, run_script, template, labels_option)
    elif mode == "classifier":
        # VLM classification
        if model in ['SigLIP', 'SigLIP2', 'CLIP', 'OpenCLIP', 'RADIO']:
            success = handle_vlm_classifier(model, template, labels_option, config, run_script)
        else:
            print(f"Classifier mode not supported for {model}")
    elif mode == "embedder":
        # VLM embedder
        if model in ['SigLIP', 'SigLIP2', 'CLIP', 'OpenCLIP', 'DINOv2', 'RADIO']:
            success = handle_vlm_embedder(model, config, run_script, dataloader)
        else:
            print(f"Embedder mode not supported for {model}")
    
    if not success:
        print(f"Failed to run {model} in {mode} mode")

if __name__ == "__main__":
    main()
