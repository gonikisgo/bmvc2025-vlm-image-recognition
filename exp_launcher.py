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

def count_model_parameters():
    """Count parameters for CLIP, SigLIP, SigLIP2, OpenCLIP, and RADIO models."""
    print("Counting parameters for models...")
    print("=" * 50)
    
    # Define model configurations
    model_configs = {
        'CLIP': {'model': 'CLIP', 'test': {'mode': 'base', 'exp_name': 'clip_count', 'folder': 'clip'}},
        'SigLIP': {'model': 'SigLIP', 'test': {'mode': 'base', 'exp_name': 'siglip_count', 'folder': 'siglip'}},
        'SigLIP2': {'model': 'SigLIP2', 'test': {'mode': 'base', 'exp_name': 'siglip2_count', 'folder': 'siglip2'}},
        'OpenCLIP': {'model': 'OpenCLIP', 'test': {'mode': 'base', 'exp_name': 'openclip_count', 'folder': 'openclip'}},
        'RADIO-G': {'model': 'RADIO-G', 'test': {'mode': 'base', 'exp_name': 'radio_count', 'folder': 'radio'}}
    }
    
    # Create a simple config object for model loading
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
    
    total_params_all = {}
    
    for model_name, config_dict in model_configs.items():
        try:
            print(f"Loading {model_name}...")
            
            # Create config object
            cfg = SimpleConfig(config_dict)
            
            # Load model
            if model_name == 'CLIP':
                model = CLIP(cfg)
            elif model_name == 'SigLIP':
                model = SigLIP(cfg)
            elif model_name == 'SigLIP2':
                model = SigLIP2(cfg)
            elif model_name == 'OpenCLIP':
                model = OpenCLIP(cfg)
            elif model_name == 'RADIO-G':
                model = RADIOEmbedder(cfg)
            
            # Count parameters
            total_params, trainable_params = count_parameters_simple(model)
            
            # Convert to millions for readability
            total_params_m = total_params / 1_000_000
            trainable_params_m = trainable_params / 1_000_000
            
            print(f"{model_name}:")
            print(f"  Total parameters: {total_params:,} ({total_params_m:.2f}M)")
            print(f"  Trainable parameters: {trainable_params:,} ({trainable_params_m:.2f}M)")
            print()
            
            total_params_all[model_name] = {
                'total': total_params,
                'trainable': trainable_params,
                'total_m': total_params_m,
                'trainable_m': trainable_params_m
            }
            
            # Clean up GPU memory
            if hasattr(model, 'model'):
                del model.model
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            print()
    
    # Print summary
    print("=" * 50)
    print("PARAMETER COUNT SUMMARY")
    print("=" * 50)
    
    for model_name, params in total_params_all.items():
        print(f"{model_name:12} | Total: {params['total_m']:6.2f}M | Trainable: {params['trainable_m']:6.2f}M")
    
    print("=" * 50)
    
    return total_params_all

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
        print("All templates processing completed successfully!")
        print("Output:", result.stdout)
    else:
        print("All templates processing failed!")
        print("Error:", result.stderr)
        print("Output:", result.stdout)
    
    return result.returncode == 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python exp_launcher.py <model_name> [mode] [template] [labels_option]")
        print("Models: SigLIP, SigLIP2, CLIP, OpenCLIP, DINO, EfficientNet-L2, EfficientNet-V2, RADIO")
        print("Modes: classifier (default), embedder (not available for EfficientNet, DINO only), count_params, all_templates")
        print("Templates: 0-7, avg, avg' (only for classifier mode of VLM models)")
        print("Labels Options: wordnet, openai, mod (only for VLM models, ignored for EfficientNet and RADIO)")
        print("Examples:")
        print("  python exp_launcher.py CLIP classifier 0 wordnet  # Run CLIP classifier with template 0 and WordNet labels")
        print("  python exp_launcher.py SigLIP embedder            # Run SigLIP embedder")
        print("  python exp_launcher.py SigLIP classifier avg openai  # Run SigLIP classifier with avg template and OpenAI labels")
        print("  python exp_launcher.py SigLIP2 all_templates mod  # Run SigLIP2 in all_templates mode with mod labels")
        print("  python exp_launcher.py OpenCLIP all_templates wordnet  # Run OpenCLIP in all_templates mode with WordNet labels")
        print("  python exp_launcher.py EfficientNet-L2            # Run EfficientNet-L2 classifier")
        print("  python exp_launcher.py DINO                      # Run DINO embedder (mode enforced)")
        print("  python exp_launcher.py RADIO classifier           # Run RADIO classifier")
        print("  python exp_launcher.py RADIO embedder             # Run RADIO embedder")
        print("  python exp_launcher.py count_params               # Count parameters for all VLM models")
        return
    
    model = sys.argv[1]
    
    # Handle count_params mode
    if model == "count_params":
        count_model_parameters()
        return
    
    mode = sys.argv[2] if len(sys.argv) > 2 else "classifier"
    template = sys.argv[3] if len(sys.argv) > 3 else None
    labels_option = sys.argv[4] if len(sys.argv) > 4 else None
    
    # Map model names to config files and run scripts
    config_map = {
        'SigLIP': ('siglip', 'run_clip.py'),
        'SigLIP2': ('siglip2', 'run_clip.py'),
        'CLIP': ('clip', 'run_clip.py'), 
        'OpenCLIP': ('openclip', 'run_clip.py'),
        'DINOv2': ('dino', 'run_clip.py'),
        'EfficientNet-L2': ('effl2', 'run_efficientnet.py'),
        'EfficientNet-V2': ('effv2', 'run_efficientnet.py'),
        'RADIO': ('radio', 'run_radio.py')
    }
    
    if model not in config_map:
        print(f"Unknown model: {model}")
        print("Available: SigLIP, SigLIP2, CLIP, OpenCLIP, DINO, EfficientNet-L2, EfficientNet-V2, RADIO")
        return
    
    # Check if mode is valid for the selected model
    if model.startswith('EfficientNet'):
        if mode != "classifier":
            print(f"EfficientNet models only support classifier mode, ignoring mode: {mode}")
        mode = "classifier"  # Force classifier mode for EfficientNet
        if template:
            print(f"EfficientNet models don't support templates, ignoring template: {template}")
            template = None
        if labels_option:
            print(f"EfficientNet models don't support labels_option, ignoring: {labels_option}")
            labels_option = None
    elif model == 'DINOv2':
        if mode != "embedder":
            print(f"DINOv2 model only supports embedder mode, ignoring mode: {mode}")
        mode = "embedder"  # Force embedder mode for DINOv2
        if template:
            print(f"DINOv2 model doesn't support templates, ignoring template: {template}")
            template = None
        if labels_option:
            print(f"DINOv2 model doesn't support labels_option, ignoring: {labels_option}")
            labels_option = None
    elif mode not in ["classifier", "embedder", "all_templates"]:
        print(f"Unknown mode: {mode}")
        print("Available modes: classifier, embedder, all_templates")
        return
    
    # Validate all_templates mode
    if mode == "all_templates":
        if model not in ["SigLIP2", "OpenCLIP"]:
            print(f"all_templates mode is only supported for SigLIP2 and OpenCLIP models")
            return
        if not labels_option:
            labels_option = 'mod'  # Default to 'mod' for all_templates
        print(f"Running {model} in all_templates mode with {labels_option} labels")
    
    # Validate template for VLM models in classifier mode
    if template and mode == "classifier" and not model.startswith('EfficientNet'):
        valid_templates = ['0', '1', '2', '3', '4', '5', '6', '7', 'avg', 'avg\'']
        if template not in valid_templates:
            print(f"Invalid template: {template}")
            print(f"Available templates: {', '.join(valid_templates)}")
            return
    
    # Validate labels_option for VLM models
    if labels_option and not model.startswith('EfficientNet') and model != 'RADIO':
        valid_labels_options = ['wordnet', 'openai', 'mod']
        if labels_option not in valid_labels_options:
            print(f"Invalid labels_option: {labels_option}")
            print(f"Available labels_options: {', '.join(valid_labels_options)}")
            return
    elif labels_option and (model.startswith('EfficientNet') or model == 'RADIO'):
        print(f"{model} doesn't support labels_option, ignoring: {labels_option}")
        labels_option = None
    
    config, run_script = config_map[model]
    
    # For all_templates mode, use the specific config
    if mode == "all_templates":
        config = f"{config}_all_templates"
    
    # For RADIO models, use run_radio_torch.py for embedder mode, run_radio.py for classifier mode
    if model == 'RADIO':
        if mode == 'embedder':
            run_script = 'run_radio_torch.py'
        else:  # classifier mode
            run_script = 'run_radio.py'
    
    # Build command with appropriate parameters
    cmd = [sys.executable, f"eval/run/{run_script}", f"test={config}"]
    
    # Add mode parameter for non-EfficientNet models
    if not model.startswith('EfficientNet'):
        cmd.append(f"mode={mode}")
    
    # Add template parameter for VLM models in classifier mode
    if template and mode == "classifier" and not model.startswith('EfficientNet'):
        cmd.append(f"test.context={template}")
    
    # Add labels_option parameter for VLM models
    if labels_option and not model.startswith('EfficientNet') and model != 'RADIO':
        cmd.append(f"test.labels_option={labels_option}")
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # If all_templates mode and the model run was successful, run the processing script
    if mode == "all_templates" and result.returncode == 0:
        print(f"\nModel {model} completed successfully in all_templates mode.")
        print("Now running post-processing to generate CSV with predictions...")
        
        # Get dataloader from config or use default
        dataloader = 'val'  # Default, could be made configurable
        
        success = run_all_templates_processing(model, labels_option, dataloader)
        
        if success:
            print(f"\nAll templates processing completed successfully for {model}!")
            print("The CSV with predictions has been saved and is ready for further processing.")
        else:
            print(f"\nWarning: All templates processing failed for {model}.")
            print("The model results may still be available, but post-processing failed.")
    elif mode == "all_templates":
        print(f"\nModel {model} failed in all_templates mode. Skipping post-processing.")

if __name__ == "__main__":
    main()
