# BMVC 2025 Vision-Language Model Image Recognition

This repository contains the code for our BMVC 2025 paper on vision-language model image recognition. 

## Experiments Entry Point (`exp_launcher.py`)

`exp_launcher.py` is the single entry point for running all main experiments.

### Usage

- Basic syntax:
  - `python exp_launcher.py <model_name> [mode] [template] [labels_option]`
- Supported models:
  - `SigLIP`, `SigLIP2`, `CLIP`, `OpenCLIP`, `DINOv2`, `EfficientNet-L2`, `EfficientNet-V2`, `RADIO`
- Modes:
  - `classifier` (default), `embedder` (for VLMs and `DINOv2`), `all_templates` (SigLIP2 and OpenCLIP only), `count_params`
- Templates (classifier mode of VLM models only):
  - `0-7`, `avg`, `avg_prime`
- Labels options (VLMs only):
  - `wordnet`, `openai`, `mod`
- Dataloader options (embedder mode only):
  - `train`, `val` (default)

### Examples

**VLM Classification:**
```bash
# Run CLIP classifier with template 0 and WordNet labels
python exp_launcher.py CLIP classifier 0 wordnet

# Run SigLIP classifier with avg template and OpenAI labels
python exp_launcher.py SigLIP classifier avg openai

# Run SigLIP classifier with avg_prime template and modified labels
python exp_launcher.py SigLIP classifier avg_prime mod
```

**VLM Embedder Mode:**
```bash
# Run SigLIP embedder (defaults to 'val' dataloader)
python exp_launcher.py SigLIP embedder

# Run SigLIP embedder with train dataloader
python exp_launcher.py SigLIP embedder train

# Run DINOv2 (embedder mode is enforced, defaults to 'val' dataloader)
python exp_launcher.py DINOv2

# Run DINOv2 embedder with train dataloader
python exp_launcher.py DINOv2 embedder train
```

**All Templates Mode:**
```bash
# Run SigLIP2 in all-templates mode with modified labels and trigger CSV post-processing
python exp_launcher.py SigLIP2 all_templates mod

# Run OpenCLIP in all-templates mode with WordNet labels
python exp_launcher.py OpenCLIP all_templates wordnet
```

**EfficientNet Classification:**
```bash
# Run EfficientNet-L2 classifier
python exp_launcher.py EfficientNet-L2
```

**RADIO Model:**
```bash
# Run RADIO classifier or embedder
python exp_launcher.py RADIO classifier
python exp_launcher.py RADIO embedder
python exp_launcher.py RADIO embedder train
```

**Parameter Counting:**
```bash
# Count parameters of VLM models
python exp_launcher.py CLIP count_params
python exp_launcher.py SigLIP2 count_params
python exp_launcher.py RADIO count_params
```

## "Cleaner Validation" labels

The "Cleaner Validation" set is external work (not ours). We use their released CSV for evaluation.

- Source file: `clean_validation.csv` available on [GitHub](https://github.com/klarajanouskova/ImageNet/blob/main/eval_corrections/verify_images/results/clean_validation.csv)
- Reference: Kisel, et al., "Flaws of ImageNet, Computer Vision's Favorite Dataset", ICLR Blogposts, 2025. See the dataset link above.

## Model parameter counting


## Two-Phase Experiment (Work in Progress)

![Two-Phase Pipeline](eval/expts/vlm/siglipv2_g_templ/2phase/two-phase.svg)

We are preparing support for a **two-phase experimental pipeline**, illustrated above.  
At present, integration into `exp_launcher.py` is still on the **todo list**, and full functionality will be added in upcoming updates.

In case of any problems, please open an issue on this GitHub repository.