# BMVC 2025 Vision-Language Model Image Recognition

This repository contains the code for our BMVC 2025 paper on vision-language model image recognition. 

### Installation

We recommend choose one of the following options (we used Python 3.11.9):

#### Option 1: Using venv

```bash
# Create virtual environment with Python 3.11.9
python3.11 -m venv bmvc_env
source bmvc_env/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Create conda environment with Python 3.11.9
conda create -n bmvc_env python=3.11.9
conda activate bmvc_env

# Install dependencies
pip install -r requirements.txt
```
### Dataset Setup

1. **Download ImageNet Dataset**: Download the ImageNet dataset from [image-net.org](https://www.image-net.org/download.php)

2. **Dataset Structure**: Ensure your dataset is organized with both `val/` and `train/` splits in corresponding folders:
   ```
   your_imagenet_path/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ...
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ...
   ```

3. **Configure Dataset Path**: Update the dataset path in the configuration file `conf/path/cmp.yaml` or create a new one:
   ```yaml
   data_dir: /path/to/your/imagenet_dataset/  # Update this path
   ```

## Experiments Entry Point (`exp_launcher.py`)

`exp_launcher.py` is the single entry point for running all main experiments.

### Usage

#### Vision-Language Model (VLM) Classification

**Basic VLM classification:**
```bash
# Run VLM classifier on validation set
python exp_launcher.py <MODEL> classifier --dataloader=val

# Run VLM classifier on training set  
python exp_launcher.py <MODEL> classifier --dataloader=train
```

**Available VLM models:**
- `CLIP` - OpenAI CLIP
- `OpenCLIP` - Open source CLIP variant
- `SigLIP` - Sigmoid loss CLIP
- `SigLIP2` - Enhanced SigLIP model
- `RADIO` - RADIO vision model with OpenCLIP adaptor head

**Configuration options:**
- `--labels-option`: Class name format (`mod`, `openai`, `wordnet`)
- `--context`: Template usage (`0`-`7` for individual templates, `avg` for averaged templates, `avg_prime` for averaged templates including base template 0, `all_templates` for all templates evaluated individually at once)
- `--dataloader`: Dataset split (`val` or `train`)

**Examples:**
```bash
# SigLIP2 with modified class names on validation set
python exp_launcher.py SigLIP2 classifier --dataloader=val --labels-option=mod

# CLIP with all templates evaluated individually
python exp_launcher.py CLIP classifier --dataloader=val --context=all_templates

# OpenCLIP with wordnet class names on training set
python exp_launcher.py OpenCLIP classifier --dataloader=train --labels-option=wordnet

# SigLIP2 using specific template 0 and modified class names
python exp_launcher.py SigLIP2 classifier --dataloader=val --context=0 --labels-option=mod


```

**Results are saved to:** `eval/results/vlm/{model_name}/{model_name}_classifier_{template}_{labels_option}.csv`

#### Supervised Models Evaluation

**Available supervised models:**
- `EfficientNetL2` - EfficientNetL2 pre-trained model
- `EfficientNet-V2` - EfficientNet-V2 pre-trained model  
- `EVA-02` - EVA-02 vision transformer

**Note:** Supervised models do not support template variations (`--context`) or label options (`--labels-option`).

**Examples:**
```bash
# EfficientNet-L2 on validation set
python exp_launcher.py EfficientNet-L2 classifier --dataloader=val

# EfficientNet-V2 on training set
python exp_launcher.py EfficientNetV2 classifier --dataloader=train

# EVA-02 on validation set
python exp_launcher.py EVA-02 classifier --dataloader=val
```

**Results are saved to:** `results/supervised_models/{model_name}/{model_name}_{split}.csv`

#### k-NN Classification, Few-shot, and K-fold Cross-validation

**Models supporting embedder mode and k-NN classification:**
- `CLIP`, `OpenCLIP`, `SigLIP`, `SigLIP2`, `DINOv2`, `RADIO`

**⚠️ IMPORTANT: Generate embeddings FIRST** - they are reused across multiple experiments and required for most tasks: **k-NN Classification**, **Few-shot evaluation**, **K-fold**, **Combination Model**

```bash
# Generate embeddings for validation set
python exp_launcher.py <MODEL> embedder --dataloader=val

# Generate embeddings for training set
python exp_launcher.py <MODEL> embedder --dataloader=train
```

Embeddings are saved to: `eval/results/embeddings/{MODEL}/{MODEL}_{split}.npy`

**Example workflow:**
```bash
# Step 1: Generate embeddings (do this first)
python exp_launcher.py SigLIP2 embedder --dataloader=val
python exp_launcher.py SigLIP2 embedder --dataloader=train

# Step 2: Run embedding-based experiments
python exp_launcher.py knn --embedding-space=SigLIP2 --set=val
python exp_launcher.py few-shot --embedding-space=SigLIP2 --few-shot=10
python exp_launcher.py SigLIP2 kfold --kfold=10 --split=train
```

See `python exp_launcher.py -help` for detailed prerequisites and complete workflow information.

### Combination Model Pipeline

![Combination Pipeline](eval/expts/vlm/SigLIP2/combination.svg)

**⚠️The combination mode requires k-fold, k-NN and VLM results to be generated first.**

```bash
# Run SigLIP2 combination model (after prerequisites are met)
python exp_launcher.py SigLIP2 combination
```
## "Cleaner Validation" Labels

The Cleaner Validation set is available on [GitHub](https://github.com/klarajanouskova/ImageNet/blob/main/eval_corrections/verify_images/results/clean_validation.csv)
It was produced as part of our previous work. If you use it in your research, please cite:
```bibitex
@inproceedings{kisel2025flawsofimagenet,
  author = {Kisel, Nikita and Volkov, Illia and Hanzelková, Kateřina and Janouskova, Klara and Matas, Jiri},
  title = {Flaws of ImageNet, Computer Vision's Favorite Dataset},
  abstract = {Since its release, ImageNet-1k has been a gold standard for evaluating model performance. It has served as the foundation of numerous other datasets and it has been widely used for pretraining. <br/> As models have improved, issues related to label correctness have become increasingly apparent. In this blog post, we analyze the issues, including incorrect labels, overlapping or ambiguous class definitions, training-evaluation domain shifts, and image duplicates. The solutions for some problems are straightforward. For others, we hope to start a broader conversation about how to improve this influential dataset to better serve future research.},
  booktitle = {ICLR Blogposts 2025},
  year = {2025},
  date = {April 28, 2025},
  note = {https://iclr-blogposts.github.io/2025/blog/imagenet-flaws/},
  url  = {https://iclr-blogposts.github.io/2025/blog/imagenet-flaws/}
} 
```

In case of any problems, please open an issue on this GitHub repository.