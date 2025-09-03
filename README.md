# BMVC 2025 Vision-Language Model Image Recognition

This repository contains the code for our BMVC 2025 paper on vision-language model image recognition. 

## Experiments Entry Point (`exp_launcher.py`)

`exp_launcher.py` is the single entry point for running all main experiments.

### Usage

- Basic syntax:
  - `python exp_launcher.py <model_name> [mode] [template] [labels_option]`
  - `python exp_launcher.py knn <embedding_space> [split]`
  - `python exp_launcher.py few-shot <embedding_space> <m_sample>`
  - `python exp_launcher.py <embedding_space> kfold <n_folds> <split>`
- Supported models:
  - `SigLIP`, `SigLIP2`, `CLIP`, `OpenCLIP`, `DINOv2`, `EfficientNet-L2`, `EfficientNet-V2`, `RADIO`
- Modes:
  - `classifier` (default), `embedder` (for VLMs and `DINOv2`), `all_templates` (SigLIP2 and OpenCLIP only), `count_params`, `knn`, `few-shot`, `kfold`, `combination` (SigLIP2 only)
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

**KNN Classification:**
```bash
# Standard KNN with SigLIP2 embeddings (classify val set using train neighbors)
python exp_launcher.py knn SigLIP2

# KNN with different embedding spaces and splits
python exp_launcher.py knn DINOv2 val
python exp_launcher.py knn SigLIP2 train  # Classify train set using val neighbors
```

**Few-shot KNN Classification:**
```bash
# Few-shot KNN with 10 samples per class (runs 250 iterations)
python exp_launcher.py few-shot DINOv2 10

# Few-shot KNN with 20 samples per class (runs 125 iterations)
python exp_launcher.py few-shot SigLIP2 20
```

**K-fold Cross-validation:**
```bash
# 10-fold cross-validation with SigLIP2 embeddings on train split
python exp_launcher.py SigLIP2 kfold 10 train

# 5-fold cross-validation with DINOv2 embeddings on val split
python exp_launcher.py DINOv2 kfold 5 val
```

**Combination Model Pipeline:**
```bash
# Run the complete combination model pipeline (SigLIP2 only)
python exp_launcher.py SigLIP2 combination
```

## Prerequisites and Dependencies

### KNN Classification, Few-shot, and K-fold Cross-validation

Before running KNN-based experiments (`knn`, `few-shot`, `kfold`), you must first generate embeddings using the embedder mode:

```bash
# Generate embeddings first
python exp_launcher.py SigLIP2 embedder           # Generate val embeddings
python exp_launcher.py SigLIP2 embedder train     # Generate train embeddings

# Then run KNN-based experiments
python exp_launcher.py knn SigLIP2 val            # KNN classification
python exp_launcher.py few-shot SigLIP2 10        # Few-shot KNN
python exp_launcher.py SigLIP2 kfold 10 train     # K-fold cross-validation
```

### Combination Model Pipeline

![Combination Pipeline](eval/expts/vlm/SigLIP2/combination.svg)

**The combination mode requires embedder and k-fold results to be generated first.** Before running the combination model, you must:

1. **Generate SigLIP2 embeddings** (if not already done):
   ```bash
   python exp_launcher.py SigLIP2 embedder           # Generate val embeddings
   python exp_launcher.py SigLIP2 embedder train     # Generate train embeddings
   ```

2. **Generate SigLIP2 10-fold k-fold results**:
   ```bash
   python exp_launcher.py SigLIP2 kfold 10 train     # Generate 10-fold results
   ```

3. **Run the combination model**:
   ```bash
   python exp_launcher.py SigLIP2 combination        # Run combination pipeline
   ```

The combination model pipeline executes three scripts in sequence:
- VL classification precision calculation
- K-fold precision calculation (requires the 10-fold results)
- Combination model that merges both approaches

## "Cleaner Validation" Labels

The "Cleaner Validation" set is external work (not ours). We use their released CSV for evaluation.

- Source file: `clean_validation.csv` available on [GitHub](https://github.com/klarajanouskova/ImageNet/blob/main/eval_corrections/verify_images/results/clean_validation.csv)
- Reference: Kisel, et al., "Flaws of ImageNet, Computer Vision's Favorite Dataset", ICLR Blogposts, 2025. See the dataset link above.

## Output Structure

Results are saved to different directories based on the experiment type:

- **VLM Classifiers**: `results/vlm/{model_name}/{model_name}_classifier_{template}_{labels_option}.csv`
- **All Templates**: `results/all_templates/{model_name}/{model_name}_classifier_all_templates_{labels_option}.csv`
- **EfficientNet/EVA-02**: `results/supervised_models/{model_name}/{model_name}.csv`
- **Embedders**: `eval/results/embeddings/{model_name}/{model_name}_{dataloader}.npy`
- **KNN**: `eval/results/knn/{embedding_space}/knn_{embedding_space}_{split}.csv`
- **Few-shot KNN**: `eval/results/knn/{embedding_space}/few-shot/{m_sample}/` (individual CSVs)
  - Summary: `eval/results/knn/{embedding_space}/few-shot/{m_sample}_shot_accuracy_summary.csv`
- **K-fold**: `eval/results/kfold/{n_folds}fold_{split}/{embedding_space}_{n_folds}fold_[1-{n_folds}].csv`

In case of any problems, please open an issue on this GitHub repository.