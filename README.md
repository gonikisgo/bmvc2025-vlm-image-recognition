# BMVC 2025 Vision-Language Model Image Recognition

This repository contains the code for our BMVC 2025 paper on vision-language model image recognition. 

## Experiments Entry Point (`exp_launcher.py`)

`exp_launcher.py` is the single entry point for running all main experiments.

### Usage

**For comprehensive usage information and examples, run:**
```bash
python exp_launcher.py -help
```

**Quick examples:**
```bash
# VLM classification
python exp_launcher.py SigLIP classifier avg mod

# Generate embeddings
python exp_launcher.py SigLIP2 embedder

# KNN classification (requires embeddings)
python exp_launcher.py knn SigLIP2

# Few-shot learning
python exp_launcher.py few-shot DINOv2 10

# K-fold cross-validation
python exp_launcher.py SigLIP2 kfold 10 train

# Combination model pipeline
python exp_launcher.py SigLIP2 combination
```

## Prerequisites and Dependencies

### KNN Classification, Few-shot, and K-fold Cross-validation

Before running KNN-based experiments, you must first generate embeddings using the embedder mode. See `-help` for detailed prerequisites.

### Combination Model Pipeline

![Combination Pipeline](eval/expts/vlm/SigLIP2/combination.svg)

**The combination mode requires embedder and k-fold results to be generated first.** Run with `-help` to see the complete prerequisite steps.

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