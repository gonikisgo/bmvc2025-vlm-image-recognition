# BMVC 2025 Vision-Language Model Image Recognition

This repository contains the code and experimental framework for our BMVC 2025 paper on vision-language model image recognition. The repository provides comprehensive evaluation of various VLM models including CLIP, OpenCLIP, SigLIP, SigLIP2, and RADIO across multiple experimental settings.

## Repository Structure

```
bmvc2025-vlm-image-recognition/
├── conf/                    # Configuration files
├── data/                    # Data modules
├── eval/                    # Evaluation framework and experiments
│   ├── expts/              # Experimental results and processing scripts
│   │   ├── few_shot/       # Few-shot learning experiments
│   │   ├── kfold/          # K-fold cross-validation calculation
│   │   ├── big_mapping/    # Big mapping experiments
│   │   ├── radio-*-knn/    # RADIO model KNN evaluation experiments
│   │   ├── siglipv2-g-knn/ # SigLIP2-g KNN evaluation experiments
│   │   └── vlm/            # Language-based VLM model evaluation
│   ├── models/             # Model implementations
│   └── utils.py            # Utility functions including parameter counting
└── requirements.txt         # Python dependencies
```

## Experimental Results to Generate

After running the experiments, the following files should be generated in their respective directories:

### 1. Few-Shot Learning Experiments (`eval/expts/few_shot/`)

**Directories to create:**
- `1/` - 1-shot learning results (should contain 2500 CSV files)
- `5/` - 5-shot learning results (should contain 500 CSV files)
- `10/` - 10-shot learning results (should contain 250 CSV files)
- `20/` - 20-shot learning results (should contain 125 CSV files)
- `50/` - 50-shot learning results (should contain 50 CSV files)
- `100/` - 100-shot learning results (should contain 25 CSV files)
- `250/` - 250-shot learning results (should contain 10 CSV files)
- `500/` - 500-shot learning results (should contain 5 CSV files)

**Expected CSV files per directory:**
Each CSV should contain columns: `img_id`, `original_label`, `k_1_pred`, `k_1_conf`, `k_3_pred`, `k_3_conf`, etc. (up to the maximum k value for that shot count)

**Results to generate:**
- `res/1_shot_acc_ci.csv` - 1-shot accuracy with confidence intervals
- `res/10_shot_acc_ci.csv` - 10-shot accuracy with confidence intervals
- `res/100_shot_acc_ci.csv` - 100-shot accuracy with confidence intervals

### 2. K-Fold Cross-Validation (`eval/expts/kfold/`)

**Expected files:**
- `10/` directory with CSV files for each fold
- `10fold_acc.csv` - Cross-validation accuracy across all folds
- `final_pred/` directory with precision-based predictions
- `k_*/` directories with confidence-precision results for different k values

### 3. Language-Based VLM Model Evaluation (`eval/expts/vlm/`)

**Expected directories and files:**
- `clip/` - CLIP model language-based evaluation results
- `openclip/` - OpenCLIP model language-based evaluation results
- `siglip/` - SigLIP model language-based evaluation results
- `siglipv2/` - SigLIP2 model language-based evaluation results
- `siglipv2_g/` - SigLIP2-g model language-based evaluation results
- `siglipv2_g_templ/` - SigLIP2-g prompt template evaluation results
  - Contains results for each individual prompt template evaluation on SigLIP2-g
  - `2phase/` - Two-phase combination experiment results
    - `2phase.py` - Script for running combination experiments
    - `2phase_train.pkl` - Training data for combination model
    - `2phase_val.csv` - Validation results
    - `2phase_val_acc.csv` - Validation accuracy
    - `2phase_precision_out.csv` - Precision-based output
- `radio_378/` - RADIO-378 model language-based evaluation results
- `radio_896/` - RADIO-896 model language-based evaluation results
- `vlm_result.csv` - Summary of all VLM model language-based accuracies
- `vlm_result_templ.csv` - Template-based evaluation results summary

**Note:** This folder contains language-based evaluation experiments where models are evaluated using text prompts and language understanding capabilities, as opposed to KNN-based evaluation in the radio-*-knn and siglipv2-g-knn folders. The `siglipv2_g_templ/` subfolder specifically evaluates how different prompt templates affect SigLIP2-g performance, with the `2phase/` folder containing results from experiments that combine modalities for improved performance.

### 4. KNN Evaluation Experiments

**Expected files:**
- `siglipv2-g-knn/` - SigLIP2-g KNN classifier evaluation results
- `radio-378-knn/` - RADIO-378 KNN classifier evaluation results
- `radio-1024-knn/` - RADIO-1024 KNN classifier evaluation results

**Note:** These folders contain KNN-based evaluation experiments for the respective models, comparing their performance using nearest neighbor classification.

### 5. Big Mapping Experiments (`eval/expts/big_mapping/`)

**Expected files:**
- Training and validation accuracy files
- Big mapping results for different models

## Model Parameter Counting

The repository includes a utility function `count_parameters_simple()` in `eval/utils.py` for estimating model parameters:

**Usage:**
```python
from eval.utils import count_parameters_simple

# For any PyTorch model
total_params, trainable_params = count_parameters_simple(model)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

## Clean Validation Set Evaluation

The repository also includes the `eval_on_clean_labels()` function in `eval/utils.py` for evaluating model performance on the Cleaner ImageNet validation set. This function uses the Cleaner validation set from [klarajanouskova/ImageNet](https://github.com/klarajanouskova/ImageNet/blob/main/eval_corrections/verify_images/results/clean_validation.csv).

**Usage:**
```python
from eval.utils import eval_on_clean_labels

# Load clean validation labels
cleaner_val = pd.read_csv('path/to/clean_validation.csv')

# Evaluate your model predictions
results = eval_on_clean_labels(your_predictions_df, cleaner_val)

# The function prints:
# - Accuracy on 'Cleaner' labels (using corrected labels)
# - Accuracy on original labels (same subset)
```

## Expected Output Structure for Experiments Running 

After running all experiments, the directory structure should contain:

```
eval/expts/
├── few_shot/                 # Few-shot learning experiments
│   ├── 1/                    # 1-shot learning results (2500 CSV files)
│   ├── 5/                    # 5-shot learning results (500 CSV files)
│   ├── 10/                   # 10-shot learning results (250 CSV files)
│   ├── 20/                   # 20-shot learning results (125 CSV files)
│   ├── 50/                   # 50-shot learning results (50 CSV files)
│   ├── 100/                  # 100-shot learning results (25 CSV files)
│   ├── 250/                  # 250-shot learning results (10 CSV files)
│   ├── 500/                  # 500-shot learning results (5 CSV files)
│   ├── res/                  # Summary results
│   │   ├── 1_shot_acc_ci.csv
│   │   ├── 10_shot_acc_ci.csv
│   │   └── 100_shot_acc_ci.csv
│   └── process_fs_res.py
├── kfold/                    # K-fold cross-validation calculation
│   ├── 10/                   # 10-fold cross-validation results
│   │   ├── siglip2-g_kfold_10_1.csv
│   │   ├── siglip2-g_kfold_10_2.csv
│   │   ├── siglip2-g_kfold_10_3.csv
│   │   ├── siglip2-g_kfold_10_4.csv
│   │   ├── siglip2-g_kfold_10_5.csv
│   │   ├── siglip2-g_kfold_10_6.csv
│   │   ├── siglip2-g_kfold_10_7.csv
│   │   ├── siglip2-g_kfold_10_8.csv
│   │   ├── siglip2-g_kfold_10_9.csv
│   │   └── siglip2-g_kfold_10_10.csv
│   ├── 10fold_acc.csv        # Cross-validation accuracy summary
│   ├── final_pred/           # Final predictions
│   │   ├── calc_acc.py
│   │   ├── precision_based_pred_k_1.csv
│   │   ├── precision_based_pred_k_3.csv
│   │   ├── precision_based_pred_k_5.csv
│   │   ├── precision_based_pred_k_7.csv
│   │   ├── precision_based_pred_k_9.csv
│   │   ├── precision_based_pred_k_11.csv
│   │   ├── precision_based_pred_k_13.csv
│   │   └── precision_based_pred_k_21.csv
│   ├── k_1/                  # K=1 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_3/                  # K=3 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_5/                  # K=5 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_7/                  # K=7 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_9/                  # K=9 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_11/                 # K=11 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_13/                 # K=13 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_21/                 # K=21 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_49/                 # K=49 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── k_51/                 # K=51 confidence-precision results
│   │   └── v_siglip2_g_conf_prec.csv
│   ├── mult_siglipv2-g_val_mod_mean8.csv
│   ├── prec_base_predictor.py
│   ├── val_siglipv2-g_knn.csv
│   └── vl_siglip2_g_conf_prec.csv
├── vlm/                      # Language-based VLM model evaluation
│   ├── clip/                 # CLIP model language-based results
│   │   ├── mult_clip_val_mod_mean7.csv
│   │   ├── mult_clip_val_mod_mean8.csv
│   │   ├── mult_clip_val_openai_mean7.csv
│   │   └── mult_clip_val_openai_mean8.csv
│   ├── openclip/             # OpenCLIP model language-based results
│   │   ├── mult_openclip_val_mod_mean7.csv
│   │   ├── mult_openclip_val_mod_mean8.csv
│   │   ├── mult_openclip_val_openai_mean7.csv
│   │   └── mult_openclip_val_openai_mean8.csv
│   ├── siglip/               # SigLIP model language-based results
│   │   ├── mult_siglip_val_mod_mean7.csv
│   │   ├── mult_siglip_val_mod_mean8.csv
│   │   ├── mult_siglip_val_openai_mean7.csv
│   │   └── mult_siglip_val_openai_mean8.csv
│   ├── siglipv2/             # SigLIP2 model language-based results
│   │   ├── mult_siglipv2_val_mod_mean7.csv
│   │   ├── mult_siglipv2_val_mod_mean8.csv
│   │   ├── mult_siglipv2_val_openai_mean7.csv
│   │   └── mult_siglipv2_val_openai_mean8.csv
│   ├── siglipv2_g/           # SigLIP2-g model language-based results
│   │   ├── mult_siglipv2-g_val_mod_mean7.csv
│   │   ├── mult_siglipv2-g_val_mod_mean8.csv
│   │   ├── mult_siglipv2-g_val_openai_mean7.csv
│   │   └── mult_siglipv2-g_val_openai_mean8.csv
│   ├── siglipv2_g_templ/     # SigLIP2-g prompt template evaluation results
│   │   ├── 2phase/           # Two-phase combination experiments
│   │   │   ├── 2phase.py
│   │   │   ├── 2phase_train.pkl
│   │   │   ├── 2phase_val.csv
│   │   │   ├── 2phase_val_acc.csv
│   │   │   └── 2phase_precision_out.csv
│   │   ├── mult_siglipv2-g_val_templ_mod_0.csv
│   │   ├── mult_siglipv2-g_val_templ_mod_1.csv
│   │   └── mult_siglipv2-g_val_templ_mod_2.csv
│   ├── radio_378/             # RADIO-378 model language-based results
│   │   ├── mult_radio-378_val_mod_mean7.csv
│   │   ├── mult_radio-378_val_mod_mean8.csv
│   │   ├── mult_radio-378_val_openai_mean7.csv
│   │   └── mult_radio-378_val_openai_mean8.csv
│   ├── radio_896/             # RADIO-896 model language-based results
│   │   ├── mult_radio-896_val_mod_mean7.csv
│   │   ├── mult_radio-896_val_mod_mean8.csv
│   │   ├── mult_radio-896_val_openai_mean7.csv
│   │   └── mult_radio-896_val_openai_mean8.csv
│   ├── process_res.py
│   ├── vlm_result.csv        # Summary of all VLM language-based results
│   └── vlm_result_templ.csv  # Template-based evaluation results
├── siglipv2-g-knn/           # SigLIP2-g KNN evaluation experiments
│   ├── siglip_g_knn.py
│   ├── siglip_knn.py
│   ├── siglip2-g_nn_val.csv
│   ├── train_siglipv2-g_knn_acc.csv
│   ├── train_siglipv2-g_knn.csv
│   ├── val_siglipv2-g_knn_acc.csv
│   └── val_siglipv2-g_knn.csv
├── radio-378-knn/             # RADIO-378 KNN evaluation experiments
│   ├── calc_acc.py
│   ├── radio_378_val_knn.csv
│   └── radio-378-val-knn-acc.csv
├── radio-1024-knn/            # RADIO-1024 KNN evaluation experiments
│   ├── radio_knn.py
│   ├── radio-1024_nn_val.csv
│   ├── train_radio_knn_acc.csv
│   ├── train_radio_knn.csv
│   ├── val_radio_knn_acc.csv
│   └── val_radio_knn.csv
└── big_mapping/               # Big mapping experiments
    ├── process_big_mapping.py
    ├── mult_siglipv2-g_val_mod_big_mapping.csv
    ├── train_big_mapping_acc.csv
    ├── train_big_mapping.csv
    ├── val_big_mapping_acc.csv
    └── val_big_mapping.csv
```

## Data Requirements

The experiments require:
- ImageNet validation set (50,000 images)

## Getting Help

If you encounter any issues or have questions while reproducing the experiments, please open a GitHub issue in this repository. We'll be happy to help you resolve any problems.