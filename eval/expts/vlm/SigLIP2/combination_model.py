import pandas as pd
from pathlib import Path
import sys

# Add the project root directory to sys.path to import eval modules
# Find the project root by looking for the directory that contains both 'eval' and 'data' folders
def find_project_root():
    current = Path(__file__).absolute()
    for parent in [current] + list(current.parents):
        if (parent / 'eval').exists() and (parent / 'data').exists() and (parent / 'eval').is_dir() and (parent / 'data').is_dir():
            return parent
    # Fallback: try going up 4 levels from the script location
    return Path(__file__).parent.parent.parent.parent.absolute()

project_root = find_project_root()
sys.path.insert(0, str(project_root))

try:
    from eval.utils import save_accuracy_results_csv
except ImportError as e:
    print(f"Error importing eval.utils: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).absolute()}")
    print(f"Project root: {project_root}")
    print(f"Contents of project root: {list(project_root.iterdir()) if project_root.exists() else 'Directory does not exist'}")
    print(f"Expected eval dir: {project_root / 'eval'}")
    print(f"Eval dir exists: {(project_root / 'eval').exists()}")
    print(f"sys.path: {sys.path}")
    raise


def load_precision_data(k_value):
    """Load precision data for both KNN and VLM models."""
    
    # Load VLM precision data
    vlm_precision_path = Path('../../../results/vlm/SigLIP2/SigLIP2_avg_prime_mod_train_precision.csv')
    if not vlm_precision_path.exists():
        raise FileNotFoundError(f"VLM precision file not found: {vlm_precision_path}")
    
    vlm_precision = pd.read_csv(vlm_precision_path)
    vlm_precision_dict = dict(zip(vlm_precision['original_label'], vlm_precision['precision']))
    
    # Load KNN precision data
    knn_precision_path = Path(f'../../../results/kfold/10fold_train/SigLIP2_{k_value}nn_10fold_precision.csv')
    if not knn_precision_path.exists():
        raise FileNotFoundError(f"Warning: k-NN precision file not found: {knn_precision_path}")

    knn_precision = pd.read_csv(knn_precision_path)
    knn_precision_dict = dict(zip(knn_precision['original_label'], knn_precision['precision']))

    return vlm_precision_dict, knn_precision_dict


def combine_predictions(k_value, verbose=False):
    """Combine k-NN and VLM predictions based on precision values."""
    
    if verbose:
        print(f'🚀 Starting combination model with k={k_value}...')
    
    # Load the prediction files
    knn_file = Path('../../../results/knn/SigLIP2/knn_SigLIP2_val.csv')
    vlm_file = Path('../../../results/vlm/SigLIP2/SigLIP2_classifier_avg_prime_mod.csv')
    
    if not knn_file.exists():
        raise FileNotFoundError(f"k-NN results file not found: {knn_file}")
    
    if not vlm_file.exists():
        raise FileNotFoundError(f"VLM results file not found: {vlm_file}")
    
    # Load data silently unless verbose
    knn_df = pd.read_csv(knn_file)
    vlm_df = pd.read_csv(vlm_file)
    
    # Load precision data
    try:
        vlm_precision_dict, knn_precision_dict = load_precision_data(k_value)
    except FileNotFoundError as e:
        if verbose:
            print(f"❌ Error loading precision data: {e}")
        return None, None
    
    # Merge the dataframes on img_id
    merged_df = pd.merge(knn_df[['original_label', 'img_id', 'k_1_pred']], 
                        vlm_df[['img_id', 'top_1_pred']], 
                        on='img_id', 
                        how='inner')
    
    if verbose:
        print(f"✅ Successfully merged {len(merged_df)} images")
    assert len(merged_df) == 50000
    
    # Add precision values for both predictions
    merged_df['knn_precision'] = merged_df['k_1_pred'].map(knn_precision_dict)
    merged_df['vlm_precision'] = merged_df['top_1_pred'].map(vlm_precision_dict)
    
    # Choose the best prediction based on precision
    def choose_best_prediction(row):
        if row['knn_precision'] > row['vlm_precision']:
            return row['k_1_pred']
        else:
            return row['top_1_pred']
    
    merged_df['combined_pred'] = merged_df.apply(choose_best_prediction, axis=1)

    # Calculate accuracy statistics
    total_images = len(merged_df)
    knn_chosen = (merged_df['knn_precision'] > merged_df['vlm_precision']).sum()
    vlm_chosen = total_images - knn_chosen
    
    # Prepare dataframes for save_accuracy_results_csv function
    # For combined accuracy: create a dataframe with combined_pred as top_1_pred
    combined_df = merged_df[['original_label', 'img_id', 'combined_pred']].copy()
    combined_df = combined_df.rename(columns={'combined_pred': 'top_1_pred'})

    # Path to clean validation labels
    clean_labels_path = '/datagrid/fix_in/clean_validation.csv'
    
    if verbose:
        print(f"\n📊 Combination Statistics:")
        print(f"   Total images: {total_images}")
        print(f"   k-NN chosen: {knn_chosen} ({knn_chosen/total_images*100:.1f}%)")
        print(f"   VLM chosen: {vlm_chosen} ({vlm_chosen/total_images*100:.1f}%)")
    
    # Save results
    output_dir = Path('../results/combination_model')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save combined accuracy results
    combined_val_acc, combined_clean_acc = None, None
    try:
        combined_val_acc, combined_clean_acc = save_accuracy_results_csv(
            combined_df, 
            clean_labels_path, 
            directory='../results/combination_model', 
            filename=f'SigLIP2_combined_k{k_value}',
            project_root=Path(__file__).parent.parent.parent.parent
        )
        if verbose:
            print(f"\n🎯 Combined accuracy: {combined_val_acc:.4f}% (Clean: {combined_clean_acc:.4f}%)")
    except Exception as e:
        if verbose:
            print(f"   Warning: Could not save combined accuracy results: {e}")

    # Save detailed results
    output_file = output_dir / f'SigLIP2_combined_k{k_value}_detailed.csv'
    result_df = merged_df[['original_label', 'img_id', 'k_1_pred', 'top_1_pred', 
                          'combined_pred', 'knn_precision', 'vlm_precision']]
    result_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"💾 Detailed results saved to: {output_file}")
        print('\n🎉 Combination model completed successfully!')
    
    return combined_val_acc, combined_clean_acc


if __name__ == "__main__":
    k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]
    results = []
    
    print("🚀 Starting combination model evaluation for all k values...")
    print("=" * 60)
    
    for i, k in enumerate(k_values, 1):
        print(f"Processing k={k} ({i}/{len(k_values)})")
        
        val_acc, clean_acc = combine_predictions(k_value=k, verbose=False)
        
        if val_acc is not None and clean_acc is not None:
            results.append({
                'k': k,
                'validation_accuracy': val_acc,
                'clean_accuracy': clean_acc
            })
        else:
            print(f"⚠️  Warning: Failed to process k={k}")
    
    # Print summary of all results
    print("\n" + "=" * 60)
    print("📊 COMBINATION MODEL ACCURACY SUMMARY")
    print("=" * 60)
    print(f"{'k':<4} {'Validation Acc (%)':<18} {'Clean Acc (%)':<15}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['k']:<4} {result['validation_accuracy']:<18.4f} {result['clean_accuracy']:<15.4f}")
    
    if results:
        # Find best performing k values
        best_val = max(results, key=lambda x: x['validation_accuracy'])
        best_clean = max(results, key=lambda x: x['clean_accuracy'])
        
        print("-" * 60)
        print(f"🏆 Best validation accuracy: k={best_val['k']} ({best_val['validation_accuracy']:.4f}%)")
        print(f"🏆 Best clean accuracy: k={best_clean['k']} ({best_clean['clean_accuracy']:.4f}%)")
    
    print("=" * 60)
    print("✅ All combination models completed successfully!")
