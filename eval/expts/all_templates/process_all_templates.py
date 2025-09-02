import pandas as pd
from collections import Counter
from typing import List
from pathlib import Path
import argparse
import sys


def majority_vote(labels: List[str]) -> str:
    return Counter(labels).most_common(1)[0][0]

def compute_knn_predictions(df: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    top_label_cols = [f'top_{i}_pred' for i in range(1, 12)]

    missing_cols = [col for col in top_label_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    required_cols = ['original_label', 'img_id']
    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    result_df = df[['original_label', 'img_id']].copy()

    for k in k_values:
        if k > len(top_label_cols):
            print(f"Warning: k={k} exceeds available predictions ({len(top_label_cols)}). Skipping.")
            continue
        current_cols = top_label_cols[:k]
        print(f"Processing k={k} with {len(current_cols)} columns")
        result_df[f'k_{k}_pred'] = df[current_cols].apply(
            lambda row: majority_vote([(int(val) % 1000) for val in row.values]), axis=1
        )
    return result_df


def compute_knn_accuracies(df: pd.DataFrame, k_values: list) -> pd.DataFrame:
    accuracies = []

    for k in k_values:
        pred_col = f'k_{k}_pred'
        if pred_col not in df.columns:
            raise ValueError(f"Missing prediction column: {pred_col}")

        correct = (df['original_label'] == df[pred_col]).sum()
        total = len(df)
        accuracy = 100 * (correct / total) if total > 0 else 0
        accuracies.append({'k': k, 'accuracy': accuracy})
    return pd.DataFrame(accuracies)


def process_all_templates_results(model_name: str, labels_option: str = 'mod', dataloader: str = 'val'):
    """Process all templates results for a given model and labels option."""
    k_values = [1, 3, 5, 7, 9, 11]
    
    # Only support SigLIP2 and OpenCLIP models
    if model_name not in ['SigLIP2', 'OpenCLIP']:
        raise ValueError(f"Model {model_name} not supported. Only SigLIP2 and OpenCLIP are supported for all_templates mode.")
    
    # Construct input filename based on model and labels option
    # The models save CSV files in all_templates structure: eval/results/all_templates/{model}/
    input_filename = f'eval/results/all_templates/{model_name}/{model_name}_classifier_all_templates_{labels_option}.csv'

    # Define output filenames with proper path structure
    output_filename = f'eval/results/all_templates/{model_name}/{model_name}_classifier_all_templates_{labels_option}_processed.csv'
    accuracy_filename = f'eval/results/all_templates/{model_name}/{model_name}_classifier_all_templates_{labels_option}_accuracy.csv'
    
    print(f"Processing {model_name} with {labels_option} labels and {dataloader} dataloader")
    print(f"Input file: {input_filename}")
    print(f"Output files: {output_filename}, {accuracy_filename}")
    
    # Check if input file exists
    if not Path(input_filename).exists():
        print(f"Error: Input file {input_filename} not found!")
        print("Please ensure the model has been run with all_templates mode first.")
        return False
    
    # Ensure output directory exists
    output_dir = Path(output_filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Load and process predictions
        print("Step 1: Loading predictions and computing k-NN results...")
        df = pd.read_csv(input_filename)
        print(f"Loaded {len(df)} predictions")
        
        result_df = compute_knn_predictions(df, k_values)
        result_df.to_csv(output_filename, index=False)
        print(f"K-NN predictions saved to: {output_filename}")
        
        # Step 2: Compute accuracies
        print("Step 2: Computing accuracies...")
        accuracy_df = compute_knn_accuracies(result_df, k_values)
        accuracy_df.to_csv(accuracy_filename, index=False)
        print(f"Accuracies saved to: {accuracy_filename}")
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Model: {model_name}")
        print(f"Labels: {labels_option}")
        print(f"Dataloader: {dataloader}")
        print(f"Total samples: {len(df)}")
        print(f"K values processed: {k_values}")
        print(f"Generated files:")
        print(f"  1. Original predictions: {input_filename}")
        print(f"  2. Processed predictions: {output_filename}")
        print(f"  3. Accuracy results: {accuracy_filename}")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"Error processing results: {e}")
        return False


def main():
    """Main function to handle command line arguments and process results."""
    parser = argparse.ArgumentParser(description='Process all templates results for VLM models (SigLIP2 and OpenCLIP only)')
    parser.add_argument('model_name', type=str, 
                       choices=['SigLIP2', 'OpenCLIP'],
                       help='Name of the model to process (only SigLIP2 and OpenCLIP supported)')
    parser.add_argument('labels_option', type=str, nargs='?', default='mod',
                       choices=['wordnet', 'openai', 'mod'],
                       help='Labels option used (default: mod)')
    parser.add_argument('--dataloader', type=str, default='val',
                       choices=['val', 'train'],
                       help='Dataloader used (default: val)')
    
    args = parser.parse_args()
    
    # Process the results
    success = process_all_templates_results(
        model_name=args.model_name,
        labels_option=args.labels_option,
        dataloader=args.dataloader
    )
    
    if success:
        print("All templates processing completed successfully!")
        sys.exit(0)
    else:
        print("All templates processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
