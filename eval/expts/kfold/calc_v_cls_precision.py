import pandas as pd
from pathlib import Path
import re

TRAIN_SET_SIZE = 1281167
FOLDS_NUM = 10


def extract_number(filename):
    match = re.search(r'(\d+)(?=\.csv$)', filename)
    return int(match.group(1)) if match else None


def parse_input(base_dir=Path(f'../../results/kfold/{FOLDS_NUM}fold_train')):
    dfs = {}
    count = 0
    for csv_file in base_dir.rglob(f"SigLIP2_10fold_*[0-9].csv"):
        try:
            fold = extract_number(csv_file.name)
            df = pd.read_csv(csv_file)
            count += df['img_id'].nunique()
            dfs[fold] = df
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    assert count == TRAIN_SET_SIZE
    print(f'📂 Successfully loaded {len(dfs)} fold CSV files with {count:,} total samples')
    return dfs


def calc_precision(df, k):
    pred_col = f'k_{k}_pred'
    if 'original_label' not in df.columns or pred_col not in df.columns:
        print(f"Warning: Required columns 'original_label' or {pred_col} not found")
        return pd.DataFrame()

    precision_df = df[df['original_label'] == df[pred_col]].groupby(pred_col).size().reset_index(name='tp')
    predicted_df = df.groupby(pred_col).size().reset_index(name='pp')

    precision = pd.merge(predicted_df, precision_df, on=pred_col, how='left')
    precision['tp'] = precision['tp'].fillna(0)
    precision['precision'] = precision['tp'] / precision['pp']

    result = df.groupby('original_label').size().reset_index(name='count')
    result = result.merge(precision[[pred_col, 'precision']],
                          left_on='original_label', right_on=pred_col,
                          how='left').drop(columns=pred_col)

    return result[['original_label', 'precision']]


def calc_accuracy(df, k):
    pred_col = f'k_{k}_pred'
    if 'original_label' not in df.columns or pred_col not in df.columns:
        print(f"Warning: Required columns 'original_label' or {pred_col} not found")
        return 0.0

    correct_predictions = (df['original_label'] == df[pred_col]).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return accuracy


def calc_average_precision():
    print('🚀 Starting precision calculation across all k-values and folds...')
    dfs = parse_input()
    assert len(dfs) == FOLDS_NUM
    output_dir = Path(f'../../results/kfold/{FOLDS_NUM}fold_train')
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f'📁 Output directory created: {output_dir.absolute()}')

    total_k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]
    print(f'📊 Processing {len(total_k_values)} different k-values: {total_k_values}')

    # Initialize list to store accuracy results
    accuracy_results = []

    for i, k in enumerate(total_k_values, 1):
        print(f'\n🔄 [{i}/{len(total_k_values)}] Calculating precision for k={k}...')
        folds = []
        fold_accuracies = []

        for fold_idx, (fold, df) in enumerate(dfs.items(), 1):
            print(f'   ⚙️  Processing fold {fold} ({fold_idx}/{FOLDS_NUM})')
            cls_fold_precision = calc_precision(df, k)
            assert len(cls_fold_precision) == 1000
            folds.append(cls_fold_precision)
            fold_accuracies.append(calc_accuracy(df, k))

        print(f'   📈 Aggregating results across {FOLDS_NUM} folds...')
        all_data = pd.concat(folds)
        result = all_data.groupby('original_label', as_index=False)[['precision']].mean()
        output_file = output_dir / f'SigLIP2_{k}nn_{FOLDS_NUM}fold_precision.csv'
        result.to_csv(output_file, index=False)
        print(f'   ✅ Saved results to: {output_file.name}')

        # Calculate average accuracy across folds
        avg_accuracy = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0.0
        accuracy_results.append({'k': k, 'average_accuracy': avg_accuracy})
        print(f'   📊 Average accuracy for k={k}: {avg_accuracy:.4f}')

    # Save accuracy results to CSV in eval/results/kfold/{k}fold folder
    accuracy_df = pd.DataFrame(accuracy_results)
    accuracy_output_file = output_dir / f'SigLIP2_{FOLDS_NUM}fold_average_accuracy.csv'
    accuracy_df.to_csv(accuracy_output_file, index=False)
    print(f'\n📈 Average accuracy results saved to: {accuracy_output_file}')

    # Find and print the best k value
    best_result = max(accuracy_results, key=lambda x: x['average_accuracy'])
    best_k = best_result['k']
    best_accuracy = best_result['average_accuracy']
    print(f'\n🏆 Best accuracy achieved with k={best_k}: {best_accuracy:.4f}')

    print('\n🎉 All precision calculations completed successfully!')


if __name__ == "__main__":
    calc_average_precision()