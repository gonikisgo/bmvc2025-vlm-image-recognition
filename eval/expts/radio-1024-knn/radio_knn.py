import pandas as pd
from collections import Counter
from typing import List


def majority_vote(labels: List[str]) -> str:
    return Counter(labels).most_common(1)[0][0]

def compute_knn_predictions(df: pd.DataFrame, k_values: List[int]) -> pd.DataFrame:
    top_label_cols = [f'top_{i}_label' for i in range(1, 52)]

    missing_cols = [col for col in top_label_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    required_cols = ['original_label', 'img_id']
    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    result_df = df[['original_label', 'img_id']].copy()

    for k in k_values:
        current_cols = top_label_cols[:k]
        print(len(current_cols))
        result_df[f'k_{k}_pred'] = df[current_cols].apply(lambda row: majority_vote(row.values.tolist()), axis=1)
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


k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]

def main1():
    df = pd.read_csv('radio-1024_nn_train.csv')
    result_df = compute_knn_predictions(df, k_values)
    result_df.to_csv('train_radio_knn.csv', index=False)


def main2():
    result_df = pd.read_csv('train_radio_knn.csv')
    accuracy_df = compute_knn_accuracies(result_df, k_values)
    accuracy_df.to_csv('train_radio_knn_acc.csv', index=False)


if __name__ == "__main__":
    main2()