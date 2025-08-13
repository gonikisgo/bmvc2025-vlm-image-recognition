import pandas as pd
from pathlib import Path
import re


k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]


def calculate_accuracy(df, pred_col):
    correct_predictions = (df['original_label'] == df[pred_col]).sum()
    total_predictions = len(df)
    accuracy = 100 * (correct_predictions / total_predictions) if total_predictions > 0 else 0
    return round(accuracy, 2)


if __name__ == "__main__":
    df = pd.read_csv('radio_378_val_knn.csv')

    results = {}
    for k in k_values:
        col = f'k_{k}_pred'
        acc = calculate_accuracy(df, col)
        results[k] = acc

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_index()
    results_df.to_csv('radio-378-val-knn-acc.csv')
