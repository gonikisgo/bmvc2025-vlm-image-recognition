import pandas as pd
from pathlib import Path
import re


k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]


def extract_number(filename):
    match = re.search(r'(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None


def read_all_csvs(base_dir=Path('.')):
    dfs = {}
    csv_count = 0

    for csv_file in base_dir.rglob(f"precision_based*.csv"):
        try:
            csv_count += 1
            df = pd.read_csv(csv_file)
            k = extract_number(csv_file.name)
            dfs[k] = df

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    assert csv_count == len(k_values)
    print(f'Found {csv_count} CSV files.')

    return dfs


def calculate_accuracy(df, pred_col):
    correct_predictions = (df['original_label'] == df[pred_col]).sum()
    total_predictions = len(df)
    accuracy = 100 * (correct_predictions / total_predictions) if total_predictions > 0 else 0
    return round(accuracy, 2)


if __name__ == "__main__":
    dfs = read_all_csvs()

    cols = ['pred_conf', 'pred_conf_red', 'pred_prec']
    results = {}

    for k, df in dfs.items():
        row = {}
        for col in cols:
            acc = calculate_accuracy(df, col)
            row[col] = acc
        results[k] = row

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.sort_index()
    results_df.to_csv('precision_combination.csv')
