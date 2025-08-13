import math

import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


k_values = [1, 3, 5, 7, 9, 11, 13, 21, 49, 51]


def process_all_csvs(base_dir, dir_name):
    path = base_dir / dir_name
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} not found.")

    try:
        k_limit = int(dir_name)
    except ValueError:
        print(f"Directory name '{dir_name}' is not an integer.")
        return pd.DataFrame()

    valid_k = [k for k in k_values if k <= k_limit]

    label_k_data = defaultdict(lambda: defaultdict(list))

    csv_count = 0
    for csv_file in tqdm(path.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            csv_count += 1

            for k in valid_k:
                pred_col = f"k_{k}_pred"
                conf_col = f"k_{k}_conf"
                if pred_col not in df.columns or conf_col not in df.columns:
                    raise ValueError(f"Columns {pred_col} or {conf_col} not found in {csv_file}")

                assert len(df["original_label"].unique()) == 1000
                for label in df["original_label"].unique():
                    df_label = df[df["original_label"] == label]
                    acc = 100 * ((df_label[pred_col] == df_label["original_label"]).mean())
                    avg_conf = df_label[conf_col].mean()

                    label_k_data[label][f"k_{k}_acc"].append(acc)
                    label_k_data[label][f"k_{k}_conf"].append(avg_conf)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f'Found {csv_count} CSV files in {path}.')

    results = []
    for label, metrics in label_k_data.items():
        row = {"label": label}
        for metric, values in metrics.items():
            row[metric] = sum(values) / len(values) if values else None
        results.append(row)

    return pd.DataFrame(results)


def main1():
    base_directory = Path('.')
    res_df = process_all_csvs(base_directory, '10')
    res_df.to_csv('res/10_shot_cls_acc.csv', index=False)


def process_all_csvs_ci(base_dir, dir_name):
    path = Path(base_dir) / dir_name
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} not found.")

    try:
        k_limit = int(dir_name)
    except ValueError:
        print(f"Directory name '{dir_name}' is not an integer.")
        return pd.DataFrame()

    valid_k = [k for k in k_values if k <= k_limit]
    k_accuracies = defaultdict(list)
    csv_count = 0

    for csv_file in tqdm(path.rglob("*.csv"), desc="Processing CSVs"):
        try:
            df = pd.read_csv(csv_file)
            assert len(df["original_label"].unique()) == 1000
            assert len(df) == 50000
            csv_count += 1

            if "original_label" not in df.columns:
                raise ValueError(f"Missing 'original_label' column in {csv_file}")

            for k in valid_k:
                pred_col = f"k_{k}_pred"
                if pred_col not in df.columns:
                    raise ValueError(f"Column {pred_col} not found in {csv_file}")

                acc = 100 * (df[pred_col] == df["original_label"]).mean()
                k_accuracies[k].append(acc)

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f'Found {csv_count} CSV files in {path}.')

    # Prepare results with 95% confidence interval
    results = []
    for k, accs in k_accuracies.items():
        n = len(accs)
        print(f"Number of accuracies for k={k}: {n}")
        if n == 0:
            continue
        mean_acc = sum(accs) / n
        std_dev = pd.Series(accs).std(ddof=1)
        ci_95 = 1.96 * (std_dev / math.sqrt(n)) if n > 1 else 0.0

        results.append({
            "k": k,
            "average_accuracy": round(mean_acc, 2),
            "deviation": round(ci_95, 4)
        })

    return pd.DataFrame(results)

def main2():
    base_directory = Path('.')
    res_df = process_all_csvs_ci(base_directory, '20')
    res_df.to_csv('res/20_shot_acc_ci.csv', index=False)

if __name__ == "__main__":
    main2()
