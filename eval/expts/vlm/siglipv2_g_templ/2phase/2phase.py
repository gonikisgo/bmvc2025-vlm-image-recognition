import pandas as pd
from pathlib import Path
import re
import pickle
from functools import reduce


def extract_number(filename):
    match = re.search(r'(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None


def read_all_csvs(base_dir=Path('..'), split='train'):
    dfs = {}
    csv_count = 0

    for csv_file in base_dir.rglob(f"mult_siglipv2-g_{split}_templ_mod*.csv"):
        try:
            csv_count += 1
            df = pd.read_csv(csv_file)
            if split == 'train':
                assert len(df) == 1281167
            else:
                assert len(df) == 50000
            templ_id = extract_number(csv_file.name)
            dfs[templ_id] = df

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f'Found {csv_count} CSV files.')

    return dfs


def calculate_class_acc(df):
    df['is_correct'] = df['original_label'] == df['top_1_pred']

    accuracy_df = df.groupby('original_label').agg({
        'is_correct': 'mean',
        'top_1_prob': 'mean'
    }).reset_index()

    accuracy_df = accuracy_df.rename(columns={
        'is_correct': 'accuracy',
        'top_1_prob': 'avg_conf'
    })

    precision_df = df[df['original_label'] == df['top_1_pred']].groupby('top_1_pred').size().reset_index(name='tp')
    predicted_df = df.groupby('top_1_pred').size().reset_index(name='pp')

    precision = pd.merge(predicted_df, precision_df, on='top_1_pred', how='left')
    precision['tp'] = precision['tp'].fillna(0)
    precision['precision'] = precision['tp'] / precision['pp']

    accuracy_df = accuracy_df.merge(precision[['top_1_pred', 'precision']],
                                    left_on='original_label', right_on='top_1_pred',
                                    how='left').drop(columns='top_1_pred')

    accuracy_df['accuracy'] = accuracy_df['accuracy'].fillna(0)
    accuracy_df['avg_conf'] = accuracy_df['avg_conf'].fillna(0)
    accuracy_df['precision'] = accuracy_df['precision'].fillna(0)

    return accuracy_df


def prepare_data():
    dfs = read_all_csvs()

    res = {}
    for idx, df in dfs.items():
        res[idx] = calculate_class_acc(df)

    with open('2phase_train.pkl', 'wb') as f:
        pickle.dump(res, f)


def calc_2phase_pred():
    templ2pred = pd.read_csv('2phase/templ2prediction.csv')
    knn_pred = pd.read_csv('2phase/val_siglipv2-g_knn.csv')
    label2id = pd.read_csv('2phase/lookup_precision.csv')

    k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]
    for k in k_values:
        print(f'Processing k = {k}')
        knn_pred[f'best_templ_id_k_{k}'] = knn_pred[f'k_{k}_pred'].apply(
            lambda x: label2id[label2id['label'] == x]['templ_id'].values[0]
        )

        def get_templ_pred(row):
            templ_id = row[f'best_templ_id_k_{k}']
            img_id = row['img_id']
            return templ2pred[templ2pred['img_id'] == img_id][f'templ_{templ_id}'].values[0]

        knn_pred[f'templ_pred_k_{k}'] = knn_pred.apply(get_templ_pred, axis=1)

    keep_cols = ['original_label', 'img_id'] + [f'templ_pred_k_{k}' for k in k_values]
    knn_pred = knn_pred[keep_cols]

    knn_pred.to_csv('2phase/2phase_precision_out.csv', index=False)


def build_lookup():
    match_keys = ['accuracy', 'avg_conf', 'precision']

    with open('2phase_train.pkl', 'rb') as f:
        id_to_df = pickle.load(f)

    for match_key in match_keys:
        print(f'Building lookup for {match_key}')
        label2df_id = get_best_ids(id_to_df, match_key)
        lookup_df = pd.DataFrame(list(label2df_id.items()), columns=['label', 'templ_id'])
        lookup_df.to_csv(f'2phase/lookup_{match_key}.csv', index=False)


def get_best_ids(id_to_df, match_key):
    best_ids = {}
    for df_id, df in id_to_df.items():
        for _, row in df.iterrows():
            label = row['original_label']
            score = row[match_key]
            if label not in best_ids or score > best_ids[label][1]:
                best_ids[label] = (df_id, score)
    return {int(label): int(df_id) for label, (df_id, _) in best_ids.items()}


def prepare_templ2prediction():
    dfs = read_all_csvs(split='val')

    renamed_dfs = [
        df[['img_id', 'top_1_pred']].rename(columns={'top_1_pred': f'templ_{df_id}'})
        for df_id, df in dfs.items()
    ]
    merged_df = reduce(lambda left, right: left.merge(right, on='img_id'), renamed_dfs)
    merged_df.to_csv('2phase/templ2prediction.csv', index=False)


if __name__ == "__main__":
    #prepare_data()
    #build_lookup()
    #prepare_templ2prediction()
    calc_2phase_pred()
