import pandas as pd
from pathlib import Path
import re
import os
from tqdm import tqdm
from functools import reduce


def extract_number(filename):
    match = re.search(r'(\d+)\.csv$', filename)
    return int(match.group(1)) if match else None


def calculate_accuracy(df, pred_col):
    correct_predictions = (df['original_label'] == df[pred_col]).sum()
    total_predictions = len(df)
    accuracy = 100 * (correct_predictions / total_predictions) if total_predictions > 0 else 0
    return round(accuracy, 2)


def parse_input(base_dir=Path('10/'), split='train'):
    dfs = {}
    count = 0

    for csv_file in base_dir.rglob(f"siglip2-g_kfold*.csv"):
        try:
            fold = extract_number(csv_file.name)
            df = pd.read_csv(csv_file)
            count += df['img_id'].nunique()
            dfs[fold] = df
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    assert count == 1281167
    print(f'Found {len(dfs)} CSV files.')

    return dfs


def calculate_fold_acc(df):
    dfs = parse_input()
    k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]

    accuracy_results = []
    for fold, df in dfs.items():
        for k in k_values:
            acc = calculate_accuracy(df, f'k_{k}_pred')
            accuracy_results.append({'fold': fold, 'k': k, 'accuracy': acc})

    acc_df = pd.DataFrame(accuracy_results)
    pivot_df = acc_df.pivot(index='k', columns='fold', values='accuracy')
    pivot_df.to_csv('10fold_acc.csv')


def calc_prec_conf(df, k):
    res_df = df.groupby('original_label').agg({
         f'k_{k}_conf': 'mean',
         f'k_{k}_conf_all': 'mean'
    }).reset_index()

    res_df = res_df.rename(columns={
         f'k_{k}_conf': 'avg_conf_reduce',
         f'k_{k}_conf_all': 'avg_conf'
    })

    precision_df = df[df['original_label'] == df[f'k_{k}_pred']].groupby(f'k_{k}_pred').size().reset_index(name='tp')
    predicted_df = df.groupby(f'k_{k}_pred').size().reset_index(name='pp')

    precision = pd.merge(predicted_df, precision_df, on=f'k_{k}_pred', how='left')
    precision['tp'] = precision['tp'].fillna(0)
    precision['precision'] = precision['tp'] / precision['pp']

    res_df = res_df.merge(precision[[f'k_{k}_pred', 'precision']],
                                    left_on='original_label', right_on=f'k_{k}_pred',
                                    how='left').drop(columns=f'k_{k}_pred')

    return res_df


def combine_preds():
    vl_pred = pd.read_csv('mult_siglipv2-g_val_mod_mean8.csv')
    v_pred = pd.read_csv('val_siglipv2-g_knn.csv')
    vl_conf_prec = pd.read_csv('vl_siglip2_g_conf_prec.csv')

    vl_conf_dict = dict(zip(vl_conf_prec['original_label'], vl_conf_prec['avg_conf']))
    vl_prec_dict = dict(zip(vl_conf_prec['original_label'], vl_conf_prec['precision']))

    og_label_map = dict(zip(vl_pred['img_id'], vl_pred['original_label']))

    vl_pred_dict = dict(zip(vl_pred['img_id'], vl_pred['top_1_pred']))
    k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]

    for k in k_values:
        print(f'Processing k = {k}')
        v_pred_k = pd.read_csv(f'k_{k}/v_siglip2_g_conf_prec.csv')
        v_pred_label_dict = dict(zip(v_pred['img_id'], v_pred[f'k_{k}_pred']))

        v_conf_dict = dict(zip(v_pred_k['original_label'], v_pred_k['avg_conf']))
        v_conf_reduce_dict = dict(zip(v_pred_k['original_label'], v_pred_k['avg_conf_reduce']))
        v_prec_dict = dict(zip(v_pred_k['original_label'], v_pred_k['precision']))

        res = []
        for img_id in tqdm(vl_pred['img_id'].unique()):
            vl_pred_label = vl_pred_dict[img_id]
            v_pred_label = v_pred_label_dict[img_id]

            vl_conf = vl_conf_dict[vl_pred_label]
            v_conf = v_conf_dict[v_pred_label]
            v_conf_reduce = v_conf_reduce_dict[v_pred_label]

            final_pred_conf = vl_pred_label if vl_conf > v_conf else v_pred_label
            final_pred_conf_reduce = vl_pred_label if vl_conf > v_conf_reduce else v_pred_label

            vl_prec = vl_prec_dict[vl_pred_label]
            v_prec = v_prec_dict[v_pred_label]
            final_pred_prec = vl_pred_label if vl_prec > v_prec else v_pred_label

            res.append({
                'original_label': og_label_map[img_id],
                'img_id': img_id,
                'pred_conf': final_pred_conf,
                'pred_conf_red': final_pred_conf_reduce,
                'pred_prec': final_pred_prec
            })

        output_dir = 'final_pred'
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(res).to_csv(f'{output_dir}/precision_based_pred_k_{k}.csv', index=False)


def calc_average_prec_conf():
    dfs = parse_input()
    for k in [1, 3, 5, 7, 9, 11, 13, 21, 51]:
        print(f'Calculating precision and confidence for k={k}')
        folds = []
        for fold, df in dfs.items():
            print(f'---Processing fold {fold}')
            folds.append(calc_prec_conf(df, k))
        all_data = pd.concat(folds)
        result = all_data.groupby('original_label', as_index=False)[['avg_conf', 'avg_conf_reduce', 'precision']].mean()

        output_dir = f'k_{k}'
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(f'{output_dir}/v_siglip2_g_conf_prec.csv', index=False)


def calc_avg_acc():
    df = pd.read_csv('10fold_acc.csv')
    df['mean'] = df.iloc[:, 1:].mean(axis=1)
    print(df)


if __name__ == "__main__":
    #combine_preds()
    calc_avg_acc()
