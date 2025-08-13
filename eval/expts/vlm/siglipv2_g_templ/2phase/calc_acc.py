import pandas as pd


def calculate_accuracy(df, pred_col):
    correct_predictions = (df['original_label'] == df[pred_col]).sum()
    total_predictions = len(df)
    accuracy = 100 * (correct_predictions / total_predictions) if total_predictions > 0 else 0
    return round(accuracy, 2)


def process_2phase():
    res_df = pd.read_csv('2phase_acc_out_table.csv')

    accuracy_results = []

    k_values = [1, 3, 5, 7, 9, 11, 13, 21, 51]
    for k in k_values:
        col_name = f'templ_pred_k_{k}'
        acc = calculate_accuracy(res_df, col_name)
        accuracy_results.append({'k': k, 'accuracy': acc})

    acc_df = pd.DataFrame(accuracy_results)
    acc_df.to_csv('precision_results.csv', index=False)


if __name__ == "__main__":
    process_2phase()
