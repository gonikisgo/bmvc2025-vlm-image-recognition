import pandas as pd


def calc_prec_conf():
    df = pd.read_csv('mult_siglipv2-g_train_mod_mean8.csv')

    res_df = df.groupby('original_label').agg({
        'top_1_prob': 'mean'
    }).reset_index()

    res_df = res_df.rename(columns={
        'top_1_prob': 'avg_conf'
    })

    precision_df = df[df['original_label'] == df['top_1_pred']].groupby('top_1_pred').size().reset_index(name='tp')
    predicted_df = df.groupby('top_1_pred').size().reset_index(name='pp')

    precision = pd.merge(predicted_df, precision_df, on='top_1_pred', how='left')
    precision['tp'] = precision['tp'].fillna(0)
    precision['precision'] = precision['tp'] / precision['pp']

    res_df = res_df.merge(precision[['top_1_pred', 'precision']],
                                    left_on='original_label', right_on='top_1_pred',
                                    how='left').drop(columns='top_1_pred')

    res_df.to_csv('vl_siglip2_g_conf_prec.csv', index=False)


if __name__ == "__main__":
    calc_prec_conf()
