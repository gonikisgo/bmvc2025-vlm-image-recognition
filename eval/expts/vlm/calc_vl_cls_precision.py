import pandas as pd
from pathlib import Path


TRAIN_SET_SIZE = 1281167

def calc_precision(df):
    if 'original_label' not in df.columns or 'top_1_pred' not in df.columns:
        print("Warning: Required columns 'original_label' or 'top_1_pred' not found")
        return pd.DataFrame()

    precision_df = df[df['original_label'] == df['top_1_pred']].groupby('top_1_pred').size().reset_index(name='tp')
    predicted_df = df.groupby('top_1_pred').size().reset_index(name='pp')

    precision = pd.merge(predicted_df, precision_df, on='top_1_pred', how='left')
    precision['tp'] = precision['tp'].fillna(0)
    precision['precision'] = precision['tp'] / precision['pp']

    result = df.groupby('original_label').size().reset_index(name='count')
    result = result.merge(precision[['top_1_pred', 'precision']],
                          left_on='original_label', right_on='top_1_pred',
                          how='left').drop(columns='top_1_pred')

    return result[['original_label', 'precision']]


def calc_cls_precision():
    print('🚀 Starting VL classification precision calculation...')
    
    # Check if the required file exists
    input_file = Path('../../results/vlm/SigLIP2/SigLIP2_classifier_avg_prime_mod_train.csv')
    if not input_file.exists():
        print(f"❌ Error: Required input file not found!")
        print(f"📍 Expected file location: {input_file.absolute()}")
        print(f"💡 Please ensure the SigLIP2 classifier results have been generated first.")
        print(f"   The file should contain columns: 'original_label', 'top_1_pred'")
        return
    
    df = pd.read_csv(input_file)
    assert len(df) == TRAIN_SET_SIZE, f"Expected {TRAIN_SET_SIZE} samples, but got {len(df)}"

    output_dir = Path(f'../../results/vlm/SigLIP2')
    output_dir.mkdir(exist_ok=True)

    precision_df = calc_precision(df)
    output_file = output_dir / 'SigLIP2_avg_prime_mod_train_precision.csv'
    precision_df.to_csv(output_file, index=False)
    print(f'💾 Results saved to: {output_file.absolute()}')

    print('\n🎉 All VL classification precision calculations completed successfully!')


if __name__ == "__main__":
    calc_cls_precision()