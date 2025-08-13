import pandas as pd
from pathlib import Path


dirs = ['clip', 'openclip', 'radio_378', 'radio_896', 'siglip', 'siglipv2', 'siglipv2_g']
#dirs = ['siglipv2_g_templ']

def calculate_accuracy(df):
    correct_predictions = (df['original_label'] == df['top_1_pred']).sum()
    total_predictions = len(df)
    accuracy = 100 * (correct_predictions / total_predictions) if total_predictions > 0 else 0
    return round(accuracy, 2)


def process_all_csvs(base_dir):
    results = []

    for dir_name in dirs:
        csv_count = 0
        path = base_dir / dir_name
        if not path.exists():
            print(f"Warning: {path} does not exist.")
            continue

        for csv_file in path.rglob("*.csv"):
            try:
                csv_count += 1
                df = pd.read_csv(csv_file)
                accuracy = calculate_accuracy(df)
                print(f"Accuracy for {csv_file}: {accuracy}")
                subfolder = csv_file.parent.name
                filename = csv_file.name
                results.append([subfolder, filename, accuracy])
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

        print(f'Found {csv_count} CSV files in {path}.')
        #assert csv_count % 5 == 0, f"Expected % 5 CSV files in {path}, but found {csv_count}."

    return pd.DataFrame(results, columns=['Folder', 'Filename', 'Accuracy'])


if __name__ == "__main__":
    base_directory = Path('.')
    output_table = process_all_csvs(base_directory)
    output_table.to_csv('vlm_result.csv', index=False)
