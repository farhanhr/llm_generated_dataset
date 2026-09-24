import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.classifier import FakeNewsClassifier

TIMESTAMP_RUN = "20260913182453"  
TARGET_FOLDER = f"data/augmented/{TIMESTAMP_RUN}/merged/balanced"

TRAIN_BASELINE_PATH = "data/raw/train_data.csv"
TEST_DATA_PATH = "data/raw/test_data.csv"

def main():
    classifier = FakeNewsClassifier()
    
    output_dir = f"results/classification/{TIMESTAMP_RUN}-classification"
    if os.path.exists(output_dir):
        time_suffix = datetime.now().strftime("%H%M%S")
        output_dir = f"{output_dir}_{time_suffix}"

    print("Memuat data uji referensi...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df = test_df.dropna(subset=['Pesan', 'Kategori'])

    print("\nTraining with baseline (raw train) data...")
    train_baseline_df = pd.read_csv(TRAIN_BASELINE_PATH)
    train_baseline_df = train_baseline_df.dropna(subset=['Pesan', 'Kategori'])

    baseline_metrics = classifier.train_and_evaluate(train_baseline_df, test_df)
    classifier.save_classification_results(baseline_metrics, "Baseline_Data_Asli", output_dir)
    print(f"-> Selesai! (Akurasi: {baseline_metrics['Akurasi']:.4f} | F1: {baseline_metrics['F1-Score']:.4f})")

    print(f"\nTraining for augmented files in: {TARGET_FOLDER}...")
    if os.path.exists(TARGET_FOLDER):
        for file_name in os.listdir(TARGET_FOLDER):
            if file_name.endswith('.csv'):
                file_path = os.path.join(TARGET_FOLDER, file_name)
                
                df_merged_train = pd.read_csv(file_path)
                df_merged_train = df_merged_train.dropna(subset=['Pesan', 'Kategori'])

                print(f"-> Melatih XGBoost pada: {file_name}")

                metrics = classifier.train_and_evaluate(df_merged_train, test_df)

                dataset_name = file_name.replace('.csv', '')
                classifier.save_classification_results(metrics, dataset_name, output_dir)
    else:
        print(f"Folder {TARGET_FOLDER} tidak ditemukan. Pastikan proses augmentasi sudah selesai dijalankan.")

    print(f"\nResult saved at: {output_dir}")

if __name__ == "__main__":
    main()