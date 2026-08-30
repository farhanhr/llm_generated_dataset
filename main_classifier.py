import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import SMSDataLoader
from src.classifier import FakeNewsClassifier

TIMESTAMP_RUN = "20260827182159"
TARGET_FOLDER = f"data/augmented/{TIMESTAMP_RUN}/merged"
RAW_DATA_PATH = "data/raw/sms_spam_indo.csv"

def main():
    classifier = FakeNewsClassifier()
    
    output_dir = f"results/classification/{TIMESTAMP_RUN}-classification"
    if os.path.exists(output_dir):
        time_suffix = datetime.now().strftime("%H%M%S")
        output_dir = f"{output_dir}_{time_suffix}"

    print("Training with raw data")
    loader = SMSDataLoader(RAW_DATA_PATH)
    normal_df, spam_df = loader.process()
    baseline_df = pd.concat([normal_df, spam_df], ignore_index=True)

    baseline_metrics = classifier.train_and_evaluate(baseline_df)
    classifier.save_classification_results(baseline_metrics, "Baseline_Data_Asli", output_dir)
    print(f"-> Selesai! (Akurasi: {baseline_metrics['Akurasi']:.4f} | F1: {baseline_metrics['F1-Score']:.4f})")

    print(f"\nTraining for: {TARGET_FOLDER}...")
    for file_name in os.listdir(TARGET_FOLDER):
        if file_name.endswith('.csv'):
            file_path = os.path.join(TARGET_FOLDER, file_name)
            df_merged = pd.read_csv(file_path)
            df_merged = df_merged.dropna(subset=['Pesan', 'Kategori'])

            print(f"-> Melatih XGBoost pada: {file_name}")
            metrics = classifier.train_and_evaluate(df_merged)

            dataset_name = file_name.replace('.csv', '')
            classifier.save_classification_results(metrics, dataset_name, output_dir)

    print(f"\nResult saved at: {output_dir}")

if __name__ == "__main__":
    main()