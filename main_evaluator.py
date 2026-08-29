import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.evaluator import TextEvaluator

TIMESTAMP_RUN = "20260828191155" 
TARGET_LOG_FOLDER = f"data/augmented/{TIMESTAMP_RUN}/augmented_log"


def main():
    evaluator = TextEvaluator()
    timestamp_eval = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"results/evaluation/{timestamp_eval}" 

    print(f"1. Membaca file log CSV di {TARGET_LOG_FOLDER}...")
    for file_name in os.listdir(TARGET_LOG_FOLDER):
        if file_name.endswith('.csv'):
            file_path = os.path.join(TARGET_LOG_FOLDER, file_name)
            df_log = pd.read_csv(file_path)
            
            parts = file_name.replace('log_', '').replace('.csv', '').split('_')

            technique_name = parts[-1]
            llm_name = "_".join(parts[:-1])

            aligned_originals = []
            synthetic_pesan = []
            parafrase_cols = [col for col in df_log.columns if col.startswith('parafrase_')]
            
            for _, row in df_log.iterrows():
                orig_text = row['original']
                for col in parafrase_cols:
                    syn_text = row[col]
                    if pd.notna(syn_text) and str(syn_text).strip() != "":
                        aligned_originals.append(orig_text)
                        synthetic_pesan.append(syn_text)

            df_synthetic = pd.DataFrame({'Kategori': ['spam'] * len(synthetic_pesan), 'Pesan': synthetic_pesan})

            print(f"-> Menilai Kualitas: {file_name} (Total: {len(df_synthetic)} baris)")
            df_scored, avg_scores = evaluator.evaluate_dataframe(df_synthetic, aligned_originals)
            
            avg_scores['Avg_BERT_Score'] = round(avg_scores['Avg_BERT_Score'], 4)
            avg_scores['Avg_ROUGE_L'] = round(avg_scores['Avg_ROUGE_L'], 4)
            df_scored['BERT_Score'] = df_scored['BERT_Score'].round(4)
            df_scored['ROUGE_L'] = df_scored['ROUGE_L'].round(4)
            
            evaluator.save_evaluation_results(df_scored, avg_scores, llm_name, technique_name, "TextQuality", output_dir)

    print(f"\nSelesai! Hasil evaluasi kualitas teks tersimpan di: {output_dir}")

if __name__ == "__main__":
    main()