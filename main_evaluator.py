import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.evaluator import TextEvaluator

TIMESTAMP_RUN = "20260827182159" 
TARGET_LOG_FOLDER = f"data/augmented/{TIMESTAMP_RUN}/augmented_log"

def main():
    evaluator = TextEvaluator()
    
    output_dir = f"results/evaluation/{TIMESTAMP_RUN}-evaluation"
    if os.path.exists(output_dir):

        time_suffix = datetime.now().strftime("%H%M%S")
        output_dir = f"{output_dir}_{time_suffix}"

    print(f"Read file at {TARGET_LOG_FOLDER}...")
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

            print(f"-> Scoring: {file_name} (Total: {len(df_synthetic)} baris)")
            df_scored, avg_scores = evaluator.evaluate_dataframe(df_synthetic, aligned_originals)
            
            for k in avg_scores.keys():
                avg_scores[k] = round(avg_scores[k], 4)
                
            metrics_cols = ['Cosine_Sim', 'BERT_Score', 'METEOR', 'ROUGE_L', 'BLEU']
            for col in metrics_cols:
                df_scored[col] = df_scored[col].round(4)
            
            evaluator.save_evaluation_results(df_scored, avg_scores, llm_name, technique_name, "TextQuality", output_dir)

    print(f"\nSelesai! Hasil evaluasi kualitas teks tersimpan di: {output_dir}")

if __name__ == "__main__":
    main()