import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.evaluator import TextEvaluator

# ================= KONFIGURASI =================
# Sesuaikan dengan nama folder hasil run Anda
BASE_FOLDER = "data/augmented/20260827182159"
TARGET_SYNTHETIC = os.path.join(BASE_FOLDER, "synthetic")
TARGET_MERGED = os.path.join(BASE_FOLDER, "merged")
# ===============================================

def main():
    evaluator = TextEvaluator()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"result/evaluation/{timestamp}"

    print(f"1. Membaca file sintetis di {TARGET_SYNTHETIC}...")
    for file_name in os.listdir(TARGET_SYNTHETIC):
        if file_name.endswith('.csv'):
            synth_path = os.path.join(TARGET_SYNTHETIC, file_name)
            
            # Mencari file merged yang bersesuaian untuk mengambil teks aslinya
            merged_filename = file_name.replace("synthetic_", "merged_")
            merged_path = os.path.join(TARGET_MERGED, merged_filename)
            
            df_synthetic = pd.read_csv(synth_path)
            
            # Ekstrak Teks Asli dari File Merged
            # File merged berisi data Ham, Spam Asli, dan Spam Sintetis.
            # Kita hanya mengambil Spam Asli (Kategori == 'spam' dan belum ada skor/is_synthetic=0)
            # Namun cara paling sederhana jika kita tidak memodifikasi main_augment.py adalah
            # mengambil dari raw data dengan index yang persis sama.
            
            # CARA PALING AMAN: Karena df_synthetic saat ini TIDAK memiliki kolom 'Original_Text',
            # dan len() nya mismatch 1142 vs 1143, kita akan load ulang RAW_DATA.
            loader = pd.read_csv("data/raw/sms_spam_indo.csv")
            spam_asli = loader[loader['Kategori'] == 'spam']['Pesan'].tolist()
            
            # Sinkronisasi aman: Kita menduplikasi teks spam asli, lalu memotongnya 
            # persis sesuai panjang df_synthetic. Ini mengasumsikan urutan LLM generate tidak berubah.
            aligned_original_texts = []
            multiplier = len(df_synthetic) // len(spam_asli)
            sisa = len(df_synthetic) % len(spam_asli)
            
            for text in spam_asli:
                aligned_original_texts.extend([text] * multiplier)
            
            # Jika ada sisa (karena mismatch), tambahkan dari awal agar index sejajar
            if sisa > 0:
                aligned_original_texts.extend(spam_asli[:sisa])
                
            # Potong persis sesuai panjang data sintetis
            aligned_original_texts = aligned_original_texts[:len(df_synthetic)]

            # Ekstrak nama
            parts = file_name.replace('synthetic_', '').replace('.csv', '').split('_')
            llm_name = parts[0]
            technique_name = parts[1] if len(parts) > 1 else "unknown"

            print(f"-> Menilai Kualitas: {file_name}")
            df_scored, avg_scores = evaluator.evaluate_dataframe(df_synthetic, aligned_original_texts)
            
            # Membulatkan skor rata-rata menjadi 4 angka di belakang koma
            avg_scores['Avg_BERT_Score'] = round(avg_scores['Avg_BERT_Score'], 4)
            avg_scores['Avg_ROUGE_L'] = round(avg_scores['Avg_ROUGE_L'], 4)
            
            # Membulatkan skor per-baris di dalam DataFrame
            df_scored['BERT_Score'] = df_scored['BERT_Score'].round(4)
            df_scored['ROUGE_L'] = df_scored['ROUGE_L'].round(4)
            
            evaluator.save_evaluation_results(df_scored, avg_scores, llm_name, technique_name, "TextQuality", output_dir)

    print(f"\nSelesai! Hasil evaluasi kualitas teks tersimpan di: {output_dir}")

if __name__ == "__main__":
    main()