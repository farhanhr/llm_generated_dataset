import pandas as pd
from bert_score import score
from rouge_score import rouge_scorer
import csv

class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def get_rouge_score(self, reference, candidate):
        """Skor ROUGE-L untuk satu baris teks."""
        if not candidate or not isinstance(candidate, str): return 0.0
        return self.rouge.score(str(reference), str(candidate))['rougeL'].fmeasure

    def get_bertscore_batch(self, references, candidates):
        """Mendapatkan list skor BERTScore individual untuk semua baris."""
        if not references or not candidates: return []
        refs = [str(r) if r else "" for r in references]
        cands = [str(c) if c else "" for c in candidates]
        P, R, F1 = score(cands, refs, lang="id", verbose=False)
        return F1.tolist()

    def evaluate_dataframe(self, df_synthetic, original_texts):
        """
        Fungsi fleksibel untuk menilai seluruh DataFrame sintetis.
        Mengembalikan: (DataFrame dengan kolom skor, Dictionary rata-rata skor)
        """
        df = df_synthetic.copy()
        
        df['ROUGE_L'] = [self.get_rouge_score(orig, cand) for orig, cand in zip(original_texts, df['Pesan'])]
        
        df['BERT_Score'] = self.get_bertscore_batch(original_texts, df['Pesan'].tolist())
        
        avg_scores = {
            'Avg_BERT_Score': df['BERT_Score'].mean(),
            'Avg_ROUGE_L': df['ROUGE_L'].mean()
        }
        
        return df, avg_scores

    def save_evaluation_results(self, df_scored, avg_scores, model, technique, test_name, timestamp_dir):
        """Menyimpan format CSV Per-Baris dan menambahkan ke Log Summary Rata-rata."""
        import os
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # Format 1: Skor Per Baris
        # Nama file contoh: LLaMA_zero-shot_TextQuality.csv
        file_name = f"{model}_{technique}_{test_name}.csv"
        file_path = os.path.join(timestamp_dir, file_name)
        
        df_scored[['Kategori', 'Pesan', 'BERT_Score', 'ROUGE_L']].to_csv(
            file_path, index=False, quoting=csv.QUOTE_NONNUMERIC
        )
        
        # Format 2: Menambahkan ke file Summary Rata-rata
        summary_path = os.path.join(timestamp_dir, "Summary_TextQuality.csv")
        summary_data = pd.DataFrame([{
            'Nama_file': file_name,
            'Model': model,
            'Teknik': technique,
            'Test': test_name,
            'BERT_score': avg_scores['Avg_BERT_Score'],
            'ROUGE_score': avg_scores['Avg_ROUGE_L']
        }])
        
        if os.path.exists(summary_path):
            summary_data.to_csv(summary_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            summary_data.to_csv(summary_path, mode='w', header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)