import os
import csv
import pandas as pd
from bert_score import score as bert_score_func
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer, util

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        self.smoother = SmoothingFunction().method1
        self.st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def get_rouge_score(self, reference, candidate):
        if not candidate or not isinstance(candidate, str): return 0.0
        return self.rouge.score(str(reference), str(candidate))['rougeL'].fmeasure

    def get_bleu_score(self, reference, candidate):
        if not candidate or not isinstance(candidate, str): return 0.0
        ref_tokens = word_tokenize(str(reference).lower())
        cand_tokens = word_tokenize(str(candidate).lower())
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smoother)
        
    def get_bertscore_batch(self, references, candidates):
        if not references or not candidates: return []
        refs = [str(r) if r else "" for r in references]
        cands = [str(c) if c else "" for c in candidates]
        P, R, F1 = bert_score_func(cands, refs, lang="id", verbose=False)
        return F1.tolist()

    def get_cosine_sim_batch(self, references, candidates):
        if not references or not candidates: return []
        refs = [str(r) if r else "" for r in references]
        cands = [str(c) if c else "" for c in candidates]
        
        ref_embeddings = self.st_model.encode(refs, convert_to_tensor=True)
        cand_embeddings = self.st_model.encode(cands, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(ref_embeddings, cand_embeddings)
        diagonal_scores = [cosine_scores[i][i].item() for i in range(len(refs))]
        return diagonal_scores

    def evaluate_dataframe(self, df_synthetic, original_texts):
        df = df_synthetic.copy()
        candidates = df['Pesan'].tolist()
        
        df['BERT_Score'] = self.get_bertscore_batch(original_texts, candidates)
        # df['Cosine_Sim'] = self.get_cosine_sim_batch(original_texts, candidates)
        
        df['ROUGE_L'] = [self.get_rouge_score(orig, cand) for orig, cand in zip(original_texts, candidates)]
        df['BLEU'] = [self.get_bleu_score(orig, cand) for orig, cand in zip(original_texts, candidates)]
        
        avg_scores = {
            'Avg_BERT_Score': df['BERT_Score'].mean(),
            # 'Avg_Cosine_Sim': df['Cosine_Sim'].mean(),
            'Avg_ROUGE_L': df['ROUGE_L'].mean(),
            'Avg_BLEU': df['BLEU'].mean()
        }
        return df, avg_scores

    def save_evaluation_results(self, df_scored, avg_scores, model, technique, test_name, timestamp_dir):
        os.makedirs(timestamp_dir, exist_ok=True)
        file_name = f"{model}_{technique}_{test_name}.csv"
        file_path = os.path.join(timestamp_dir, file_name)
        
        cols_order = ['Kategori', 'Pesan', 'BERT_Score', 'ROUGE_L', 'BLEU']
        df_scored[cols_order].to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        summary_path = os.path.join(timestamp_dir, "Summary_TextQuality.csv")
        summary_data = pd.DataFrame([{
            'Nama_file': file_name,
            'Model': model,
            'Teknik': technique,
            'BERT_score': avg_scores['Avg_BERT_Score'],
            # 'Cosine_Sim': avg_scores['Avg_Cosine_Sim'],
            'ROUGE_score': avg_scores['Avg_ROUGE_L'],
            'BLEU': avg_scores['Avg_BLEU']
        }])
        
        if os.path.exists(summary_path):
            summary_data.to_csv(summary_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            summary_data.to_csv(summary_path, mode='w', header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)