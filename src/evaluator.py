from bert_score import score
from rouge_score import rouge_scorer

class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def evaluate_row_rouge(self, reference, candidate):
        """Menghasilkan skor ROUGE-L untuk satu baris teks."""
        if not candidate: return 0.0
        return self.rouge.score(reference, candidate)['rougeL'].fmeasure

    def evaluate_batch_bertscore(self, references, candidates):
        """
        Menghasilkan skor BERTScore individual dan rata-rata.
        Mengembalikan: (list_skor_per_baris, rata_rata_skor)
        """
        if not references or not candidates: return [], 0.0
        P, R, F1 = score(candidates, references, lang="id", verbose=False)
        f1_scores = F1.tolist()
        average_f1 = F1.mean().item()
        return f1_scores, average_f1