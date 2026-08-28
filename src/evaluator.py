from bert_score import score
from rouge_score import rouge_scorer

class TextEvaluator:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    def evaluate_bertscore(self, references, candidates):
        P, R, F1 = score(candidates, references, lang="id", verbose=False)
        return F1.mean().item()

    def evaluate_rouge(self, reference, candidate):
        scores = self.rouge.score(reference, candidate)
        return scores['rougeL'].fmeasure