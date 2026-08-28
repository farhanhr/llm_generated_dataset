import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
import csv
import os

class FakeNewsClassifier:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

    def train_and_evaluate(self, df):
        """Melatih XGBoost dan mengembalikan metrik evaluasi klasifikasi (Dibulatkan 4 desimal)."""
        X = self.vectorizer.fit_transform(df['Pesan']).toarray()
        y = df['Kategori'].map({'ham': 0, 'spam': 1}).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'Akurasi': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred), 4),
            'Recall': round(recall_score(y_test, y_pred), 4),
            'F1-Score': round(f1_score(y_test, y_pred), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_prob), 4)
        }
        return metrics

    def save_classification_results(self, metrics, dataset_name, timestamp_dir):
        os.makedirs(timestamp_dir, exist_ok=True)
        summary_path = os.path.join(timestamp_dir, "Summary_Classification.csv")
        
        metrics['Dataset'] = dataset_name
        df_metrics = pd.DataFrame([metrics])
        
        cols = ['Dataset', 'Akurasi', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        df_metrics = df_metrics[cols]
        
        if os.path.exists(summary_path):
            df_metrics.to_csv(summary_path, mode='a', header=False, index=False, quoting=csv.QUOTE_NONNUMERIC)
        else:
            df_metrics.to_csv(summary_path, mode='w', header=True, index=False, quoting=csv.QUOTE_NONNUMERIC)