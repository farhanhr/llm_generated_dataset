import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
import csv
import os

class FakeNewsClassifier:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = XGBClassifier(eval_metric='logloss', random_state=42)

    def train_and_evaluate(self, train_df, test_df):
        """Melatih XGBoost dan mengembalikan metrik evaluasi klasifikasi tanpa Data Leakage."""
        
        X_train = self.vectorizer.fit_transform(train_df['Pesan']).toarray()
        X_test = self.vectorizer.transform(test_df['Pesan']).toarray()

        label_map = {'ham': 0, 'normal': 0, 'spam': 1}
        y_train = train_df['Kategori'].map(label_map).values
        y_test = test_df['Kategori'].map(label_map).values

        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'Akurasi': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
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