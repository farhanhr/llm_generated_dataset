import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

class FakeNewsClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

    def train_and_evaluate(self, df):
        """Melatih model klasifikasi dan mengembalikan metrik performa."""
        X = self.vectorizer.fit_transform(df['clean_text']).toarray()
        
        y = df['Kategori'].map({'ham': 0, 'spam': 1}).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'Akurasi': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        return metrics