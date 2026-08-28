import pandas as pd
import re

class SMSDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def clean_text(self, text):
        """Standardisasi teks dasar."""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def process(self):
        """Memuat data dan memisahkan kelas spam untuk diaugmentasi."""
        df = pd.read_csv(self.file_path)
        
        df['clean_text'] = df['Pesan'].apply(self.clean_text)
        
        spam_df = df[df['Kategori'] == 'spam'].copy()
        normal_df = df[df['Kategori'] == 'ham'].copy()
        
        return normal_df, spam_df