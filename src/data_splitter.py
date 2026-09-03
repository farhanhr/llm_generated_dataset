import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, raw_data_path, output_dir):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir

    def split_and_save(self, test_size=0.2, random_state=42):
        print(f"Membaca data dari: {self.raw_data_path}")
        try:
            df = pd.read_csv(self.raw_data_path)
            
            if 'Kategori' not in df.columns or 'Pesan' not in df.columns:
                raise ValueError("Dataset harus memiliki kolom 'Kategori' dan 'Pesan'")

            df = df[['Kategori', 'Pesan']].dropna()

            print(f"Total data awal: {len(df)} baris")

            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=df['Kategori']
            )

            print(f"Data latih (Train): {len(train_df)} baris")
            print(f"Data uji (Test) : {len(test_df)} baris")

            # Simpan ke CSV
            train_path = os.path.join(self.output_dir, "train_data.csv")
            test_path = os.path.join(self.output_dir, "test_data.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"Data latih disimpan di: {train_path}")
            print(f"Data uji disimpan di: {test_path}")

        except Exception as e:
            print(f"Terjadi kesalahan saat membagi data: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    
    raw_file_path = os.path.join(root_dir, 'data', 'raw', 'sms_spam_indo.csv')
    output_directory = os.path.join(root_dir, 'data', 'raw')
    
    splitter = DataSplitter(raw_file_path, output_directory)
    splitter.split_and_save()