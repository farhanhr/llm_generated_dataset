import os
import pandas as pd
from tqdm import tqdm
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

from augmenter import TextAugmenter

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAW_DATA_PATH = "data/raw/train_data.csv"

MODELS_CONFIG = {
    'GPT_3.5_Turbo': ('gpt', 'gpt-3.5-turbo'),
    'GPT_4.1': ('gpt', 'gpt-4.1-2025-04-14'),
    'LLaMA2_7B': ('ollama', 'llama2:7b'),
    'LLaMA3_8B': ('ollama', 'llama3:8b'),
}

class DatasetBalancer:
    def __init__(self, timestamp_folder, target_multiplier=None):
        """
        :param target_multiplier: Angka multiplier (NUM_VARIATIONS) dari main_augment.
        Jika None, hanya mem-balance mengikuti kelas mayoritas saat ini.
        """
        self.timestamp = timestamp_folder
        self.target_multiplier = target_multiplier
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(current_dir, '..'))
        
        self.raw_data_path = os.path.join(self.root_dir, RAW_DATA_PATH)
        self.base_dir = os.path.join(self.root_dir, 'data', 'augmented', self.timestamp)
        self.synthetic_dir = os.path.join(self.base_dir, 'synthetic')
        self.log_dir = os.path.join(self.base_dir, 'augmented_log')
        self.balanced_dir = os.path.join(self.base_dir, 'merged', 'balanced')
        
        os.makedirs(self.balanced_dir, exist_ok=True)
        
        self.raw_df = pd.read_csv(self.raw_data_path).dropna(subset=['Kategori', 'Pesan'])
        self.augmenter = TextAugmenter(GEMINI_API_KEY, OPENAI_API_KEY)
        
        raw_counts = self.raw_df['Kategori'].value_counts()
        self.raw_spam = raw_counts.get('spam', 0)
        self.raw_ham = raw_counts.get('ham', 0)
        self.max_raw = max(self.raw_spam, self.raw_ham)

    def extract_model_info(self, filename):
        name = filename.replace("synthetic_", "").replace(".csv", "")
        parts = name.split("_")
        technique = parts[-1] 
        model_name = "_".join(parts[:-1])
        return model_name, technique

    def process(self):
        if not os.path.exists(self.synthetic_dir):
            print(f"Error: Folder {self.synthetic_dir} tidak ditemukan.")
            return

        synthetic_files = [f for f in os.listdir(self.synthetic_dir) if f.endswith('.csv')]
        if not synthetic_files:
            print("Tidak ada file synthetic untuk diproses.")
            return

        for filename in synthetic_files:
            file_path = os.path.join(self.synthetic_dir, filename)
            df_syn = pd.read_csv(file_path).dropna(subset=['Kategori', 'Pesan'])
            model_name, technique = self.extract_model_info(filename)
            
            syn_counts = df_syn['Kategori'].value_counts()
            syn_spam = syn_counts.get('spam', 0)
            syn_ham = syn_counts.get('ham', 0)
            
            total_current_spam = self.raw_spam + syn_spam
            total_current_ham = self.raw_ham + syn_ham
            
            print(f"\n[{model_name} | {technique}]")
            print(f"Total Saat Ini (Raw+Syn) -> Spam: {total_current_spam}, Ham: {total_current_ham}")

            if self.target_multiplier is not None:
                target_per_class = self.max_raw + (self.max_raw * self.target_multiplier)
                print(f"Mode: Multiplier ({self.target_multiplier}x). Target per kelas: {target_per_class}")
            else:
                target_per_class = max(total_current_spam, total_current_ham)
                print(f"Mode: Default Balance. Target mengikuti mayoritas: {target_per_class}")

            deficit_spam = max(0, target_per_class - total_current_spam)
            deficit_ham = max(0, target_per_class - total_current_ham)
            
            if deficit_spam == 0 and deficit_ham == 0:
                print("-> Dataset sudah mencapai target dan seimbang sempurna. Skip generasi.")
                merged_df = pd.concat([self.raw_df, df_syn], ignore_index=True)
                merged_path = os.path.join(self.balanced_dir, f"balanced_{model_name}_{technique}.csv")
                merged_df.to_csv(merged_path, index=False, quoting=1)
                continue
                
            print(f"-> Defisit yang harus dibuat -> Spam: {deficit_spam}, Ham: {deficit_ham}")
            
            if model_name not in MODELS_CONFIG:
                print(f"Error: Konfigurasi model '{model_name}' tidak ditemukan. Lewati.")
                continue
            provider, api_model_name = MODELS_CONFIG[model_name]
            
            new_synthetic_rows = []
            new_log_rows = []
            
            def generate_for_class(cls_name, deficit_count):
                if deficit_count <= 0: return
                raw_cls = self.raw_df[self.raw_df['Kategori'] == cls_name]
                replace = deficit_count > len(raw_cls)
                sampled_raw = raw_cls.sample(n=deficit_count, replace=replace, random_state=42)
                
                for _, row in tqdm(sampled_raw.iterrows(), total=deficit_count, desc=f"Gen {cls_name}"):
                    original_text = str(row['Pesan'])
                    prompt = self.augmenter.get_prompt(technique, original_text)
                    
                    if provider == 'gemini': syn_text = self.augmenter.augment_with_gemini(prompt, api_model_name)
                    elif provider == 'gpt': syn_text = self.augmenter.augment_with_gpt(prompt, api_model_name)
                    elif provider == 'ollama': syn_text = self.augmenter.augment_with_ollama(prompt, api_model_name)
                    else: syn_text = ""
                        
                    if syn_text:
                        new_synthetic_rows.append({'Kategori': cls_name, 'Pesan': syn_text})
                        new_log_rows.append({'original': original_text, 'parafrase_1': syn_text})
            
            generate_for_class('spam', deficit_spam)
            generate_for_class('ham', deficit_ham)
            
            if not new_synthetic_rows:
                print("-> Gagal menghasilkan data sintetis baru (API Error).")
                continue
                
            df_new_log = pd.DataFrame(new_log_rows)
            log_csv_path = os.path.join(self.log_dir, f"log_{model_name}_{technique}.csv")
            if os.path.exists(log_csv_path):
                old_log = pd.read_csv(log_csv_path)
                for col in old_log.columns:
                    if col not in df_new_log.columns: df_new_log[col] = ""
                df_new_log = df_new_log[old_log.columns]
                df_new_log.to_csv(log_csv_path, mode='a', header=False, index=False, quoting=1)
            else:
                df_new_log.to_csv(log_csv_path, mode='w', header=True, index=False, quoting=1)
                
            df_new_syn = pd.DataFrame(new_synthetic_rows)
            df_new_syn.to_csv(file_path, mode='a', header=False, index=False, quoting=1)
            
            df_syn_updated = pd.read_csv(file_path)
            merged_df = pd.concat([self.raw_df, df_syn_updated], ignore_index=True)
            balanced_path = os.path.join(self.balanced_dir, f"balanced_{model_name}_{technique}.csv")
            merged_df.to_csv(balanced_path, index=False, quoting=1)
            
            print(f"-> Selesai! Data diseimbangkan dan disimpan ke {balanced_path}")

if __name__ == "__main__":
    TARGET_TIMESTAMP = "20260902141148" 
    
    MULTIPLIER = 3 
    
    balancer = DatasetBalancer(TARGET_TIMESTAMP, target_multiplier=MULTIPLIER)
    balancer.process()