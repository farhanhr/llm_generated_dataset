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
SUMOPOD_API_KEY = os.getenv("SUMOPOD_API_KEY")
RAW_DATA_PATH = "data/raw/train_data.csv"

MODELS_CONFIG = {
    # 'Gemini_3.5_Flash': ('sumopod', 'gemini/gemini-3.5-flash'),
    # 'GPT_3.5_Turbo': ('gpt', 'gpt-3.5-turbo'),
    # 'GPT_4.1': ('gpt', 'gpt-4.1-2025-04-14'),
    # 'LLaMA2_7B': ('ollama', 'llama2:7b'),
    # 'LLaMA3_8B': ('ollama', 'llama3:8b'),
    'Qwen3_8B': ('ollama', 'qwen3:8b')
}

class TargetedBalancer:
    def __init__(self, timestamp_folder):
        self.timestamp = timestamp_folder
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(current_dir, '..'))
        
        self.raw_data_path = os.path.join(self.root_dir, RAW_DATA_PATH)
        self.base_dir = os.path.join(self.root_dir, 'data', 'augmented', self.timestamp)
        
        self.synthetic_dir = os.path.join(self.base_dir, 'synthetic')
        self.log_dir = os.path.join(self.base_dir, 'augmented_log')
        self.balanced_dir = os.path.join(self.base_dir, 'merged', 'balanced')
        
        os.makedirs(self.balanced_dir, exist_ok=True)
        
        self.raw_df = pd.read_csv(self.raw_data_path).dropna(subset=['Kategori', 'Pesan'])
        self.text_to_label = dict(zip(self.raw_df['Pesan'].astype(str).str.strip(), self.raw_df['Kategori']))
        
        self.augmenter = TextAugmenter(GEMINI_API_KEY, OPENAI_API_KEY, SUMOPOD_API_KEY)

    def process(self):
        if not os.path.exists(self.log_dir):
            print(f"Error: Folder Log {self.log_dir} tidak ditemukan.")
            return

        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.csv')]
        
        if not log_files:
            print("Tidak ada file log untuk diproses.")
            return

        for filename in log_files:
            clean_name = filename.replace("log_", "").replace(".csv", "")
            parts = clean_name.split("_")
            technique = parts[-1] 
            model_name = "_".join(parts[:-1])
            
            if model_name not in MODELS_CONFIG:
                continue
                
            provider, api_model_name = MODELS_CONFIG[model_name]
            log_path = os.path.join(self.log_dir, filename)
            
            print(f"\n[SCANNING LUBANG KOSONG] {model_name} | {technique}")
            df_log = pd.read_csv(log_path)
            
            para_cols = [c for c in df_log.columns if c.startswith('parafrase_')]
            
            lubang_ditemukan = 0
            
            # Iterasi mencari data kosong
            for idx, row in tqdm(df_log.iterrows(), total=len(df_log), desc="Filling Gaps"):
                original_text = str(row['original']).strip()
                label = self.text_to_label.get(original_text)
                
                if not label:
                    continue 
                
                for col in para_cols:
                    cell_value = row[col]
                    if pd.isna(cell_value) or str(cell_value).strip() == "":
                        lubang_ditemukan += 1
                        prompt = self.augmenter.get_prompt(technique, original_text)
                        
                        # Panggil LLM
                        syn_text = ""
                        if provider == 'gemini': 
                            syn_text = self.augmenter.augment_with_gemini(prompt, api_model_name)
                        elif provider == 'gpt': 
                            syn_text = self.augmenter.augment_with_gpt(prompt, api_model_name)
                        elif provider == 'ollama': 
                            syn_text = self.augmenter.augment_with_ollama(prompt, api_model_name)
                        elif provider == 'sumopod': 
                            syn_text = self.augmenter.augment_with_sumopod(prompt, api_model_name)
                        
                        if syn_text:
                            df_log.at[idx, col] = syn_text
            
            if lubang_ditemukan == 0:
                print("-> Dataset sudah sempurna (Tidak ada error). Skip regerenasi.")
            else:
                print(f"-> Berhasil menambal {lubang_ditemukan} data kosong.")
                df_log.to_csv(log_path, index=False, quoting=1)
            
            synthetic_records = []
            for _, row in df_log.iterrows():
                original_text = str(row['original']).strip()
                label = self.text_to_label.get(original_text)
                if label:
                    for col in para_cols:
                        val = row[col]
                        if pd.notna(val) and str(val).strip() != "":
                            synthetic_records.append({'Kategori': label, 'Pesan': str(val).strip()})
                            
            df_new_syn = pd.DataFrame(synthetic_records)
            
            syn_path = os.path.join(self.synthetic_dir, f"synthetic_{clean_name}.csv")
            df_new_syn.to_csv(syn_path, index=False, quoting=1)
            
            merged_df = pd.concat([self.raw_df, df_new_syn], ignore_index=True)
            balanced_path = os.path.join(self.balanced_dir, f"balanced_{clean_name}.csv")
            merged_df.to_csv(balanced_path, index=False, quoting=1)
            
            print(f"-> File Synthetic & Balanced berhasil direkonstruksi.")

if __name__ == "__main__":
    TARGET_TIMESTAMP = "20260913182453" 
    
    balancer = TargetedBalancer(TARGET_TIMESTAMP)
    balancer.process()