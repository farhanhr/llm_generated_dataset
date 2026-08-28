import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import SMSDataLoader
from src.augmenter import TextAugmenter

# ================= KONFIGURASI =================
GEMINI_API_KEY = "DUMMY_KEY"
OPENAI_API_KEY = "DUMMY_KEY"
DATA_PATH = "data/raw/sms_spam_indo.csv"
NUM_VARIATIONS = 2  

MODELS_CONFIG = {
    'Gemini_3.5_Flash_Lite': ('gemini', 'gemini-3.5-flash-lite'),
    'GPT_4.1_Nano': ('gpt', 'gpt-4.1-nano'),
    'GPT_5_Nano': ('gpt', 'gpt-5-nano'),
    'LLaMA2_7B': ('ollama', 'llama2:7b'),
    'LLaMA3_8B': ('ollama', 'llama3:8b'),
    'Qwen3_8B': ('ollama', 'qwen3:8b'),
    'Gemma4_e4B': ('ollama', 'gemma4:e4b')
}

TECHNIQUES_TO_RUN = ['zero-shot', 'few-shot', 'role-prompting']

def create_directory_structure(base_dir):
    folders = ['synthetic', 'merged', 'augmented_log']
    paths = {}
    for f in folders:
        path = os.path.join(base_dir, f)
        os.makedirs(path, exist_ok=True)
        paths[f] = path
    return paths

def save_txt_log(original_text, synthetic_texts, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f'Original : "{original_text}"\n')
        f.write('Parafrase :\n')
        for syn in synthetic_texts:
            f.write(f'- "{syn}"\n')
        f.write('\n' + '='*50 + '\n\n')

def main():
    loader = SMSDataLoader(DATA_PATH)
    normal_df, spam_df = loader.process()
    
    # Batasi sementara untuk testing. Ganti head(3) menjadi tolist() untuk full dataset.
    spam_texts = spam_df['Pesan'].head(3).tolist() 
    
    augmenter = TextAugmenter(GEMINI_API_KEY, OPENAI_API_KEY)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = f"data/augmented/{timestamp}"
    dirs = create_directory_structure(base_dir)
    print(f"Struktur direktori dibuat di: {base_dir}")
    
    for model_display, (provider, api_model_name) in MODELS_CONFIG.items():
        for technique in TECHNIQUES_TO_RUN:
            print(f"\n---> Memproses: [{model_display}] | [{technique}] | Multiplier: {NUM_VARIATIONS}x")
            
            synthetic_data = []
            log_csv_data = [] 
            txt_log_path = os.path.join(dirs['augmented_log'], f"log_{model_display}_{technique}.txt")
            
            for original_text in tqdm(spam_texts, desc=f"{model_display} - {technique}"):
                prompt = augmenter.get_prompt(technique, original_text)
                current_paraphrases = []
                row_log = {'original': original_text}
                
                for i in range(NUM_VARIATIONS):
                    if provider == 'gemini': 
                        syn_text = augmenter.augment_with_gemini(prompt, api_model_name)
                    elif provider == 'gpt': 
                        syn_text = augmenter.augment_with_gpt(prompt, api_model_name)
                    elif provider == 'ollama': 
                        syn_text = augmenter.augment_with_ollama(prompt, api_model_name)
                    else: 
                        syn_text = ""
                    
                    if syn_text and syn_text not in current_paraphrases:
                        current_paraphrases.append(syn_text)
                        synthetic_data.append({'Kategori': 'spam', 'Pesan': syn_text})
                        row_log[f'parafrase_{i+1}'] = syn_text
                    else:
                        row_log[f'parafrase_{i+1}'] = "" 
                
                if current_paraphrases:
                    save_txt_log(original_text, current_paraphrases, txt_log_path)
                    log_csv_data.append(row_log)
            
            df_log_csv = pd.DataFrame(log_csv_data)
            if not df_log_csv.empty:
                file_log_csv = os.path.join(dirs['augmented_log'], f"log_{model_display}_{technique}.csv")
                df_log_csv.to_csv(file_log_csv, index=False, quoting=1)
            
            synthetic_df = pd.DataFrame(synthetic_data, columns=['Kategori', 'Pesan'])
            if not synthetic_df.empty:
                file_synth = os.path.join(dirs['synthetic'], f"synthetic_{model_display}_{technique}.csv")
                synthetic_df.to_csv(file_synth, index=False, quoting=1) 
                
                raw_df_cleaned = pd.concat([normal_df[['Kategori', 'Pesan']], spam_df[['Kategori', 'Pesan']]])
                merged_df = pd.concat([raw_df_cleaned, synthetic_df], ignore_index=True)
                
                file_merged = os.path.join(dirs['merged'], f"merged_{model_display}_{technique}.csv")
                merged_df.to_csv(file_merged, index=False, quoting=1)

if __name__ == "__main__":
    main()