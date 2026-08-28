import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import SMSDataLoader
from src.augmenter import TextAugmenter

GEMINI_API_KEY = "DUMMY_KEY"
OPENAI_API_KEY = "DUMMY_KEY"
DATA_PATH = "data/raw/sms_spam_indo.csv"

NUM_VARIATIONS = 2  

LLMS_TO_RUN = ['LLaMA'] 
TECHNIQUES_TO_RUN = ['zero-shot', 'few-shot', 'role-prompting']

def create_directory_structure(base_dir):
    """Membuat arsitektur folder baru sesuai permintaan."""
    folders = ['synthetic', 'merged', 'txt_logs']
    paths = {}
    for f in folders:
        path = os.path.join(base_dir, f)
        os.makedirs(path, exist_ok=True)
        paths[f] = path
    return paths

def save_txt_log(original_text, synthetic_texts, filepath):
    """Menyimpan log txt histori parafrase."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f'Original : "{original_text}"\n')
        f.write('Parafrase :\n')
        for syn in synthetic_texts:
            f.write(f'- "{syn}"\n')
        f.write('\n' + '='*50 + '\n\n')

def main():
    print("1. Memuat data...")
    loader = SMSDataLoader(DATA_PATH)
    normal_df, spam_df = loader.process()
    
    spam_texts = spam_df['Pesan'].tolist() 
    
    augmenter = TextAugmenter(GEMINI_API_KEY, OPENAI_API_KEY)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base_dir = f"data/augmented/{timestamp}"
    dirs = create_directory_structure(base_dir)
    print(f"Struktur direktori dibuat di: {base_dir}")
    
    for llm in LLMS_TO_RUN:
        for technique in TECHNIQUES_TO_RUN:
            print(f"\n---> Memproses: [{llm}] | [{technique}] | Multiplier: {NUM_VARIATIONS}x")
            
            synthetic_data = []
            txt_log_path = os.path.join(dirs['txt_logs'], f"log_{llm}_{technique}.txt")
            
            for original_text in tqdm(spam_texts, desc=f"{llm} - {technique}"):
                prompt = augmenter.get_prompt(technique, original_text)
                
                current_paraphrases = []

                for _ in range(NUM_VARIATIONS):
                    if llm == 'Gemini':
                        syn_text = augmenter.augment_with_gemini(prompt)
                    elif llm == 'GPT':
                        syn_text = augmenter.augment_with_gpt(prompt)
                    elif llm == 'LLaMA':
                        syn_text = augmenter.augment_with_llama(prompt)
                    
                    if syn_text and syn_text not in current_paraphrases:
                        current_paraphrases.append(syn_text)
                        synthetic_data.append({
                            'Kategori': 'spam',
                            'Pesan': syn_text
                        })
                
                if current_paraphrases:
                    save_txt_log(original_text, current_paraphrases, txt_log_path)
            
            synthetic_df = pd.DataFrame(synthetic_data, columns=['Kategori', 'Pesan'])
            
            if not synthetic_df.empty:
                file_synth = os.path.join(dirs['synthetic'], f"synthetic_{llm}_{technique}.csv")
                synthetic_df.to_csv(file_synth, index=False, quoting=1) # quoting=1 memaksa semua teks diapit kutipan ganda untuk aman dari koma
                
                raw_df_cleaned = pd.concat([normal_df[['Kategori', 'Pesan']], spam_df[['Kategori', 'Pesan']]])
                merged_df = pd.concat([raw_df_cleaned, synthetic_df], ignore_index=True)
                
                file_merged = os.path.join(dirs['merged'], f"merged_{llm}_{technique}.csv")
                merged_df.to_csv(file_merged, index=False, quoting=1)
                
                print(f"Selesai! Baris Asli: {len(raw_df_cleaned)} | Baris Sintetis: {len(synthetic_df)} | Total Merged: {len(merged_df)}")

if __name__ == "__main__":
    main()