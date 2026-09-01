import os
import json
import io
import pandas as pd
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from src.augmenter import TextAugmenter
from src.evaluator import TextEvaluator
from src.classifier import FakeNewsClassifier
from src.data_loader import SMSDataLoader

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAW_DATA_PATH = "data/raw/sms_spam_indo.csv"

augmenter = TextAugmenter(GEMINI_API_KEY, OPENAI_API_KEY)
evaluator = TextEvaluator()
classifier = FakeNewsClassifier()

MODELS_CONFIG = {
#    'Gemini_3.5_Flash_Lite': ('gemini', 'gemini-3.5-flash-lite'),
   'GPT_3.5_Turbo': ('gpt', 'gpt-3.5-turbo'), #Legacy
#    'GPT_4o_Mini': ('gpt', 'gpt-4o-mini'),
   'GPT_4.1': ('gpt', 'gpt-4.1-2025-04-14'),
#    'GPT_4.1_Nano': ('gpt', 'gpt-4.1-nano'),
#    'GPT_5_Nano': ('gpt', 'gpt-5-nano'), #GPT versi 5 menggunakan temperatur default dan tidak bisa diubah

##Open source Ollama Models
    'LLaMA2_7B': ('ollama', 'llama2:7b'),
    'LLaMA3_8B': ('ollama', 'llama3:8b'),
    # 'Qwen3_8B': ('ollama', 'qwen3:8b'),
    # 'Gemma4_e4B': ('ollama', 'gemma4:e4b'),
    # 'Aya_Expanse_8B': ('ollama', 'aya-expanse:8b'),
    # 'Deepseek_r1_8B': ('ollama', 'deepseek-r1:8b'), #Model yang membutuhkan waktu untuk berpikir
}

class SingleTextRequest(BaseModel):
    text: str
    model_name: str
    technique: str
    temperature: float
    variance: int = 1

@app.get("/")
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/api/single")
async def process_single(data: SingleTextRequest):
    if data.model_name not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail="Model tidak valid.")
    
    provider, api_model_name = MODELS_CONFIG[data.model_name]
    prompt = augmenter.get_prompt(data.technique, data.text)
    
    # Looping sebanyak jumlah variansi yang diminta
    results = []
    for _ in range(data.variance):
        syn_text = ""
        if provider == 'gemini': 
            syn_text = augmenter.augment_with_gemini(prompt, api_model_name, data.temperature)
        elif provider == 'gpt': 
            syn_text = augmenter.augment_with_gpt(prompt, api_model_name, data.temperature)
        elif provider == 'ollama': 
            syn_text = augmenter.augment_with_ollama(prompt, api_model_name, data.temperature)
            
        if syn_text:
            results.append(syn_text)
            
    if not results:
        raise HTTPException(status_code=500, detail="Gagal menghasilkan teks.")
        
    return {"original": data.text, "augmented": results}


@app.websocket("/ws/batch")
async def process_batch(websocket: WebSocket):
    await websocket.accept()
    try:
        payload = await websocket.receive_text()
        config = json.loads(payload)
        
        model_name = config.get('model_name')
        technique = config.get('technique')
        temperature = float(config.get('temperature', 0.8))
        variance = int(config.get('variance', 1))
        csv_content = config.get('csv_data')
        
        provider, api_model_name = MODELS_CONFIG.get(model_name, ('ollama', 'llama3:8b'))
        
        df_input = pd.read_csv(io.StringIO(csv_content))
        if 'Pesan' not in df_input.columns:
            await websocket.send_json({"status": "error", "message": "CSV harus memiliki kolom 'Pesan'."})
            return

        spam_texts = df_input[df_input['Kategori'] == 'spam']['Pesan'].tolist() if 'Kategori' in df_input.columns else df_input['Pesan'].tolist()
        
        ## Batasi data batch untuk demo interface
        # spam_texts = spam_texts[:20] 
        total_rows = len(spam_texts)
        synthetic_data = []
        aligned_originals = []
        
        for i, original_text in enumerate(spam_texts):
            prompt = augmenter.get_prompt(technique, original_text)
            
            for _ in range(variance):
                if provider == 'gemini': syn_text = augmenter.augment_with_gemini(prompt, api_model_name, temperature)
                elif provider == 'gpt': syn_text = augmenter.augment_with_gpt(prompt, api_model_name, temperature)
                else: syn_text = augmenter.augment_with_ollama(prompt, api_model_name, temperature)
                    
                if syn_text:
                    synthetic_data.append({'Kategori': 'spam', 'Pesan': syn_text})
                    aligned_originals.append(original_text)
            
            await websocket.send_json({
                "status": "processing",
                "progress": int(((i + 1) / total_rows) * 50),
                "message": f"Augmentasi LLM: Baris {i+1} dari {total_rows} selesai..."
            })
            
        df_synthetic = pd.DataFrame(synthetic_data)
        
        # Evaluasi Kualitas Teks
        await websocket.send_json({"status": "processing", "progress": 65, "message": "Mengevaluasi Kualitas Semantik"})
        df_scored, avg_scores = evaluator.evaluate_dataframe(df_synthetic, aligned_originals)
        for k in avg_scores.keys(): avg_scores[k] = round(avg_scores[k], 4)
            
        # Train XGBoost
        await websocket.send_json({"status": "processing", "progress": 85, "message": "Melatih XGBoost Classifier untuk Komparasi..."})
        
        loader = SMSDataLoader(RAW_DATA_PATH)
        normal_df, spam_df = loader.process()
        baseline_df = pd.concat([normal_df, spam_df], ignore_index=True)
        merged_df = pd.concat([baseline_df, df_synthetic], ignore_index=True)
        
        if 'Kategori' in merged_df.columns and 'Pesan' in merged_df.columns:
            merged_df = merged_df[['Kategori', 'Pesan']]

        metrics_baseline = classifier.train_and_evaluate(baseline_df)
        metrics_augmented = classifier.train_and_evaluate(merged_df)
        csv_export = merged_df.to_csv(index=False)
        
        await websocket.send_json({
            "status": "done",
            "progress": 100,
            "message": "Pemrosesan & Evaluasi Selesai!",
            "text_quality": avg_scores,
            "classification_baseline": metrics_baseline,
            "classification_augmented": metrics_augmented,
            "merged_csv": csv_export
        })
        
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})