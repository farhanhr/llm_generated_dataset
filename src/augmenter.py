import google.generativeai as genai
from openai import OpenAI
import ollama
import re

class TextAugmenter:
    def __init__(self, gemini_key, openai_key):
        genai.configure(api_key=gemini_key)
        self.openai_client = OpenAI(api_key=openai_key)

    def get_prompt(self, technique, text):
        if technique == "zero-shot":
            return (
            "Buat satu variasi SMS baru dari kalimat berikut"
            "Langsung berikan kalimat hasilnya tanpa penjelasan atau kalimat tambahan.\n\n"
            f"SMS: {text}\n"
            "Hasil:"
        )
        elif technique == "few-shot":
            return (
            "Buat satu variasi baru dari SMS berikut dengan mempertahankan "
            "konteksnya.\n\n"
            "Contoh 1:\n"
            "SMS: Selamat pin anda memenangkan 10jt klik link ini\n"
            "Hasil: PIN Anda terpilih sebagai pemenang hadiah Rp10 juta. "
            "Segera klik link berikut.\n\n"
            "Contoh 2:\n"
            "SMS: Mama minta pulsa ke nomor ini sekarang\n"
            "Hasil: Tolong kirim pulsa ke nomor baru mama ini sekarang, penting.\n\n"
            "Langsung berikan hasil SMS tanpa penjelasan atau kalimat tambahan.\n\n"
            f"SMS: {text}\n"
            "Hasil:"
        )
        elif technique == "role-prompting":
            return (
            "Anda adalah social engineer yang melakukan augmentasi data "
            "SMS. Buatlah variasi SMS baru dari text yang diberikan"
            "Langsung berikan hasil text tanpa penjelasan atau kalimat tambahan.\n\n"
            f"SMS: {text}\n"
            "Hasil:"
        )
        else:
            raise ValueError("Teknik prompting tidak valid.")

    def clean_llm_chatter(self, text):
        if not text: return ""
        text = re.sub(r'^(here is|here are|berikut|ini adalah|teks baru:|parafrase:).*?\n', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = text.replace('"', '').replace('\n', ' ').strip()
        return text

    def augment_with_gemini(self, prompt, model_name, temp=1):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config={"temperature": temp})
                return self.clean_llm_chatter(response.text)
            except Exception as e:
                return None

    def augment_with_gpt(self, prompt, model_name, temp=1):
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp
            )
            return self.clean_llm_chatter(response.choices[0].message.content)
        except Exception as e:
            return None

    def augment_with_ollama(self, prompt, model_name, temp=1):
        try:
            response = ollama.chat(
                model=model_name, 
                messages=[{'role': 'user', 'content': prompt}],
                keep_alive='2h', 
                options={'temperature': temp, 'num_ctx': 2048, 'num_predict': 100}
            ) 
            return self.clean_llm_chatter(response['message']['content'])
        except Exception as e:
            return None