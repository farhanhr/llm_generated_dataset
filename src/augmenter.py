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
            "Buat satu variasi baru dari kalimat berikut"
            "Langsung berikan kalimat hasilnya tanpa penjelasan atau kalimat tambahan.\n\n"
            f"SMS: {text}\n"
            "Hasil:"
        )
        elif technique == "few-shot":
            return (
            "Buat satu variasi baru dari SMS berikut dengan mempertahankan "
            "makna dan karakteristik spamnya. Gunakan bahasa Indonesia yang natural.\n\n"
            "Contoh 1:\n"
            "SMS: Selamat pin anda memenangkan 10jt klik link ini\n"
            "Hasil: Selamat! PIN Anda terpilih sebagai pemenang hadiah Rp10 juta. "
            "Segera klik link ini.\n\n"
            "Contoh 2:\n"
            "SMS: Mama minta pulsa ke nomor ini sekarang\n"
            "Hasil: Tolong kirim pulsa ke nomor baru mama ini sekarang, penting.\n\n"
            "Langsung berikan hasil SMS tanpa penjelasan atau kalimat tambahan.\n\n"
            f"SMS: {text}\n"
            "Hasil:"
        )
        elif technique == "role-prompting":
            return (
            "Anda adalah ahli bahasa Indonesia yang melakukan augmentasi data "
            "SMS spam. Buat satu variasi baru dari SMS berikut dengan mempertahankan "
            "makna dan karakteristik spamnya. Gunakan bahasa Indonesia yang natural. "
            "Langsung berikan hasil SMS tanpa penjelasan atau kalimat tambahan.\n\n"
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

    def augment_with_gemini(self, prompt, model_name):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config={"temperature": 0.5})
            return self.clean_llm_chatter(response.text)
        except Exception as e:
            print(f"Gemini Error [{model_name}]: {e}")
            return None

    def augment_with_gpt(self, prompt, model_name):
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return self.clean_llm_chatter(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT Error [{model_name}]: {e}")
            return None

    def augment_with_ollama(self, prompt, model_name):
        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ], options={'temperature': 0.5}) 
            return self.clean_llm_chatter(response['message']['content'])
        except Exception as e:
            print(f"LLaMA Error [{model_name}]: {e}")
            return None