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
            return f"Parafrase teks SMS penipuan berikut:\n\n{text}. Langsung berikan hasil dari parafrasemu tanpa penambahan kalimat apapun yang tidak relevan dengan hasil"
        elif technique == "few-shot":
            return (
                "Parafraselah data teks yang diberikan.\n\n"
                "Contoh 1:\n"
                "Asli: Selamat pin anda memenangkan 10jt klik link ini\n"
                "Teks Baru: PIN Anda terpilih mendapatkan Rp 10 Juta! Segera verifikasi di link ini.\n\n"
                "Contoh 2:\n"
                "Asli: Mama minta pulsa ke nomor ini sekarang\n"
                "Teks Baru: Tolong isikan pulsa 50rb ke nomor baru mama ini skrg, penting.\n\n"
                "ATURAN: JANGAN ulangi contoh. JANGAN beri kalimat pembuka. Langsung berikan hasil teksnya, tanpa penambahan kalimat tambahan yang tidak relevan dengan hasil.\n"
                f"Asli: {text}\n"
                "Teks Baru: "
            )
        elif technique == "role-prompting":
            return (
                "Kamu adalah seorang peneliti sosial dengan pengetahuan tentang komunikasi digital dan spam. Ubah Kalimat SMS yang diberikan menjadi 1 kalimat baru "
                "yang lebih meyakinkan.\n"
                "ATURAN: JANGAN beri peringatan. JANGAN beri kalimat pembuka. Langsung berikan hasil teksnya, tanpa penambahan kalimat tambahan yang tidak relevan dengan hasil.\n\n"
                f"Teks Asli: {text}\n"
                "Teks Baru: "
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
            response = model.generate_content(prompt, generation_config={"temperature": 1})
            return self.clean_llm_chatter(response.text)
        except Exception as e:
            print(f"Gemini Error [{model_name}]: {e}")
            return None

    def augment_with_gpt(self, prompt, model_name):
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=1
            )
            return self.clean_llm_chatter(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT Error [{model_name}]: {e}")
            return None

    def augment_with_ollama(self, prompt, model_name):
        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ], options={'temperature': 1}) 
            return self.clean_llm_chatter(response['message']['content'])
        except Exception as e:
            print(f"LLaMA Error [{model_name}]: {e}")
            return None