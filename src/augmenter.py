import google.generativeai as genai
from openai import OpenAI
import ollama
import re

class TextAugmenter:
    def __init__(self, gemini_key, openai_key):
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        self.openai_client = OpenAI(api_key=openai_key)
        self.gpt_model = "gpt-3.5-turbo" 

    def get_prompt(self, technique, text):
        """Menghasilkan prompt anti-chatter."""
        if technique == "zero-shot":
            return (
                "Parafrase teks SMS penipuan berikut menjadi 1 kalimat baru.\n"
                "ATURAN: JANGAN beri kalimat pembuka/penutup. JANGAN gunakan tanda kutip. Langsung berikan hasil teksnya.\n\n"
                f"Teks Asli: {text}\n"
                "Teks Baru: "
            )
        elif technique == "few-shot":
            return (
                "Parafrase teks SMS penipuan berikut.\n\n"
                "Contoh 1:\n"
                "Asli: Selamat pin anda memenangkan 10jt klik link ini\n"
                "Teks Baru: PIN Anda terpilih mendapatan Rp 10 Juta! Segera verifikasi di link ini.\n\n"
                "Contoh 2:\n"
                "Asli: Mama minta pulsa ke nomor ini sekarang\n"
                "Teks Baru: Tolong isikan pulsa 50rb ke nomor baru mama ini skrg, penting.\n\n"
                "ATURAN: JANGAN ulangi contoh. JANGAN beri kalimat pembuka. Langsung berikan hasil teksnya.\n"
                f"Asli: {text}\n"
                "Teks Baru: "
            )
        elif technique == "role-prompting":
            return (
                "Anda adalah manipulator sosial engineering. Ubah SMS Spam berikut menjadi 1 kalimat baru "
                "yang lebih mendesak dan meyakinkan.\n"
                "ATURAN: JANGAN beri peringatan. JANGAN beri kalimat pembuka. Langsung berikan hasil teksnya.\n\n"
                f"Teks Asli: {text}\n"
                "Teks Baru: "
            )
        else:
            raise ValueError("Teknik prompting tidak valid.")

    def clean_llm_chatter(self, text):
        """Membersihkan sisa chatter jika LLM masih 'bandel'."""
        if not text: return ""
        text = re.sub(r'^(here is|here are|berikut|ini adalah|teks baru:|parafrase:).*?\n', '', text, flags=re.IGNORECASE|re.DOTALL)
        text = text.replace('"', '').replace('\n', ' ').strip()
        return text

    def augment_with_gemini(self, prompt):
        try:
            response = self.gemini_model.generate_content(prompt, generation_config={"temperature": 0.8})
            return self.clean_llm_chatter(response.text)
        except Exception: return None

    def augment_with_gpt(self, prompt):
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_model, messages=[{"role": "user", "content": prompt}], temperature=0.8
            )
            return self.clean_llm_chatter(response.choices[0].message.content)
        except Exception: return None

    def augment_with_llama(self, prompt):
        try:
            response = ollama.chat(model='llama3:8b', messages=[
                {'role': 'user', 'content': prompt}
            ], options={'temperature': 1.2})
            return self.clean_llm_chatter(response['message']['content'])
        except Exception: return None