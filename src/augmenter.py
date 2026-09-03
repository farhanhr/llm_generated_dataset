from google import genai
from google.genai import types
from openai import OpenAI
import ollama
import re

class TextAugmenter:
    def __init__(self, gemini_key, openai_key):
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.openai_client = OpenAI(api_key=openai_key)

    def get_prompt(self, technique, text):
        if technique == "zero-shot":
            return (
                f"""
                OBJECTIVE:
                Buat satu SMS baru berdasarkan SMS yang diberikan.

                CONSTRAINTS:
                1. Pertahankan informasi penting yang terdapat dalam SMS asli.
                2. Gunakan bahasa yang natural dan realistis seperti SMS yang ditulis manusia.
                3. Pertahankan karakteristik gaya SMS apabila terdapat pada SMS asli,
                seperti bahasa indonesia yang informal dan tidak baku serta singkatan.
                4. Jangan mengubah makna utama SMS.
                5. Jangan menghasilkan salinan identik dari SMS asli.

                OUTPUT:
                Hanya keluarkan satu SMS hasil augmentasi.
                Jangan berikan label, penjelasan, catatan, pembuka, penutup,
                tanda kutip, atau markdown.
                Mulai langsung dengan teks SMS dan berhenti setelah SMS selesai.

                INPUT SMS:
                {text}

                GENERATE ONLY THE SMS BELOW:
                """
                )
        
        elif technique == "few-shot":
            return (
                f"""
                OBJECTIVE:
                Buat satu SMS baru berdasarkan SMS yang diberikan.

                CONSTRAINTS:
                1. Pertahankan informasi penting yang terdapat dalam SMS asli.
                2. Gunakan bahasa yang natural dan realistis seperti SMS yang ditulis manusia.
                3. Pertahankan karakteristik gaya SMS apabila terdapat pada SMS asli,
                seperti bahasa indonesia yang informal dan tidak baku serta singkatan.
                4. Jangan mengubah makna utama SMS.
                5. Jangan menghasilkan salinan identik dari SMS asli.

                OUTPUT:
                Hanya keluarkan satu SMS hasil augmentasi.
                Jangan berikan label, penjelasan, catatan, pembuka, penutup,
                tanda kutip, atau markdown.
                Mulai langsung dengan teks SMS dan berhenti setelah SMS selesai.

                EXAMPLES:

                SPAM Example 1:
                INPUT:
                SMS: Selamat! Anda memenangkan hadiah Rp10.000.000. Hubungi 08123456789 untuk klaim sekarang.
                OUTPUT:
                Kamu berhak mendapatkan hadiah Rp10 juta. Segera hubungi 08123456789 untuk proses klaim.

                SPAM Example 2:
                INPUT:
                SMS: Dapatkan pinjaman cepat tanpa jaminan. Cair dalam 24 jam. Klik https://contoh.com sekarang!
                OUTPUT:
                Butuh dana cepat? Pinjaman tanpa jaminan bisa cair dalam waktu 24 jam. Langsung klik https://contoh.com ya!

                HAM Example 1:
                INPUT: Nanti sore jadi kumpul di kafe biasa gak?
                OUTPUT: Sore ini kita tetep nongkrong di tempat biasa kan?

                HAM Example 2:
                INPUT: Tolong belikan telur sama beras ya pas pulang nanti.
                OUTPUT: Nanti pas jalan pulang, titip beliin beras sama telur dong.

                
                INPUT SMS:
                {text}

                GENERATE ONLY THE SMS BELOW:
                """
            )
        elif technique == "role-prompting":
            return (
                f"""
                ROLE / TASK:
                Anda adalah seorang NLP Data Augmentation Specialist yang berpengalaman
                dalam membuat data sintetis untuk dataset klasifikasi SMS.

                OBJECTIVE:
                Buat satu SMS baru berdasarkan SMS yang diberikan.

                CONSTRAINTS:
                1. Pertahankan informasi penting yang terdapat dalam SMS asli.
                2. Gunakan bahasa yang natural dan realistis seperti SMS yang ditulis manusia.
                3. Pertahankan karakteristik gaya SMS apabila terdapat pada SMS asli,
                seperti bahasa indonesia yang informal dan tidak baku serta singkatan.
                4. Jangan mengubah makna utama SMS.
                5. Jangan menghasilkan salinan identik dari SMS asli.

                OUTPUT:
                Hanya keluarkan satu SMS hasil augmentasi.
                Jangan berikan label, penjelasan, catatan, pembuka, penutup,
                tanda kutip, atau markdown.
                Mulai langsung dengan teks SMS dan berhenti setelah SMS selesai.

                INPUT SMS:
                {text}

                GENERATE ONLY THE SMS BELOW:
                """
            )

        else:
            raise ValueError("Teknik prompting tidak valid.")

    def clean_llm_chatter(self, text):
        if not text:
            return ""

        text = text.strip()

        patterns = [
            r'^(?:hasil(?: augmentasi)?|output|result|augmented(?: sms)?(?: result)?|sms(?: baru| augmented output)?)\s*:\s*',
            r'^(?:berikut(?: adalah)?|ini adalah)\s+(?:hasil(?: augmentasi)?|output|sms(?: baru)?)\s*:\s*',
            r'^(?:here is|here are|sure|certainly)\s*:?\s*',
            r'^(?:here is|here are)\s+the\s+(?:augmented\s+)?sms\s*:?\s*',
            r'^I can help you generate a new SMS message based on the one provided\.\s*Here is the output\s*:?\s*',
            r'^ ,I can help you generate a new SMS message based on the one provided\.\s*Here is the output\s*:?\s*',
        ]

        for pattern in patterns:
            text = re.sub(
                pattern,
                '',
                text,
                count=1,
                flags=re.IGNORECASE
            )

        text = text.strip().strip('"').strip("'").strip()

        text = re.sub(r'\s*\n\s*', ' ', text)

        return text.strip()

    def augment_with_gemini(self, prompt, model_name, temp=0.8):
        try:
            response = self.gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    top_p=0.9, #Membuang kata/token acak diluar probabilitas kumulatif 90%
                )
            )
            return self.clean_llm_chatter(response.text)
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None

    def augment_with_gpt(self, prompt, model_name, temp=0.8):
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                top_p=0.9,
                frequency_penalty=1.1, #repetition penalty
            )
            return self.clean_llm_chatter(response.choices[0].message.content)
        except Exception as e:
            print(f"GPT Error: {e}")
            return None
        
    def augment_with_ollama(self, prompt, model_name, temp=0.8):
        try:
            response = ollama.chat(
                model=model_name, 
                messages=[{'role': 'user', 'content': prompt}],
                keep_alive='2h', 
                options={
                    'temperature': temp,
                    'top_p': 0.9,           
                    'repeat_penalty': 1.1,  # repetition penalty
                    # 'num_ctx': 1024,
                }
            ) 
            return self.clean_llm_chatter(response['message']['content'])
        except Exception as e:
            print(f"LLaMA Error [{model_name}]: {e}")
            return None