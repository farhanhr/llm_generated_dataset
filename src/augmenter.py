from google import genai
from google.genai import types
from openai import OpenAI
import ollama
import re
import time

class TextAugmenter:
    def __init__(self, gemini_key, openai_key, sumopod_key):
        # self.gemini_client = genai.Client(api_key=gemini_key)
        self.openai_client = OpenAI(api_key=openai_key)
        if sumopod_key:
            self.sumopod_client = OpenAI(
                api_key=sumopod_key,
                base_url="https://ai.sumopod.com/v1" 
            )

    def get_prompt(self, technique, text):
        if technique == "zero-shot":
            return (
                f"""
                OBJECTIVE:
                Buat satu SMS baru dalam bahasa indonesia berdasarkan SMS yang diberikan.

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
                Buat satu SMS baru dalam bahasa indonesia berdasarkan SMS yang diberikan.

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
                Buat satu SMS baru dalam bahasa indonesia berdasarkan SMS yang diberikan.

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

        text = re.sub(
            r'^```(?:text|sms|message)?\s*',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'\s*```$',
            '',
            text,
            flags=re.IGNORECASE
        )
        patterns = [
            r'^SMS\s*:\s*',
            r'^MESSAGE\s*:\s*',
            r'^TEXT\s*:\s*',
            r'^.*?\b(?:Generated SMS|Augmented SMS|augmentasi|Rewritten SMS|New SMS|SMS hasil augmentasi|Hasil SMS|Output SMS|Response|Here is the SMS|Berikut adalah SMS|Berikut hasil augmentasi|diberikan|input|provided)\s*:\s*["\']?\s*',


            r'^(?:output|result|response|answer|message)\s*:\s*',
            r'^(?:rewritten|rewritten sms|rewritten message)\s*:\s*',
            r'^(?:augmented|augmented sms|augmented message)\s*:\s*',
            r'^(?:generated|generated sms|generated message)\s*:\s*',
            r'^(?:new sms|new message)\s*:\s*',

            r'^(?:hasil|hasil augmentasi|hasil parafrase)\s*:\s*',
            r'^(?:output|respon|jawaban|pesan)\s*:\s*',
            r'^(?:sms baru|sms hasil|sms hasil augmentasi)\s*:\s*',
            r'^(?:teks baru|teks hasil|teks hasil augmentasi)\s*:\s*',

            r'^(?:output|result|response)\s+(?:sms|message)\s*:\s*',
            r'^(?:augmented|rewritten|generated)\s+(?:sms|message)\s*:\s*',
            r'^(?:hasil|output)\s+(?:sms|pesan)\s*:\s*',

            r'^(?:berikut(?: adalah)?|ini adalah)\s+(?:hasil(?:nya)?|output|respon|jawaban)\s*:\s*',
            r'^(?:berikut(?: adalah)?|ini adalah)\s+(?:sms|pesan)(?: baru| hasil)?\s*:\s*',
            r'^(?:berikut(?: adalah)?|ini)\s+(?:hasil augmentasi|hasil parafrase)\s*:\s*',
            r'^(?:berikut|ini)\s+(?:sms|pesan)(?: yang sudah)?\s+(?:diubah|diparafrase|diaugmentasi)\s*:\s*',

            r'^(?:here is|here are)\s+(?:the\s+)?(?:output|result|response|answer)\s*:\s*',
            r'^(?:here is|here are)\s+(?:the\s+)?(?:rewritten|augmented|generated)\s+(?:sms|message)\s*:\s*',
            r'^(?:here is|here are)\s+(?:a\s+)?(?:new|rewritten|augmented)\s+(?:sms|message)\s*:\s*',
            r'^(?:the\s+)?(?:rewritten|augmented|generated)\s+(?:sms|message)\s+is\s*:\s*',

            r'^(?:sure|certainly|absolutely|of course)\s*[,:!]?\s*',
            r'^(?:alright|all right)\s*[,:!]?\s*',
            r'^(?:yes|sure thing)\s*[,:!]?\s*',

            r'^(?:here\'s|here is)\s+(?:one|a|the)\s+(?:possible\s+)?(?:version|rewrite|example)\s*:\s*',
            r'^(?:here\'s|here is)\s+(?:the\s+)?(?:rewritten|augmented)\s+(?:version|text)\s*:\s*',

            r'^I can help you generate a new SMS message based on the one provided\.?\s*',
            r'^I can help you rewrite the SMS\.?\s*',
            r'^I can help you create a new SMS\.?\s*',
            r'^I\'d be happy to help(?: you)?\.?\s*',
            r'^I would be happy to help(?: you)?\.?\s*',

            r'^Here is the output you requested\.?\s*',
            r'^Here is the SMS you requested\.?\s*',
            r'^Here is the rewritten SMS you requested\.?\s*',
            r'^Here is the augmented SMS you requested\.?\s*',

            r'^Based on the provided SMS,?\s*',
            r'^Based on the original SMS,?\s*',
            r'^Based on your input,?\s*',
            r'^Based on the message provided,?\s*',

            r'^Berdasarkan SMS(?: yang diberikan)?,?\s*',
            r'^Berdasarkan pesan(?: yang diberikan)?,?\s*',
            r'^Berdasarkan input(?: yang diberikan)?,?\s*',
            r'^Dari SMS(?: yang diberikan)?,?\s*',
            r'^Dari pesan(?: yang diberikan)?,?\s*',

            r'^SMS\s*:\s*',
            r'^MESSAGE\s*:\s*',
            r'^TEXT\s*:\s*',
            r'^OUTPUT\s+SMS\s*:\s*',
            r'^RESULT\s+SMS\s*:\s*',

            r'^(?:sms)\s+(?:augmented|augmentasi|hasil|output|baru|generated|rewritten)\s*:?\s*',
            r'^(?:sms)\s+(?:augmented|augmentasi|hasil|output|baru|generated|rewritten)\s*:\s*',
            r'^(?:sms)\s+(?:output|result|response|answer)\s*:?\s*',

            r'^(?:output|result|response|answer|message)\s+(?:sms|message|text)\s*',
            r'^(?:augmented|rewritten|generated)\s+(?:sms|message|text)\s*',

            r'^(?:sms)\s+(?:yang\s+)?(?:sudah\s+)?(?:di)?(?:augmentasi|parafrase|ubah|modifikasi)\s*:?\s*',
            r'^(?:berikut)\s+(?:adalah\s+)?(?:sebuah\s+)?(?:sms|pesan)\s+(?:hasil\s+)?(?:augmentasi|parafrase)\s*:?\s*',
            r'^(?:berikut)\s+(?:adalah\s+)?(?:hasil\s+)?(?:augmentasi|parafrase)\s+(?:sms|pesan)\s*:?\s*',
            r'^(?:berikut)\s+(?:adalah\s+)?(?:sms|pesan)\s+(?:baru|hasil|yang\s+sudah\s+(?:diubah|diperbarui|diaugmentasi))\s*:?\s*',
            r'^(?:here\'s|here is)\s+(?:a\s+)?possible\s+(?:sms|message|output|result)\s*(?:in\s+english|in\s+bahasa\s+indonesia)?\s*(?:based\s+on\s+.*?)?:\s*',
            r'^(?:here\'s|here is)\s+(?:a\s+)?possible\s+(?:sms|message)\s+(?:in\s+bahasa\s+indonesia)\s*:?\s*',
            r'^(?:here\'s|here is)\s+(?:a\s+)?possible\s+(?:output|result)\s+(?:for\s+the\s+given\s+input)\s*:?\s*',
            r'^(?:here\'s|here is)\s+(?:the\s+)?(?:sms|message|output|result)\s+(?:based\s+on\s+the\s+given\s+input)\s*:?\s*',
            r'^(?:here\'s|here is)\s+(?:a\s+)?(?:new|rewritten|augmented)\s+(?:sms|message)\s+(?:based\s+on\s+.*?):\s*',
            r'^I can help you generate.*?(?:SMS|message).*?(?:provided|given)\.?\s*',
            r'^I can help you rewrite.*?(?:SMS|message).*?\s*',
            r'^I can help you create.*?(?:SMS|message).*?\s*',
            r'^Here is a possible.*?(?:SMS|message).*?:\s*',
            r'^Here\'s a possible.*?(?:SMS|message).*?:\s*',
            r'^Here is an? example.*?(?:SMS|message).*?:\s*',
            r'^Here\'s an? example.*?(?:SMS|message).*?:\s*',
            r'^(?:the\s+)?following\s+(?:is\s+the\s+)?(?:rewritten|augmented|generated|new)\s+(?:sms|message)\s*:?\s*',
            r'^(?:the\s+)?following\s+(?:sms|message)\s+(?:is|has\s+been)\s+(?:rewritten|augmented|generated)\s*:?\s*',
            r'^(?:berikut)\s+(?:contoh\s+)?(?:sms|pesan)\s+(?:yang\s+baru|baru|hasilnya?)\s*:?\s*',
            r'^(?:berikut)\s+(?:contoh\s+)?(?:hasil|output)\s+(?:sms|pesan)\s*:?\s*',
            r'\s+(?:saya)\s+(?:telah|sudah)?\s*(?:mengikuti|memenuhi|menerapkan)\s+(?:instruksi|ketentuan|persyaratan|pertahankan)\b.*$',
            r'\s+(?:saya)\s+(?:telah|sudah)?\s*(?:mempertahankan|menjaga|memenuhi)\s+(?:informasi|makna|gaya|karakteristik)\b.*$',
            r'\s+pertahankan\s+informasi\s+penting\s+yang\s+terdapat\s+dalam\s+(?:sms|pesan)\s+asli\b.*$',
            r'\s+gunakan\s+bahasa\s+yang\s+natural\s+dan\s+realistis\s+seperti\s+(?:sms|pesan)\s+yang\s+ditulis\s+manusia\b.*$',
            r'\s+jangan\s+(?:mengubah|merubah)\s+(?:makna|arti)\s+(?:utama|asli)\b.*$',
            r'\s+jangan\s+(?:menghasilkan|membuat)\s+(?:salinan|copy)\s+(?:yang\s+)?identik\b.*$',
            r'\s+Explain\s*:\s*.*$',
            r'\s+Explanation\s*:\s*.*$',
            r'\s+The original SMS contains\b.*$',
            r'\s+To augment the SMS\b.*$',
            r'\s+To augment the SMS,?\s+we\b.*$',
            r'\s+To augment the SMS,?\s+I\b.*$',
            r'\s+we maintained the same\b.*$',
            r'\s+we have maintained the same\b.*$',
            r'\s+This SMS maintains the same informative content as the original SMS.*$',
            r'\s+This SMS is a possible augmentation of the input SMS.*$',
            r'^(?:ayok|ayo|hey|hi|hello)\s*,?\s*here\'s\s+the\s+generated\s+sms\s+based\s+on\s+the\s+given\s+input\s*:\s*["\']?',        
            r'^here\'s\s+the\s+augmented\s+sms\s+based\s+on\s+the\s+given\s+input\s*:\s*["\']?',
            r'^berikut\s+ini\s+adalah\s+sms\s+yang\s+dihasilkan\s+setelah\s+menggunakan\s+teknik\s+augmentasi\s+sms\s*:\s*["\']?',
        ]
        for pattern in patterns:
            text = re.sub(
                pattern,
                '',
                text,
                count=1,
                flags=re.IGNORECASE
            )

        ending_patterns = [
            r'\s+I hope this helps!?\.?\s*$',
            r'\s+I hope this was helpful!?\.?\s*$',
            r'\s+Hope this helps!?\.?\s*$',
            r'\s+Let me know if you need anything else!?\.?\s*$',
            r'\s+Let me know if you need further assistance!?\.?\s*$',
            r'\s+Feel free to ask if you need anything else!?\.?\s*$',
            r'\s+I\'d be happy to help with anything else!?\.?\s*$',
            r'\s+Semoga ini membantu!?\.?\s*$',
            r'\s+Semoga membantu!?\.?\s*$',
            r'\s+Jika ada pertanyaan.*$',
            r'\s+Silakan beri tahu.*$',
            r'\s+Jika membutuhkan bantuan.*$',
            r'\s+I hope you find this helpful!?\.?\s*$',
            r'\s+I hope this response helps!?\.?\s*$',
            r'\s+Hope this is helpful!?\.?\s*$',
            r'\s+I hope the rewritten SMS helps!?\.?\s*$',
            r'\s+I hope the augmented SMS helps!?\.?\s*$',
            r'\s+Please let me know if.*$',
            r'\s+Let me know if.*$',
            r'\s+Feel free to let me know.*$',
            r'\s+If you have any questions.*$',
            r'\s+If you need any further assistance.*$',
            r'\s+Semoga SMS ini membantu!?\.?\s*$',
            r'\s+Semoga hasil augmentasi ini membantu!?\.?\s*$',
            r'\s+Semoga hasilnya membantu!?\.?\s*$',
            r'\s+Semoga pesan ini membantu!?\.?\s*$',
            r'\s+Jika ada hal lain yang.*$',
            r'\s+Jika ada yang ingin ditanyakan.*$',
            r'\s+Jika ada pertanyaan lebih lanjut.*$',
            r'\s+Explain\s*:\s*.*$',
            r'\s+Explanation\s*:\s*.*$',
            r'\s+The original SMS contains\b.*$',
            r'\s+To augment the SMS\b.*$',
            r'\s+To augment the SMS,?\s+we\b.*$',
            r'\s+To augment the SMS,?\s+I\b.*$',
            r'\s+we maintained the same\b.*$',
            r'\s+we have maintained the same\b.*$',
            r'\s+Translation\s*:\s*.*$',
            r'\s+Keterangan\s*:\s*.*$',
            r'\s+This SMS maintains the same informative content as the original SMS.*$',
            r'\s+This SMS is a possible augmentation of the input SMS.*$',
            r'\s*(?:Explanation|Explain|Keterangan)\s*:\s*.*$'
            r'\s*(?:Explanation|Explain|Penjelasan|Keterangan)\s*:\s*.*$',
        ]

        for pattern in ending_patterns:
            text = re.sub(
                pattern,
                '',
                text,
                count=1,
                flags=re.IGNORECASE
            )


        text = re.sub(r'\s*\n\s*', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        text = text.strip()

        if len(text) >= 2:
            if text[0] == '"' and text[-1] == '"':
                text = text[1:-1].strip()

            elif text[0] == "'" and text[-1] == "'":
                text = text[1:-1].strip()

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
                keep_alive='3h', 
                options={
                    'temperature': temp,
                    'top_p': 0.9,           
                    'repeat_penalty': 1.1,  # repetition penalty
                    # 'num_predict': 150,
                    # 'num_ctx': 1024,
                }
            ) 
            return self.clean_llm_chatter(response['message']['content'])
        except Exception as e:
            print(f"LLaMA Error [{model_name}]: {e}")
            return None

    def augment_with_sumopod(self, prompt, model_name, temp=0.8, max_retries=3):
        import time
        for attempt in range(max_retries):
            try:
                response = self.sumopod_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    top_p=0.9,
                )
                return self.clean_llm_chatter(response.choices[0].message.content)
                
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg:
                    wait_time = (2 ** attempt) * 2
                    print(f"Sumopod Limit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Sumopod Error: {e}")
                    return None
                    
        return None