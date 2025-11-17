import os
import sys
import sqlite3
import time
import numpy as np
import sounddevice as sd
import pyttsx3
import whisper
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------
# Configuración inicial
# -----------------------
SISCO_NOMBRE = "Sisco"
CREADOR = "Edy Fernando Cjumo Mejia"
DB_FILE = "sisco_memoria.db"

# Inicializar TTS
tts = pyttsx3.init()
tts.setProperty('rate', 175)
tts.setProperty('volume', 0.9)
voices = tts.getProperty('voices')
for voice in voices:
    if 'spanish' in voice.name.lower() or 'es' in voice.languages:
        tts.setProperty('voice', voice.id)
        break

# Inicializar Whisper
whisper_model = whisper.load_model("base")

# -----------------------
# Base de datos SQLite
# -----------------------
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memoria(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario TEXT,
    texto TEXT,
    embedding BLOB,
    fecha TIMESTAMP
)
""")
conn.commit()

# -----------------------
# Modelo de embeddings
# -----------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# Cargar LLM local
# -----------------------
MODEL_NAME = "TheBloke/LLaMA-3B-GPTQ"  # Cambiar según modelo disponible
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=300)

# -----------------------
# Funciones
# -----------------------
def tts_speak(texto):
    tts.say(texto)
    tts.runAndWait()

def grabar_audio(segundos=10, fs=44100):
    print(f"🎤 Habla ahora ({segundos}s)...")
    audio = sd.rec(int(segundos*fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def transcribir_audio(audio, fs=44100):
    wav_file = "temp.wav"
    from scipy.io.wavfile import write
    audio_int = np.int16(audio * 32767)
    write(wav_file, fs, audio_int)
    result = whisper_model.transcribe(wav_file, language='es', fp16=False)
    return result["text"].strip()

def guardar_memoria(usuario, texto):
    embedding = embedding_model.encode(texto).astype('float32')
    cursor.execute("INSERT INTO memoria(usuario,texto,embedding,fecha) VALUES(?,?,?,?)",
                   (usuario, texto, embedding.tobytes(), datetime.now()))
    conn.commit()

def buscar_contexto(texto, top_k=3):
    embedding = embedding_model.encode(texto).astype('float32')
    cursor.execute("SELECT texto, embedding FROM memoria")
    rows = cursor.fetchall()
    if not rows:
        return []
    embeddings = np.array([np.frombuffer(r[1], dtype='float32') for r in rows])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(np.array([embedding]), top_k)
    contextos = [rows[i][0] for i in I[0]]
    return contextos

def generar_respuesta(prompt, contexto=[]):
    contexto_texto = "\n".join(contexto)
    full_prompt = f"Soy {SISCO_NOMBRE}, un asistente conversacional en español. Mi creador es {CREADOR}.\n"
    full_prompt += f"Contexto previo:\n{contexto_texto}\nUsuario: {prompt}\n{SISCO_NOMBRE}:"
    salida = generator(full_prompt, max_length=300, do_sample=True, temperature=0.7)
    respuesta = salida[0]['generated_text'].split(f"{SISCO_NOMBRE}:")[-1].strip()
    return respuesta

# -----------------------
# Ciclo principal
# -----------------------
def chat_voz():
    while True:
        audio = grabar_audio(5)
        texto = transcribir_audio(audio)
        if texto.lower() in ["salir", "adiós", "chao"]:
            print(f"{SISCO_NOMBRE}: ¡Hasta luego, {CREADOR}!")
            tts_speak(f"¡Hasta luego, {CREADOR}!")
            break
        guardar_memoria(CREADOR, texto)
        contexto = buscar_contexto(texto)
        respuesta = generar_respuesta(texto, contexto)
        print(f"{SISCO_NOMBRE}: {respuesta}")
        tts_speak(respuesta)
        guardar_memoria(SISCO_NOMBRE, respuesta)

def chat_texto():
    print(f"\n💬 Chat por texto con {SISCO_NOMBRE} (Escribe 'salir' para volver al menú)")
    while True:
        texto = input("👤 Tú: ").strip()
        if texto.lower() in ["salir", "adiós", "chao"]:
            print(f"{SISCO_NOMBRE}: ¡Hasta luego, {CREADOR}!")
            break
        guardar_memoria(CREADOR, texto)
        contexto = buscar_contexto(texto)
        respuesta = generar_respuesta(texto, contexto)
        print(f"{SISCO_NOMBRE}: {respuesta}")
        guardar_memoria(SISCO_NOMBRE, respuesta)

# -----------------------
# Menú principal
# -----------------------
def menu():
    while True:
        print("\n" + "="*60)
        print(f"🤖 Bienvenido a {SISCO_NOMBRE} 2.0")
        print("="*60)
        print("1. 💬 Chat por texto")
        print("2. 🎤 Chat por voz")
        print("3. 🚪 Salir")
        opcion = input("👉 Opción (1-3): ").strip()
        if opcion == "1":
            chat_texto()
        elif opcion == "2":
            chat_voz()
        elif opcion == "3":
            print(f"👋 {SISCO_NOMBRE} te dice hasta luego, {CREADOR}!")
            break
        else:
            print("❌ Opción inválida.")

if __name__ == "__main__":
    menu()
