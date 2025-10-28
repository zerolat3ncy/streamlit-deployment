import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os
import soundfile as sf

st.set_page_config(page_title="Swahili ASR & Translation", layout="wide")

st.title("Nyanja Speech Recognition and Translation")

@st.cache_resource
def load_models():
    whisper_model = WhisperModel("zerolat3ncy/faster-whisper-small-nya-tknEN", device="cpu", compute_type="int8")
    translation_model = pipeline("translation", model="helsinki-nlp/opus-mt-ny-en", device="cpu")
    return whisper_model, translation_model

def convert_audio(audio_bytes):
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name
    
    audio_data, sr = librosa.load(temp_path, sr=16000)
    
    with NamedTemporaryFile(delete=False, suffix=".wav") as converted_file:
        sf.write(converted_file.name, audio_data, sr, format='WAV')
        converted_path = converted_file.name
    
    os.unlink(temp_path)
    return converted_path

def transcribe(model, audio_path):
    segments, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    
    return transcription.strip()

def translate(model, text):
    if not text or text.strip() == "":
        return ""
    
    if len(text.split()) > 100:
        text = " ".join(text.split()[:100])
    
    result = model(text)
    return result[0]['translation_text']

def calculate_accuracy(reference, hypothesis):
    if not reference or not hypothesis:
        return None
    
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    matches = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    accuracy = (matches / max(len(ref_words), len(hyp_words))) * 100
    
    return accuracy

whisper_model, translation_model = load_models()

input_method = st.radio("Input method:", ["Upload Audio File", "Record Audio"])

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload Nyanja audio", type=['wav', 'mp3', 'ogg', 'flac', 'm4a'])
    if uploaded_file is not None:
        st.audio(uploaded_file)
        audio_data = uploaded_file
else:
    recorded_audio = st.audio_input("Record Nyanja audio")
    if recorded_audio is not None:
        st.audio(recorded_audio)
        audio_data = recorded_audio

col1, col2 = st.columns(2)

with col1:
    reference_nyanja = st.text_area("Reference Nyanja (optional)", placeholder="Enter correct Nyanja text")

with col2:
    reference_english = st.text_area("Reference English (optional)", placeholder="Enter correct English text")

if audio_data is not None:
    if st.button("Process", type="primary"):
        try:
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                if hasattr(audio_data, 'read'):
                    audio_bytes = audio_data.read()
                else:
                    audio_bytes = audio_data.getvalue()
                
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            if input_method == "Record Audio":
                converted_path = convert_audio(audio_bytes)
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                tmp_path = converted_path
            
            transcription = transcribe(whisper_model, tmp_path)
            translation = translate(translation_model, transcription)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Nyanja Transcription")
                st.write(transcription)
                
                if reference_nyanja.strip():
                    accuracy = calculate_accuracy(reference_nyanja, transcription)
                    if accuracy is not None:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
            
            with col_b:
                st.subheader("English Translation")
                st.write(translation)
                
                if reference_english.strip():
                    accuracy = calculate_accuracy(reference_english, translation)
                    if accuracy is not None:
                        st.metric("Accuracy", f"{accuracy:.1f}%")
            
            st.download_button("Download Transcription", transcription, "transcription.txt", "text/plain")
            st.download_button("Download Translation", translation, "translation.txt", "text/plain")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
