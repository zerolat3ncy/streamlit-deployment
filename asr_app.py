import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os
import jiwer
from jiwer import wer
import soundfile as sf

# Set page config
st.set_page_config(page_title="Swahili ASR & Translation", layout="wide")

st.title("Swahili Speech Recognition and Translation System")
st.write("Upload a Swahili audio file or record audio for transcription and translation to English")

# Initialize ASR model with caching
@st.cache_resource
def load_asr_model():
    """Load Wav2Vec2 model for Swahili speech recognition"""
    try:
        model_name = "dot1qprod/wav2vec2-2500-steps"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        st.error(f"Error loading ASR model: {e}")
        return None, None

# Initialize Translation model with caching
@st.cache_resource
def load_translation_model():
    """Load Swahili to English translation model"""
    try:
        translation_pipeline = pipeline(
            "translation",
            model="Rogendo/sw-en",
            device="cpu"
        )
        return translation_pipeline
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None

def convert_audio_format(audio_bytes):
    """Convert recorded audio to proper WAV format"""
    try:
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            temp_path = tmp_file.name
        
        audio_data, sr = librosa.load(temp_path, sr=16000)
        
        with NamedTemporaryFile(delete=False, suffix=".wav") as converted_file:
            sf.write(converted_file.name, audio_data, sr, format='WAV')
            converted_path = converted_file.name
        
        os.unlink(temp_path)
        return converted_path, sr
        
    except Exception as e:
        st.error(f"Error converting audio: {e}")
        return None, None

def preprocess_audio(audio_path, target_sr=16000):
    """Preprocess audio for Wav2Vec2 model"""
    try:
        speech, sr = librosa.load(audio_path, sr=target_sr)
        
        audio_duration = len(speech) / sr
        if audio_duration > 30:
            st.warning(f"Audio truncated to 30 seconds (original: {audio_duration:.1f}s)")
            speech = speech[:target_sr * 30]
        
        if np.max(np.abs(speech)) > 0:
            speech = speech / np.max(np.abs(speech))
        
        return speech, sr
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None

def transcribe_audio(processor, model, audio_path):
    """Transcribe audio using Wav2Vec2 model"""
    try:
        speech, sr = preprocess_audio(audio_path)
        if speech is None:
            return "Error: Could not process audio file", None
        
        inputs = processor(
            speech, 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription, speech
        
    except Exception as e:
        return f"Error during transcription: {str(e)}", None

def translate_text(translation_pipeline, text):
    """Translate Swahili text to English"""
    try:
        if not text or text.strip() == "":
            return "No text to translate", None
        
        if len(text.split()) > 100:
            text = " ".join(text.split()[:100])
            st.warning("Text truncated to 100 words for faster processing")
        
        result = translation_pipeline(text)
        translated_text = result[0]['translation_text']
        
        return translated_text, result[0]
    except Exception as e:
        return f"Error during translation: {str(e)}", None

def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate between reference and hypothesis text"""
    try:
        if not reference or not hypothesis:
            return None, "Missing reference or hypothesis text"
        
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.RemovePunctuation(),
            jiwer.Strip(),
        ])
        
        reference_clean = transformation(reference)
        hypothesis_clean = transformation(hypothesis)
        
        error_rate = wer(reference_clean, hypothesis_clean)
        word_accuracy = max(0, (1 - error_rate) * 100)
        
        return error_rate, word_accuracy
        
    except Exception as e:
        return None, f"Error calculating WER: {str(e)}"

# Load models
with st.spinner("Loading ASR model..."):
    processor, model = load_asr_model()

with st.spinner("Loading translation model..."):
    translation_pipeline = load_translation_model()

if processor is None or model is None:
    st.error("Failed to load ASR model")
    st.stop()

if translation_pipeline is None:
    st.error("Failed to load translation model")
    st.stop()

# Audio input methods
input_method = st.radio("Input method:", ["Upload Audio File", "Record Audio"])

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload Swahili audio file", 
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, OGG, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio_data = uploaded_file

else:
    recorded_audio = st.audio_input("Record Swahili audio")
    if recorded_audio is not None:
        st.audio(recorded_audio, format='audio/wav')
        audio_data = recorded_audio

# Reference text input for WER calculation
st.subheader("Evaluation Metrics")
col_ref1, col_ref2 = st.columns(2)

with col_ref1:
    reference_swahili = st.text_area(
        "Reference Swahili Text:",
        placeholder="Enter correct Swahili transcription for WER calculation",
        help="Optional: Provide reference text to calculate Word Error Rate"
    )

with col_ref2:
    reference_english = st.text_area(
        "Reference English Translation:",
        placeholder="Enter correct English translation for WER calculation",
        help="Optional: Provide reference translation to calculate Word Error Rate"
    )

# Process audio when available
if audio_data is not None:
    if st.button("Process Audio", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Processing audio file...")
            progress_bar.progress(25)
            
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                if hasattr(audio_data, 'read'):
                    audio_bytes = audio_data.read()
                else:
                    audio_bytes = audio_data.getvalue()
                
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            if input_method == "Record Audio":
                status_text.text("Converting audio format...")
                progress_bar.progress(50)
                converted_path, sr = convert_audio_format(audio_bytes)
                if converted_path:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    tmp_path = converted_path
            
            status_text.text("Transcribing audio...")
            progress_bar.progress(75)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Swahili Transcription")
                transcription, audio_signal = transcribe_audio(processor, model, tmp_path)
                
                if transcription.startswith("Error:"):
                    st.error(transcription)
                else:
                    st.success("Transcription completed")
                    st.write(transcription)
                    
                    if reference_swahili.strip():
                        wer_score, word_accuracy = calculate_wer(reference_swahili, transcription)
                        if wer_score is not None:
                            st.metric("Word Error Rate (Swahili)", f"{wer_score:.3f}")
                            st.metric("Word Accuracy (Swahili)", f"{word_accuracy:.1f}%")
            
            status_text.text("Translating to English...")
            progress_bar.progress(90)
            
            with col2:
                st.subheader("English Translation")
                if not transcription.startswith("Error:"):
                    translation, translation_result = translate_text(translation_pipeline, transcription)
                    
                    if translation.startswith("Error:"):
                        st.error(translation)
                    else:
                        st.success("Translation completed")
                        st.write(translation)
                        
                        if reference_english.strip():
                            wer_score, word_accuracy = calculate_wer(reference_english, translation)
                            if wer_score is not None:
                                st.metric("Word Error Rate (English)", f"{wer_score:.3f}")
                                st.metric("Word Accuracy (English)", f"{word_accuracy:.1f}%")
            
            progress_bar.progress(100)
            status_text.text("Processing completed")
            
            st.subheader("Download Results")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                st.download_button(
                    label="Download Swahili Transcription",
                    data=transcription,
                    file_name="swahili_transcription.txt",
                    mime="text/plain"
                )
            
            with col_dl2:
                if not transcription.startswith("Error:"):
                    st.download_button(
                        label="Download English Translation",
                        data=translation,
                        file_name="english_translation.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            progress_bar.empty()
            status_text.empty()

# System information
with st.expander("System Information"):
    st.write("""
    This system provides Swahili speech recognition and translation to English.
    
    Models:
    - ASR: dot1qprod/wav2vec2-2500-steps (Swahili speech recognition)
    - Translation: Rogendo/sw-en (Swahili to English)
    
    Word Error Rate (WER) measures transcription and translation accuracy.
    Lower WER values indicate better performance.
    
    For optimal results:
    - Use clear audio recordings
    - Limit audio duration to 5-15 seconds
    - WAV format at 16kHz sampling rate recommended
    """)
