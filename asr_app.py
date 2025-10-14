import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os

# Set page config
st.set_page_config(page_title="Swahili ASR & Translation", page_icon="🎙️")

st.title("🎙️ Swahili Speech Recognition & Translation")
st.write("Upload a Swahili audio file or record your voice to transcribe and translate to English")

# Initialize ASR model with caching
@st.cache_resource
def load_asr_model():
    """Load your custom Wav2Vec2 model for Swahili"""
    try:
        model_name = "dot1qprod/wav2vec2-2500-steps"
        
        # Load processor and model for Wav2Vec2
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
        # Using Rogendo/sw-en model for Swahili to English translation
        translation_pipeline = pipeline(
            "translation",
            model="Rogendo/sw-en",  # Swahili to English model
            device="cpu"
        )
        return translation_pipeline
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        return None

def preprocess_audio(audio_path, target_sr=16000):
    """Preprocess audio to match Wav2Vec2 requirements"""
    try:
        # Load audio with librosa (handles various formats)
        speech, sr = librosa.load(audio_path, sr=target_sr)
        
        # Normalize audio
        if np.max(np.abs(speech)) > 0:
            speech = speech / np.max(np.abs(speech))
        
        return speech, sr
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None

def transcribe_audio(processor, model, audio_path):
    """Transcribe audio using Wav2Vec2 model"""
    try:
        # Preprocess audio
        speech, sr = preprocess_audio(audio_path)
        if speech is None:
            return "Error: Could not process audio file"
        
        # Process with Wav2Vec2
        inputs = processor(
            speech, 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get model predictions
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        # Get predicted tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode tokens to text
        transcription = processor.batch_decode(predicted_ids)[0]
        
        return transcription
        
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def translate_text(translation_pipeline, text):
    """Translate Swahili text to English"""
    try:
        if not text or text.strip() == "":
            return "No text to translate"
        
        # Perform translation
        result = translation_pipeline(text)
        translated_text = result[0]['translation_text']
        
        return translated_text
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Load models
with st.spinner("Loading ASR model..."):
    processor, model = load_asr_model()

with st.spinner("Loading translation model..."):
    translation_pipeline = load_translation_model()

if processor is None or model is None:
    st.error("Failed to load ASR model. Please check your model configuration.")
    st.stop()

if translation_pipeline is None:
    st.error("Failed to load translation model. Please check your model configuration.")
    st.stop()

# Audio input methods
input_method = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload Swahili audio file", 
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, OGG, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        audio_data = uploaded_file

else:  # Record Audio
    recorded_audio = st.audio_input("Record your voice (Speak Swahili)")
    if recorded_audio is not None:
        st.audio(recorded_audio, format='audio/wav')
        audio_data = recorded_audio

# Process audio when available
if audio_data is not None:
    if st.button("Transcribe & Translate", type="primary"):
        with st.spinner("Processing audio..."):
            try:
                # Save uploaded audio to temporary file
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    if hasattr(audio_data, 'read'):
                        tmp_file.write(audio_data.read())
                    else:
                        tmp_file.write(audio_data.getvalue())
                    tmp_path = tmp_file.name
                
                # Create columns for results
                col1, col2 = st.columns(2)
                
                # Transcribe using Wav2Vec2 model
                with col1:
                    st.subheader("🗣️ Swahili Transcription")
                    transcription = transcribe_audio(processor, model, tmp_path)
                    
                    if transcription.startswith("Error:"):
                        st.error(transcription)
                    else:
                        st.success("Transcription completed!")
                        st.write(transcription)
                
                # Translate to English
                with col2:
                    st.subheader("🌍 English Translation")
                    if not transcription.startswith("Error:"):
                        translation = translate_text(translation_pipeline, transcription)
                        
                        if translation.startswith("Error:"):
                            st.error(translation)
                        else:
                            st.success("Translation completed!")
                            st.write(translation)
                
                # Download buttons
                st.subheader("📥 Download Results")
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
                st.error(f"Error during processing: {str(e)}")
            finally:
                # Clean up temporary file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Add some information
with st.expander("ℹ️ About this ASR & Translation Demo"):
    st.write("""
    This demo uses a Wav2Vec2 speech recognition model for Swahili and machine translation to convert to English.
    
    **Features:**
    - Upload Swahili audio files (WAV, MP3, OGG, FLAC, M4A)
    - Record Swahili audio directly in the browser
    - Automatic transcription of Swahili speech
    - Machine translation from Swahili to English
    - Download both transcription and translation
    
    **Models Used:**
    - **ASR Model:** dot1qprod/wav2vec2-2500-steps (Swahili speech recognition)
    - **Translation Model:** Rogendo/sw-en (Swahili to English)
    
    **Note:** 
    - The ASR model works best with clear Swahili speech at 16kHz sampling rate
    - Translation quality depends on the clarity and grammar of the transcribed text
    """)

# Optional: Add a text input for direct translation
with st.expander("🔤 Direct Text Translation (Swahili to English)"):
    st.write("Have Swahili text you want to translate? Use the tool below:")
    
    swahili_text = st.text_area("Enter Swahili text:", placeholder="Andika kitu hapa...")
    
    if st.button("Translate Text"):
        if swahili_text.strip():
            with st.spinner("Translating..."):
                translation = translate_text(translation_pipeline, swahili_text)
                if not translation.startswith("Error:"):
                    st.subheader("English Translation:")
                    st.write(translation)
                    
                    st.download_button(
                        label="Download Translation",
                        data=translation,
                        file_name="text_translation.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(translation)
        else:
            st.warning("Please enter some Swahili text to translate.")
