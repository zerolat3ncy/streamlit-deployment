import streamlit as st
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os

# Set page config
st.set_page_config(page_title="ASR Demo", page_icon="🎙️")

st.title("🎙️ Automatic Speech Recognition Demo")
st.write("Upload an audio file or record your voice to transcribe it to text")

# Initialize ASR model with caching
@st.cache_resource
def load_asr_model():
    """Load your custom Wav2Vec2 model"""
    try:
        model_name = "dot1qprod/wav2vec2-2500-steps"
        
        # Load processor and model for Wav2Vec2
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        return processor, model
        
    except Exception as e:
        st.error(f"Error loading custom model: {e}")
        return None, None

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

# Load model
with st.spinner("Loading ASR model..."):
    processor, model = load_asr_model()

if processor is None or model is None:
    st.error("Failed to load ASR model. Please check your model configuration.")
    st.stop()

# Audio input methods
input_method = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])

audio_data = None

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload audio file", 
        type=['wav', 'mp3', 'ogg', 'flac', 'm4a'],
        help="Supported formats: WAV, MP3, OGG, FLAC, M4A"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        audio_data = uploaded_file

else:  # Record Audio
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio is not None:
        st.audio(recorded_audio, format='audio/wav')
        audio_data = recorded_audio

# Process audio when available
if audio_data is not None:
    if st.button("Transcribe Audio", type="primary"):
        with st.spinner("Transcribing audio..."):
            try:
                # Save uploaded audio to temporary file
                with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    if hasattr(audio_data, 'read'):
                        tmp_file.write(audio_data.read())
                    else:
                        tmp_file.write(audio_data.getvalue())
                    tmp_path = tmp_file.name
                
                # Transcribe using Wav2Vec2 model
                transcription = transcribe_audio(processor, model, tmp_path)
                
                # Display results
                if transcription.startswith("Error:"):
                    st.error(transcription)
                else:
                    st.success("Transcription completed!")
                    st.subheader("📝 Transcription:")
                    st.write(transcription)
                    
                    # Add download button for transcription
                    st.download_button(
                        label="Download Transcription as Text",
                        data=transcription,
                        file_name="transcription.txt",
                        mime="text/plain"
                    )
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
            finally:
                # Clean up temporary file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Add some information
with st.expander("ℹ️ About this ASR Demo"):
    st.write("""
    This demo uses a Wav2Vec2 speech recognition model to convert audio to text.
    
    **Features:**
    - Upload audio files (WAV, MP3, OGG, FLAC, M4A)
    - Record audio directly in the browser
    - Download transcriptions as text files
    
    **Current model:** dot1qprod/wav2vec2-2500-steps
    **Model type:** Wav2Vec2 (CTC-based)
    
    **Note:** The model works best with clear speech at 16kHz sampling rate.
    """)
