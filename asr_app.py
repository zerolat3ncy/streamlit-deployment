import streamlit as st
import torch
import torchaudio
from transformers import pipeline
import numpy as np
import io
from tempfile import NamedTemporaryFile
import os

# Set page config
st.set_page_config(page_title="ASR Demo", page_icon="🎙️")

st.title("🎙️ Automatic Speech Recognition Demo")
st.write("Upload an audio file or record your voice to transcribe it to text")

# Initialize ASR pipeline with caching
@st.cache_resource
def load_asr_model():
    """Load your custom ASR model"""
    try:
        # Example 1: Using a custom Hugging Face model
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_name = "dot1qprod/wav2vec2-2500-steps"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name)
        
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device="cpu"  # or "cuda" if available
        )
        return asr_pipeline
        
    except Exception as e:
        st.error(f"Error loading custom model: {e}")
        return None
# Load model
with st.spinner("Loading ASR model..."):
    asr_pipeline = load_asr_model()

if asr_pipeline is None:
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
                
                # Transcribe using ASR pipeline
                result = asr_pipeline(
                    tmp_path,
                    return_timestamps=False  # Set to True if you want word-level timestamps
                )
                
                # Display results
                st.success("Transcription completed!")
                st.subheader("📝 Transcription:")
                st.write(result["text"])
                
                # Add download button for transcription
                st.download_button(
                    label="Download Transcription as Text",
                    data=result["text"],
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")
            finally:
                # Clean up temporary file
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Add some information
with st.expander("ℹ️ About this ASR Demo"):
    st.write("""
    This demo uses a speech recognition model to convert audio to text.
    
    **Features:**
    - Upload audio files (WAV, MP3, OGG, FLAC, M4A)
    - Record audio directly in the browser
    - Download transcriptions as text files
    
    **Current model:** Whisper Tiny (replace with your custom ASR model)
    """)