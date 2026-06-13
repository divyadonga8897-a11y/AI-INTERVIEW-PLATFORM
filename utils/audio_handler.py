import os
import tempfile
from gtts import gTTS
from utils.groq_client import transcribe_audio

def generate_tts_audio(text: str, cache_dir: str = "assets/audio") -> str:
    """
    Converts text to speech using gTTS and saves it as an MP3 file.
    Returns the absolute path of the generated audio file.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique filename based on hash of text to avoid regenerating if possible
    import hashlib
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    filename = f"q_{text_hash}.mp3"
    filepath = os.path.join(cache_dir, filename)
    
    # If already exists, reuse it
    if os.path.exists(filepath):
        return filepath
        
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"gTTS Speech synthesis failed: {e}")
        # Return empty path to handle failure gracefully in UI
        return ""

def process_and_transcribe_mic(audio_bytes, api_key: str = None, provider: str = "groq") -> str:
    """
    Saves raw audio bytes from st.audio_input to a temp file and sends it to Groq/OpenAI Whisper.
    Returns the transcribed text.
    """
    if not audio_bytes:
        return ""
        
    try:
        # Save bytes to a temporary file with standard suffix so Whisper API parses it correctly
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
            
        try:
            # Transcribe
            transcript = transcribe_audio(temp_file_path, api_key=api_key, provider=provider)
            return transcript
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    except Exception as e:
        print(f"Error processing audio recording: {e}")
        return ""
