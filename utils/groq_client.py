import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_llm_client(api_key=None, provider="groq"):
    """
    Returns an initialized LLM client (Groq or OpenAI) based on the provider.
    If api_key is not provided, it attempts to load from the environment.
    """
    if provider == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in your env or sidebar.")
        return Groq(api_key=key), "llama-3.3-70b-versatile"
    else:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key not found. Please set OPENAI_API_KEY in your env or sidebar.")
        return OpenAI(api_key=key), "gpt-4o-mini"

def generate_chat_response(prompt: str, system_prompt: str = "", api_key: str = None, provider: str = "groq", json_mode: bool = False) -> str:
    """
    Generates a chat completion response from Groq Llama or OpenAI GPT.
    Supports JSON mode where applicable.
    """
    try:
        client, model = get_llm_client(api_key, provider)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Determine extra arguments (e.g. response_format for JSON mode)
        kwargs = {}
        if json_mode:
            if provider == "groq":
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {"type": "json_object"}
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            **kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Fallback error messaging
        return f"Error communicating with {provider.upper()} API: {str(e)}"

def transcribe_audio(audio_file_path: str, api_key: str = None, provider: str = "groq") -> str:
    """
    Transcribes an audio file using Groq Whisper (whisper-large-v3) or OpenAI Whisper (whisper-1).
    """
    try:
        client, _ = get_llm_client(api_key, provider)
        model = "whisper-large-v3" if provider == "groq" else "whisper-1"
        
        with open(audio_file_path, "rb") as file_obj:
            translation = client.audio.transcriptions.create(
                file=file_obj,
                model=model,
                response_format="text"
            )
            return translation
    except Exception as e:
        return f"Transcription error: {str(e)}"
