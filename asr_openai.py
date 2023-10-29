import openai
import os
import time
import logging
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class AutomaticSpeechRecognition():
    """
    Class for automatic speech recognition(ASR).

    This class uses faster whisper model for low latency ASR

    Args:
        model_size: size of model (small, base, etc.)
    """
    def __init__(self):
        pass

    def run_transcription(self, filepath):
        audio_file= open(filepath, "rb")
        sentence = openai.Audio.transcribe("whisper-1", audio_file)
        
        logging.debug(f'transcription: {sentence}')
        
        return sentence