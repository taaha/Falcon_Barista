import elevenlabs
from elevenlabs import generate, save
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
elevenlabs.set_api_key(elevenlabs_api_key)

class ElevenLabsTTS():
    """
    Class for Eleven Labs TTS.

    This class uses elevenlab free tier to give TTS response

    Args:
        None
    """
    def __init__(self):
        self.response_number = 0
        pass

    def restart_state(self):
        self.response_number = 0        

    def tts_generate_audio(self, input):
        audio = generate(text=input, voice="Giovanni")
        self.response_number = self.response_number + 1
        file_path = f"data//tts_responses//test_{self.response_number}.wav"
        save(
            audio,               # Audio bytes (returned by generate)
            file_path
        )
        return file_path