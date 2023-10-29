import gradio as gr
from asr_openai import AutomaticSpeechRecognition
from tts_elevenlabs import ElevenLabsTTS
from falcon_7b_llm import Falcon_7b_llm
import logging
import os

logging.basicConfig(level=logging.INFO)

def delete_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            os.remove(file_path)

def generate_response(input_audio):
    sentence = asr.run_transcription(input_audio)
    # sentence = 'how are you?'
    print(sentence)
    llm_response = llm.get_llm_response(sentence['text'])
    output_audio = tts.tts_generate_audio(llm_response)
    # output_audio = tts.tts_generate_audio(sentence)
    chatbot_history.append(((input_audio,), (output_audio,)))
    return chatbot_history

delete_files_in_folder('data//tts_responses')

title = "<h1 style='text-align: center; color: #ffffff; font-size: 40px;'> Falcon Barista (Pre-Alpha Release)"

asr = AutomaticSpeechRecognition()
tts = ElevenLabsTTS()
llm = Falcon_7b_llm()
chatbot_history = []
order_display=False

def restart_chat():
    delete_files_in_folder('data//tts_responses')
    global chatbot_history
    chatbot_history = []
    tts.restart_state()
    llm.restart_state()
    return chatbot_history

def end_chat():
    delete_files_in_folder('data//tts_responses')
    global chatbot_history
    chatbot_history = []
    tts.restart_state()
    llm.restart_state()
    return chatbot_history

with gr.Blocks() as demo:
    gr.Markdown(title)
    order_title = gr.Markdown('### Your Order', visible=False)
    with gr.Row():
        gr.Image('https://i.imgur.com/fHCFI2T.png', label="Look how cute is Falcon Barista")
        with gr.Column():
            chatbot = gr.Chatbot(label='Chat with Falcon Barista', avatar_images=('data//user_avatar_logo.png','data//falcon_logo_transparent.png'), scale=2)
            mic = gr.Audio(source="microphone", type='filepath', scale=1)
            mic.stop_recording(generate_response, mic, chatbot)
    with gr.Row():
        restart_btn = gr.Button(value="Clear Chat", scale=1, variant='stop')
        restart_btn.click(restart_chat, outputs=[chatbot])
        end_btn = gr.Button(value="End Chat and Confirm Order", scale=2, variant='primary')
        end_btn.click(end_chat, outputs=[chatbot])
    
    order_title = gr.Markdown('### Your Order')
    order_summary = gr.DataFrame(
        headers=['item', 'quantity']
    )

if __name__ == "__main__":
    demo.launch()