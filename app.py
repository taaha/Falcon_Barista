import gradio as gr
import pandas as pd
from asr_openai import AutomaticSpeechRecognition
from tts_elevenlabs import ElevenLabsTTS
from falcon_7b_llm import Falcon_7b_llm
from order_parser import Order_Parser
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
    global order_dict
    try:
        order_dict = order_taking.order_parser(sentence['text'])
        print(order_dict)
    except Exception as e:
        print('order parsing failed')
        print(e)
    llm_response = llm.get_llm_response(sentence['text'])
    print(llm_response)
    output_audio = tts.tts_generate_audio(llm_response)
    # output_audio = tts.tts_generate_audio(sentence)
    chatbot_history.append(((input_audio,), (output_audio,)))
    return chatbot_history

delete_files_in_folder('data//tts_responses')

title = "<h1 style='text-align: center; color: #ffffff; font-size: 40px;'> Falcon Barista - Proof of Concept (POC)"

asr = AutomaticSpeechRecognition()
tts = ElevenLabsTTS()
llm = Falcon_7b_llm()
order_taking = Order_Parser()
chatbot_history = []
order_display=False
order_dict={}

df = pd.DataFrame({
    "item" : [], 
    "quantity" : [], 
}) 

s = df#.style.format("{:.2f}")

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown('''Note: This is just a POC and has several issues
1. High Latency
2. Models are still in fine tuning phase and get confused easily
                ''')
    order_title = gr.Markdown('### Your Order', visible=False)
    with gr.Row():
        gr.Image('https://i.imgur.com/fHCFI2T.png', label="Look how cute is Falcon Barista")
        with gr.Column():
            chatbot = gr.Chatbot(label='Chat with Falcon Barista', avatar_images=('data//user_avatar_logo.png','data//falcon_logo_transparent.png'), scale=2)
            mic = gr.Audio(source="microphone", type='filepath', scale=1)
            mic.stop_recording(generate_response, mic, chatbot)
    with gr.Row():
        restart_btn = gr.Button(value="Restart Chat", scale=1, variant='stop')
        # restart_btn.click(restart_chat, outputs=[chatbot])
        end_btn = gr.Button(value="End Chat and Confirm Order", scale=2, variant='primary')
        
    with gr.Column(visible=False) as output_col:
        order_title = gr.Markdown('### Your Order')
        order_summary = gr.DataFrame(s)

    def restart_chat():
        delete_files_in_folder('data//tts_responses')
        global chatbot_history
        chatbot_history = []
        global order_dict
        order_dict = {}
        global df
        df = pd.DataFrame({
            "item" : [], 
            "quantity" : [], 
        })
        order_taking.restart_state() 
        tts.restart_state()
        llm.restart_state()
        return {
            chatbot: [],
            output_col: gr.Column(visible=False)
        }

    def end_chat():
        df = pd.DataFrame(list(order_dict.items()), columns=['item', 'quantity'])

        s = df#.style.format("{:.2f}")
        return {
            output_col: gr.Column(visible=True),
            order_summary: gr.DataFrame(s, visible=True)
            }

    restart_btn.click(restart_chat, outputs=[chatbot, output_col])
    end_btn.click(end_chat, outputs=[output_col, order_summary])

if __name__ == "__main__":
    demo.launch()