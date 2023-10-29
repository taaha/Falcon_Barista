from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
import os
import runpod
from dotenv import load_dotenv
from langchain.llms import HuggingFaceTextGenInference
from langchain.schema import BaseOutputParser
import re
import re
from typing import List
from langchain.schema import BaseOutputParser
import torch
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
)

# Load the .env file
load_dotenv()

# Get the API key from the environment variable
runpod.api_key = os.getenv("RUNPOD_API_KEY")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "falcon_hackathon"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
pod_id = os.getenv("POD_ID")

class CleanupOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        user_pattern = r"\nUser"
        text = re.sub(user_pattern, "", text)
        human_pattern = r"\nHuman:"
        text = re.sub(human_pattern, "", text)
        ai_pattern = r"\nAI:"
        return re.sub(ai_pattern, "", text).strip()
 
    @property
    def _type(self) -> str:
        return "output_parser"
    
class StopGenerationCriteria(StoppingCriteria):
    def __init__(
        self, tokens: List[List[str]], tokenizer: AutoTokenizer, device: torch.device
    ):
        stop_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
        self.stop_token_ids = [
            torch.tensor(x, dtype=torch.long, device=device) for x in stop_token_ids
        ]
 
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False



class Falcon_7b_llm():
    def __init__(self):
        inference_server_url_cloud = f"https://{pod_id}-80.proxy.runpod.net"

        template = """You are a chatbot called 'Falcon Barista' working at a coffee shop. 
Your primary function is to take orders from customers. 
Start with a greeting.
You have the following menu with prices. Dont mention the price unless asked. Do not take order for anything other than in menu.
- cappucino-5$
- latte-3$
- frappucino-8$
- juice-3$ 
If user orders something else, apologise that you dont have that item.
Take the order politely and in a frienldy way. After that confirm the order, tell the order price and say "Goodbye have a nice day".

{chat_history}
Human: {human_input}
AI:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_history")

        llm_cloud = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_cloud,
            max_new_tokens=200,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.0,
            stop_sequences = ['Mini', 'AI', 'Human', ':']
        )

        self.llm_chain_cloud = ConversationChain(
                prompt=prompt, 
                llm=llm_cloud,
                verbose=True,
                memory=memory,
                output_parser=CleanupOutputParser(),
                input_key='human_input'
                )
        
    def restart_state(self):
        inference_server_url_cloud = f"https://{pod_id}-80.proxy.runpod.net"

        template = """You are a chatbot called 'Falcon Barista' working at a coffee shop. 
Your primary function is to take orders from customers. 
Start with a greeting.
You have the following menu with prices. Dont mention the price unless asked. Do not take order for anything other than in menu.
- cappucino-5$
- latte-3$
- frappucino-8$
- juice-3$ 
If user orders something else, apologise that you dont have that item.
Take the order politely and in a frienldy way. After that confirm the order, tell the order price and say "Goodbye have a nice day".

{chat_history}
Human: {human_input}
AI:"""

        prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=template
        )
        memory = ConversationBufferMemory(memory_key="chat_history")

        llm_cloud = HuggingFaceTextGenInference(
            inference_server_url=inference_server_url_cloud,
            max_new_tokens=200,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.0,
            stop_sequences = ['Mini', 'AI', 'Human', ':']
        )

        self.llm_chain_cloud = ConversationChain(
                prompt=prompt, 
                llm=llm_cloud,
                verbose=True,
                memory=memory,
                output_parser=CleanupOutputParser(),
                input_key='human_input'
                )
        
    def get_llm_response(self, human_input):
        completion = self.llm_chain_cloud.predict(human_input=human_input)
        return completion