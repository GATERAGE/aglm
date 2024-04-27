# uiux.py
import os
import gradio as gr
import fire
import time
import pathlib
import ujson
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from automind import format_to_llama_chat_style
from memory import save_conversation_memory
import logging

# Configuration and Constants
MODEL_CONFIG = {
    "default_tokenizer": "gpt2",
    "models_folder": "./models"
}

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Enum for model types
class Model_Type(Enum):
    gptq = 1
    ggml = 2
    full_precision = 3

# Helper functions
def get_model_type(model_name):
    if "gptq" in model_name.lower():
        return Model_Type.gptq
    elif "ggml" in model_name.lower():
        return Model_Type.ggml
    else:
        return Model_Type.full_precision

def create_folder_if_not_exists(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {folder_path}: {e}")
        raise

def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
        model = AutoGPTQForCausalLM.from_quantized(model_name, device_map="auto", use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def init_auto_model_and_tokenizer(model_name, model_type, file_name=None):
    model_type = get_model_type(model_name)
    if model_type == Model_Type.ggml:
        models_folder = MODEL_CONFIG["models_folder"]
        create_folder_if_not_exists(models_folder)
        file_path = hf_hub_download(repo_id=model_name, filename=file_name, local_dir=models_folder)
        model = Llama(file_path)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["default_tokenizer"])
    else:
        model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type)
    return model, tokenizer

def run_ui(model, tokenizer, is_chat_model, model_type, save_history=True):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        conversation_memory = []

        def user(user_message, memory):
            nonlocal conversation_memory
            conversation_memory = memory + [[user_message, None]]
            return "", conversation_memory

        def bot(memory):
            nonlocal conversation_memory
            if is_chat_model:
                instruction = format_to_llama_chat_style(memory)
            else:
                instruction = memory[-1][0]

            memory[-1][1] = ""
            try:
                if model_type == Model_Type.ggml:
                    for chunk in model(prompt=instruction, stream=True):
                        token = chunk["choices"][0]["text"]
                        memory[-1][1] += token
                else:
                    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
                    outputs = model.generate(input_ids=inputs["input_ids"])
                    memory[-1][1] = tokenizer.decode(outputs[0], skip_special_tokens=True)

                yield memory

                if save_history:
                    save_conversation_memory(conversation_memory)
            except ValueError as e:
                if "Requested tokens exceed context window of 4096" in str(e):
                    memory[-1][1] = "Error: Instruction too long to process."
                    yield memory
                else:
                    logging.exception("Unhandled exception occurred during bot generation.")
                    raise

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.queue()
    demo.launch(share=False, debug=True)

# Main function
def main(model_name=None, file_name=None, save_history=True):
    assert model_name is not None, "model_name argument is missing."
    is_chat_model = 'chat' in model_name.lower()
    model_type = get_model_type(model_name)
    model, tokenizer = init_auto_model_and_tokenizer(model_name, model_type, file_name)
    run_ui(model, tokenizer, is_chat_model, model_type, save_history=save_history)

if __name__ == '__main__':
    fire.Fire(main)

