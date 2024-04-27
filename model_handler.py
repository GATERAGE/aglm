import os
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from aglm import LlamaModel

class Model_Type(Enum):
    gptq = 1
    ggml = 2
    full_precision = 3

def get_model_type(model_name):
    if "gptq" in model_name.lower():
        return Model_Type.gptq
    elif "ggml" in model_name.lower():
        return Model_Type.ggml
    else:
        return Model_Type.full_precision

def create_folder_if_not_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)

def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = LlamaModel(model_name)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using default for simplicity
    return model, tokenizer

def init_auto_model_and_tokenizer(model_name, model_type, file_name, config):
    if model_type == Model_Type.ggml:
        models_folder = config["models_folder"]
        create_folder_if_not_exists(models_folder)
        file_path = hf_hub_download(repo_id=model_name, filename=file_name, local_dir=models_folder)
        model = LlamaModel(file_path)
        tokenizer = AutoTokenizer.from_pretrained(config["default_tokenizer"])
    else:
        model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type)
    return model, tokenizer
