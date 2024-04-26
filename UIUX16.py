import argparse
import os
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Importing custom modules
from aglm import LlamaModel
from memory import save_conversation_memory

class Model_Type(Enum):
    gptq = 1
    ggml = 2
    gguf = 3
    full_precision = 4

def get_model_type(model_name):
    if "gptq" in model_name.lower():
        return Model_Type.gptq
    elif "ggml" in model_name.lower():
        return Model_Type.ggml
    elif "gguf" in model_name.lower():
        return Model_Type.gguf
    else:
        return Model_Type.full_precision

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def initialize_model(model_name, model_type):
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    try:
        if model_type == Model_Type.ggml:
            filename = "pytorch_model.bin"
            file_path = hf_hub_download(repo_id=model_name, filename=filename, local_dir=models_folder)
            model = LlamaModel(file_path, n_ctx=4096)
        elif model_type == Model_Type.gguf:
            filename = "flax_model.msgpack"
            file_path = hf_hub_download(repo_id=model_name, filename=filename, local_dir=models_folder)
            model = LlamaModel(file_path, n_ctx=4096)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name} with type {model_type}: {e}")
        raise
    return model, tokenizer

def run_ui(model, tokenizer, is_chat_model, model_type, save_history=True):
    from gradio.interface import Interface
    
    def chat_function(input_text):
        inputs = tokenizer(input_text, return_tensors='pt')
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if save_history:
            save_conversation_memory(model_name=model.name, user_input=input_text, model_response=response)
        return response
    
    iface = Interface(fn=chat_function, inputs="text", outputs="text", title="Chat with AI")
    iface.launch(share=True)

def main():
    parser = argparse.ArgumentParser(description="Run the AI model")
    parser.add_argument("--model_name", required=True, help="Model name on Hugging Face Hub")
    parser.add_argument("--tokenizer_name", required=True, help="Tokenizer name on Hugging Face Hub")
    parser.add_argument("--model_type", required=True, help="Type of the model, if applicable")
    parser.add_argument("--save_history", action='store_true', help="Flag to determine if history should be saved")
    args = parser.parse_args()
    
    model_type = get_model_type(args.model_name)
    model, tokenizer = initialize_model(args.model_name, model_type)
    run_ui(model, tokenizer, 'chat' in args.model_name.lower(), model_type, args.save_history)

if __name__ == '__main__':
    main()
