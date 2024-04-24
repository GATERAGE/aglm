import os
import gradio as gr
import fire
import argparse
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

from MASTERMIND import MASTERMIND
from logic import LogicTables
from reasoning import SocraticReasoning
from prediction import Predictor
from epistemic import AutoepistemicAgent
from bdi import Belief, Desire, Intention, Goal, Reward
from memory import save_conversation_memory
from aglm import LlamaModel

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

def initialize_model(model_name, model_type):
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    try:
        if model_type in [Model_Type.ggml, Model_Type.gguf]:
            filename = "pytorch_model.bin" if model_type == Model_Type.ggml else "flax_model.msgpack"
            file_path = hf_hub_download(repo_id=model_name, filename=filename, local_dir=models_folder)
            model = Llama(file_path, n_ctx=4096)
            tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Default tokenizer for GGML and GGUF
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name} with type {model_type}: {e}")
        raise
    return model, tokenizer

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def run_ui(model, tokenizer, is_chat_model, model_type, save_history=True):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        conversation_memory = []
        epistemic_agent = AutoepistemicAgent(initial_beliefs={'The sky is blue': True})
        belief = Belief("The sky is blue")  # Initialize Belief

        def user(user_message, memory):
            nonlocal conversation_memory
            conversation_memory.append([user_message, None])

            # Process belief and simulate adding conflicting information
            belief.process_belief()
            epistemic_agent.add_information({'The sky is blue': False})
            epistemic_agent.revise_beliefs()

            current_beliefs = f"Processed Belief: {belief}"
            memory[-1][1] = current_beliefs
            return "", memory

        def bot(memory):
            nonlocal conversation_memory
            conversation_memory = memory
            instruction = memory[-1][0]

            response = model.generate_contextual_output(instruction)

            memory[-1][1] = f"Response: {response}"

            if save_history:
                save_conversation_memory(conversation_memory)

            return memory

        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(bot, inputs=[chatbot], outputs=[chatbot])
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.launch(share=False, debug=True)

def main():
    parser = argparse.ArgumentParser(description="Run the UIUX model")
    parser.add_argument("--model_name", required=True, help="Model name on Hugging Face Hub")
    parser.add_argument("--tokenizer_name", required=True, help="Tokenizer name on Hugging Face Hub")
    parser.add_argument("--model_type", required=True, help="Type of the model, if applicable")
    parser.add_argument("--save_history", action='store_true', help="Flag to determine if history should be saved")
    parser.add_argument("--file_name", required=True, help="File name for saving or loading model specifics")
    
    args = parser.parse_args()
    
    # Initialize tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Example of using the model
    print(f"Model and tokenizer for {args.model_name} loaded successfully.")
    
    if args.save_history:
        print("History will be saved.")

    if args.file_name:
        print(f"Working with file: {args.file_name}")

if __name__ == '__main__':
    fire.Fire(main)
