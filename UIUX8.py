import os
import gradio as gr
import fire
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
    full_precision = 3

def get_model_type(model_name):
    if "gptq" in model_name.lower():
        return Model_Type.gptq
    elif "ggml" in model_name.lower():
        return Model_Type.ggml
    else:
        return Model_Type.full_precision

def initialize_model(model_name, model_type):
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    if model_type == Model_Type.ggml:
        file_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", local_dir=models_folder)
        model = Llama(file_path, n_ctx=4096)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Default tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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

def main(model_name=None, file_name=None, save_history=True):
    assert model_name, "model_name argument is missing."
    model_type = get_model_type(model_name)
    model, tokenizer = initialize_model(model_name, model_type)
    run_ui(model, tokenizer, 'chat' in model_name.lower(), model_type, save_history=save_history)

if __name__ == '__main__':
    fire.Fire(main)
