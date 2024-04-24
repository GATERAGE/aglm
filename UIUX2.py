import os
import gradio as gr
import fire
from enum import Enum
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from MASTERMIND import MASTERMIND
from logic import LogicTables
from reasoning import SocraticReasoning
from prediction import Predictor
from epistemic import AutoepistemicAgent
# from autonomize import Autonomizer
# from nonmonotonic import NonmonotonicReasoner
# from socratic import SocraticQuestioner
from bdi import BDIModel
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
    models_folder = "./models/"  # Ensure your models are stored here
    aglm_model = LlamaModel(model_name, models_folder)
    return aglm_model.model, aglm_model.tokenizer

def run_ui(model, tokenizer, is_chat_model, model_type, save_history=True):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        conversation_memory = []
        epistemic_agent = AutoepistemicAgent(initial_beliefs={'The sky is blue': True})

        def user(user_message, memory):
            nonlocal conversation_memory
            conversation_memory.append([user_message, None])

            # Simulate user adding conflicting information
            epistemic_agent.add_information({'The sky is blue': False})
            epistemic_agent.revise_beliefs()

            current_beliefs = epistemic_agent.beliefs
            memory[-1][1] = f"Updated beliefs: {current_beliefs}"
            return "", memory

        def bot(memory):
            nonlocal conversation_memory
            conversation_memory = memory
            instruction = memory[-1][0]

            aglm_model = LlamaModel(model_name, "./models/")
            response = aglm_model.generate_contextual_output(instruction)

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
