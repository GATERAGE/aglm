import os
import gradio as gr
import fire
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from MASTERMIND import MASTERMIND
from logic import LogicTables
from reasoning import SocraticReasoning
from prediction import Predictor
from epistemic import AutoepistemicAgent
from bdi import Belief, Desire, Intention, Goal, Reward  # Import BDI components
from memory import save_conversation_memory

class Model_Type(Enum):
    gptq = 1
    ggml = 2
    full_precision = 3

class LlamaModel:
    def __init__(self, model_name, models_folder="./models"):
        self.model_name = model_name
        self.models_folder = models_folder
        self.model, self.tokenizer = self.initialize_model()

    def initialize_model(self):
        # Check if model_name is a path or a model ID
        if os.path.isdir(self.model_name):
            model_path = self.model_name
        else:
            model_path = hf_hub_download(repo_id=self.model_name, filename="pytorch_model.bin")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

def get_model_type(model_name):
    if "gptq" in model_name.lower():
        return Model_Type.gptq
    elif "ggml" in model_name.lower():
        return Model_Type.ggml
    else:
        return Model_Type.full_precision

def initialize_model(model_name, model_type):
    return LlamaModel(model_name).model, LlamaModel(model_name).tokenizer

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
