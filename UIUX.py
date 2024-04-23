import os
import gradio as gr
import fire
import pathlib
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from MASTERMIND import MASTERMIND
from logic import LogicTables
from reasoning import SocraticReasoning
from prediction import Predictor
from epistemic import KnowledgeBase
from autonomize import Autonomizer
from nonmonotonic import NonmonotonicReasoner
from socratic import SocraticQuestioner
from bdi import BDIModel

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

def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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

            # Example to integrate prediction using updated Predictor class
            predictor = Predictor("models")  # Assuming model directory is "models"
            features_df = None  # Placeholder for actual feature DataFrame
            predictions = predictor.predict(features_df) if features_df is not None else "No features provided"

            memory[-1][1] = f"Predictions: {predictions}"
            return "", memory

        def bot(memory):
            nonlocal conversation_memory
            conversation_memory = memory
            instruction = memory[-1][0]

            try:
                if model_type == Model_Type.ggml:
                    for chunk in model(prompt=instruction, stream=True):
                        memory[-1][1] += chunk["choices"][0]["text"]
                else:
                    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
                    outputs = model.generate(input_ids=inputs["input_ids"])
                    memory[-1][1] += tokenizer.decode(outputs[0], skip_special_tokens=True)

                if save_history:
                    # Placeholder function to save conversation memory
                    pass
            except ValueError as e:
                memory[-1][1] = "Error: Instruction too long to process."

            return memory

        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(bot, inputs=[chatbot], outputs=[chatbot])
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.launch(share=False, debug=True)

def main(model_name=None, file_name=None, save_history=True):
    assert model_name, "model_name argument is missing."
    model_type = get_model_type(model_name)
    model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type)
    run_ui(model, tokenizer, 'chat' in model_name.lower(), model_type, save_history=save_history)

if __name__ == '__main__':
    fire.Fire(main)
