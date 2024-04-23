import os
import gradio as gr
import fire
import time
import pathlib
import ujson
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

# Import MASTERMIND modules
from logic import LogicalOperator
from reasoning import Reasoner
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

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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
            # Integrating MASTERMIND's reasoning and prediction
            reasoner = Reasoner()
            predictor = Predictor(model)
            socratic_questioner = SocraticQuestioner(topic=user_message)

            # Generate reasoning and predictions
            reasoning_output = reasoner.deduce(user_message)
            prediction_output = predictor.forecast(user_message)
            socratic_question = socratic_questioner.generate_question()

            # Update conversation memory with new outputs
            memory[-1][1] = f"Reasoning: {reasoning_output}, Prediction: {prediction_output}, Socratic Question: {socratic_question}"
            return "", memory

        def bot(memory):
            nonlocal conversation_memory
            conversation_memory = memory
            user_input = memory[-1][0]

            # Using LLAMA style or direct inputs based on model type
            instruction = format_to_llama_chat_style(memory) if is_chat_model else user_input

            try:
                if model_type == Model_Type.ggml:
                    # Handle specific generation for LLAMA models
                    for chunk in model(prompt=instruction, stream=True):
                        memory[-1][1] += chunk["choices"][0]["text"]
                else:
                    inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
                    outputs = model.generate(input_ids=inputs["input_ids"])
                    memory[-1][1] += tokenizer.decode(outputs[0], skip_special_tokens=True)

                if save_history:
                    save_conversation_memory(conversation_memory)
            except ValueError as e:
                memory[-1][1] = "Error: Instruction too long to process."

            return memory

        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(bot, inputs=[chatbot], outputs=[chatbot])
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.launch(share=False, debug=True)

def main(model_name=None, file_name=None, save_history=True):
    assert model_name, "model_name argument is missing."
    model_type = get_model_type(model_name)
    if model_type == Model_Type.ggml:
        assert file_name, "When model_name is provided for a GGML quantized model, file_name argument must also be provided."

    model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type)
    run_ui(model, tokenizer, 'chat' in model_name.lower(), model_type, save_history=save_history)

if __name__ == '__main__':
    fire.Fire(main)