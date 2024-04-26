import os
import gradio as gr
import fire
from enum import Enum
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

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def initialize_model(model_name, model_type):
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    if model_type in [Model_Type.ggml, Model_Type.gguf]:
        filename = "pytorch_model.bin" if model_type == Model_Type.ggml else "flax_model.msgpack"
        file_path = hf_hub_download(repo_id=model_name, filename=filename, local_dir=models_folder)
        model = LlamaModel(model_name, models_folder)
    else:
        print(f"Unsupported model type: {model_type}")
        raise ValueError("Unsupported model type")
    return model

def run_ui(model, is_chat_model, save_history=True):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        conversation_memory = []
        epistemic_agent = AutoepistemicAgent(initial_beliefs={'The sky is blue': True})
        socratic_reasoning = SocraticReasoning()
        logic_tables = LogicTables()
        predictor = Predictor()

        def user(user_message, memory):
            nonlocal conversation_memory
            conversation_memory.append([user_message, None])

            # Process beliefs and reasoning
            epistemic_agent.add_information({'user_message': user_message})
            epistemic_agent.revise_beliefs()
            reasoned_response = socratic_reasoning.apply_reasoning(user_message)
            predicted_outcome = predictor.predict_outcome(user_message)

            response = model.generate_contextual_output(reasoned_response)
            memory[-1][1] = f"Reasoned Response: {response}, Predicted Outcome: {predicted_outcome}"

            if save_history:
                save_conversation_memory(conversation_memory)

            return "", memory

        def bot(memory):
            nonlocal conversation_memory
            conversation_memory = memory
            instruction = memory[-1][0]
            response = model.generate_contextual_output(instruction)
            memory[-1][1] = response

            if save_history:
                save_conversation_memory(conversation_memory)

            return memory

        msg.submit(user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(bot, inputs=[chatbot], outputs=[chatbot])
        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    demo.launch(share=False, debug=True)

def main(model_name, model_type, save_history=False):
    model_type_enum = get_model_type(model_name)
    model = initialize_model(model_name, model_type_enum)
    run_ui(model, 'chat' in model_name.lower(), save_history)

if __name__ == '__main__':
    fire.Fire(main)
