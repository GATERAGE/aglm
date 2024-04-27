import gradio as gr
from memory import save_conversation_memory
from logic import LogicTables
from reasoning import SocraticReasoning
from prediction import Predictor
from epistemic import AutoepistemicAgent
from bdi import Belief, Desire, Intention, Goal, Reward
from MASTERMIND import MASTERMIND

def run_ui(model, tokenizer, is_chat_model, model_type, save_history=True):
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")
        conversation_memory = []

        def user(user_message, memory):
            nonlocal conversation_memory
            conversation_memory = memory + [[user_message, None]]
            # Here you could integrate LogicTables or SocraticReasoning
            logic_response = LogicTables.process(user_message)
            reasoned_response = SocraticReasoning.evaluate(user_message)
            return logic_response, reasoned_response, conversation_memory

        def bot(memory):
            nonlocal conversation_memory
            # Example of how MASTERMIND could be utilized
            MASTERMIND.process(memory[-1][0])
            # Continue usual processing...

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)
    demo.queue()
    demo.launch(share=False, debug=True)
