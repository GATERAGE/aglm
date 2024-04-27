import fire
from model_handler import init_auto_model_and_tokenizer, get_model_type
from ui_components import run_ui
from config import MODEL_CONFIG

def main(model_name=None, file_name=None, save_history=True):
    assert model_name is not None, "model_name argument is missing."
    is_chat_model = 'chat' in model_name.lower()
    model_type = get_model_type(model_name)
    model, tokenizer = init_auto_model_and_tokenizer(model_name, model_type, file_name, MODEL_CONFIG)
    run_ui(model, tokenizer, is_chat_model, model_type, save_history=save_history)

if __name__ == '__main__':
    fire.Fire(main)
