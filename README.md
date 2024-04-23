# aGLM Autonomous General Learning Model<br />
"I Am a Learning Machine"<br /><br />
project codename = aGLM Autonomous General Learning Model<br />
project direction = <a href="https://rage.pythai.net">RAGE</a><br /><br />
"MASTERMIND controller for aGLM with RAGE."<br />
uiux.py provides Gradio output to local server for local language model interaction<br />
Gradio interacts with html javascript and css <a href="https://www.gradio.app/guides/custom-CSS-and-JS">custom Gradio</a><br />
<a href="https://www.gradio.app/guides/getting-started-with-the-js-client">Gradio Javascript Client</a><br />
Documentation: codephreak = uiux.py + memory.py + automind.py + aglm.py<br />

# User Interface and Interaction (uiux.py)<br />

(uiux.py) provides a user interface using the Gradio library to facilitate user interaction.
This chatbot interface takes the user input, processes it, generating response includingc onversation memory handling storing and managing user instructions and model responses. The result is a local language model prompted to run as "Professor Codephreak is an expert in computer programming ....." that refers to itself as "codephreak"
Contextual Conversation Management<br />

(automind.py) provides the mechanism to format and managing conversation history using the format_to_llama_chat_style function.
(automind.py) creates coherent conversation context integrating memory management and with chatbot behavior.
Handling different model types and initialization is based on model name calling the model from the models folder. <br />

(memory.py) handles Conversation Memory Management by  creating a class DialogEntry to represent individual conversation dialog entries.
A function save_conversation_memory is called to save conversation history as JSON files. Memory is created by storing user instructions and model responses in memory files for context management as .json outputs with a timestamp.<br />

# LLAMA Model Interaction<br />
(aglm.py)<br />

Initializes the LLAMA language model and tokenizer based on the specified models_folder models handling processing and tokenization of conversation context using the LLAMA model then generating contextually relevant responses using the LLAMA model. AUTOMIND uses a LLAMA language model to generate responses based on user instructions and conversation history. (uiux.py) encompasses user interface design, memory management, context handling, and interaction with the language model to create a conversational experience for users.<br />

-----------------------------------

## [automindx.install](https://github.com/pythaiml/automindx/blob/main/automindx.install)


To install right click "Save link as ..." [automindx.install](https://github.com/pythaiml/automindx/blob/main/automindx.install)
 chmod +x automind.install && automind.install

details and verbose procedure<br />
instructions tested on Ubuntu 22.04 Linux Mint 21.2 and Mandrake Linux 
1. Right-click the following link: [automindx.install](https://github.com/pythaiml/automindx/blob/main/automindx.install)

2. Choose "Save link as..." or "Download linked file" from the context menu.
3. Select a location on your computer to save the file.
4. from the terminal
5. chmod +x automind.install && ./automind.install<br />

---------------------------------
<br />

# RUN <a href="https://github.com/Professor-Codephreak">Professor Codephreak</a> with a desire to create automindx<br />
auto-downloads model llama2-7b-chat-codeCherryPop-qLoRA-GGML<br />
creates <b>Professor Codephreak</b> an expert in computer science, machine learning and computer programming<br />
prompt includes an agenda to create the automindx deployment environment</b><br />
note: This repo is under development<br />
inputs larger than 4096 characters will crash the input --> response screen<br />

```bash
python3 uiux.py --model_name="TheBloke/llama2-7b-chat-codeCherryPop-qLoRA-GGML" --tokenizer_name="TheBloke/llama2-7b-chat-codeCherryPop-qLoRA-GGML" --model_type="ggml" --save_history --file_name="llama-2-7b-chat-codeCherryPop.ggmlv3.q4_1.bin"
```


