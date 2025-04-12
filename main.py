from llama_cpp import Llama
from rich import print as rich_print
from rich.prompt import Prompt

from huggingface_hub import hf_hub_download
import os

def download_gguf_if_needed(model_id, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    local_file = os.path.join(save_dir, filename)

    if not os.path.exists(local_file):
        rich_print(f"Downloading {filename} from {model_id}...")
        return hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
    else:
        rich_print("Model file already exists.")
        return local_file

# Example usage
gguf_path = download_gguf_if_needed(
    model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    save_dir="./models/mistral"
)

rich_print(gguf_path)

# Load model
llm = Llama(
    model_path=gguf_path,
    n_ctx=32768, # for more tokens in output. Max value for the above downloaded model is 32768
    n_gpu_layers=35,  # tweak if you run into CUDA OOM
    verbose=False
)

rich_print("[bold green]Mistral Chatbot Ready! Type 'exit' to quit.[/bold green]")

chat_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def format_chat_history(history):
    formatted = ""
    for turn in history:
        if turn["role"] == "user":
            formatted += f"[INST] {turn['content']} [/INST]\n"
        elif turn["role"] == "assistant":
            formatted += f"{turn['content']}\n"
    return formatted

while True:
    user_input = Prompt.ask("[bold blue]You:[/bold blue] ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append({"role": "user", "content": user_input})
    prompt = format_chat_history(chat_history)
    response_text = ""

    rich_print(f"[bold magenta]Bot:[/bold magenta] ", end="")

    output_stream = llm(prompt, max_tokens=300, stop=["</s>"], stream=True)
    for chunk in output_stream:
        rich_print(chunk["choices"][0]["text"], end="")
        response_text += chunk["choices"][0]["text"]
    
    rich_print("")

    chat_history.append({"role": "assistant", "content": response_text})
