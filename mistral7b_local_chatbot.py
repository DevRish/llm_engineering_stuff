from llama_cpp import Llama
from rich import print as rich_print
from rich.prompt import Prompt
from utils import download_model_from_hf

model_path = download_model_from_hf(
    model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    save_dir="./models/mistral"
)

# Load GGUF model using llama.cpp
llm = Llama(
    model_path=model_path,
    n_ctx=32768, # higher value, better quality responses, more RAM consumed (32768 takes about 4GB RAM by itself)
    n_gpu_layers=35,  # higher value, more model layers run on GPU instead of CPU (for 4GB VRAM GPU, 35 to 40 layers is fine)
    verbose=False
)

rich_print("[bold green]Mistral Chatbot Ready! Type 'exit' to quit.[/bold green]")

chat_history = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# PROMPTING:
# Mistral, LLaMA, Alpaca and ChatML all use same format for structured dialogues
# Rules:
# - role: "user" gets wrapped in "[INST]...[/INST]" (INST => Instruction)
# - role: "system" gets wrapped in "<<SYS>>...<</SYS>>" (SYS => System)
# - role: "assisstant" gets wrapped in nothing
def format_chat_history(history):
    formatted = ""
    for turn in history:
        if turn["role"] == "user":
            formatted += f"[INST] {turn['content']} [/INST]\n"
        elif turn["role"] == "assistant":
            formatted += f" {turn['content']} \n"
        elif turn["role"] == "system":
            formatted += f"<<SYS>> {turn['content']} <</SYS>>\n"
    return formatted

while True:
    user_input = Prompt.ask("[bold blue]You:[/bold blue] ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append({"role": "user", "content": user_input})
    prompt = format_chat_history(chat_history)
    response_text = ""

    # print(f"\nPROMPT:\n{prompt}")

    rich_print(f"[bold magenta]Bot:[/bold magenta] ", end="")

    output_stream = llm(prompt, max_tokens=300, stop=["</s>"], stream=True)
    for chunk in output_stream:
        rich_print(chunk["choices"][0]["text"], end="")
        response_text += chunk["choices"][0]["text"]
    
    rich_print("")

    chat_history.append({"role": "assistant", "content": response_text})
