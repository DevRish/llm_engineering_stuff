from llama_cpp import Llama
from rich import print as rich_print
from rich.prompt import Prompt
from prompts.mistral_samples import classification_prompt_sample, personalization_prompt_sample, summarization_prompt_sample
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

def generate_prompt_response(prompt):
    rich_print(f"[bold magenta]Response:[/bold magenta] ", end="")

    output_stream = llm(prompt, max_tokens=1000, stop=["</s>"], stream=True)
    for chunk in output_stream:
        rich_print(chunk["choices"][0]["text"], end="")

    rich_print("")

demo_choice = Prompt.ask('''
[bold cyan]Available Demos:[/bold cyan][bold cyan3]
1. Bank request classification
2. Essay Summarization
3. Personalized email from Mortgage Lender Customer Service[/bold cyan3]
[bold turquoise4]
Enter choice[/bold turquoise4]''')

rich_print()

if demo_choice == "1":
    user_input = Prompt.ask("[bold blue]Inquiry[/bold blue]")
    # Example:
    # Inquiry: I am inquiring about the availability of your cards in the EU, as I am a resident of France and am interested in using your cards.
    # Expected Response: country support
    prompt = classification_prompt_sample(user_input)
    generate_prompt_response(prompt)
elif demo_choice == "2":
    user_input = Prompt.ask("[bold blue]Essay:[/bold blue] ")
    # Example Essay:
    """
    Penguins are fascinating, flightless birds that primarily inhabit the Southern Hemisphere, with the majority found in and around Antarctica. Unlike most birds, penguins are adapted for life in the water. Their wings have evolved into flippers, allowing them to swim with remarkable speed and agility. These adaptations make them excellent hunters, feeding on fish, squid, and krill. Despite their chilly reputation, not all penguins live in cold climates. Species like the Galápagos penguin thrive near the equator. Penguins are also highly social creatures, often forming large colonies for breeding and protection. Their distinctive black-and-white coloring serves as natural camouflage while swimming, helping them avoid predators like seals and orcas. Overall, penguins are a symbol of adaptability and resilience, surviving in some of the harshest environments on Earth. Their charming waddles and social behavior have made them beloved animals worldwide.
    """
    prompt = summarization_prompt_sample(user_input)
    generate_prompt_response(prompt)
elif demo_choice == "3":
    user_input = Prompt.ask("[bold blue]Customer Email:[/bold blue] ")
    # Example Customer Email:
    """
    Dear mortgage lender, What's your 30-year fixed-rate APR, how is it compared to the 15-year fixed rate? Regards, Anna
    """
    prompt = personalization_prompt_sample(user_input)
    generate_prompt_response(prompt)
else:
    rich_print("[bold red]Invalid Input[/bold red]")

# https://docs.mistral.ai/guides/prompting_capabilities/