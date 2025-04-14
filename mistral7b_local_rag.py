# https://docs.mistral.ai/guides/rag/

import requests
from rich import print as rich_print
from rich.prompt import Prompt

from llama_cpp import Llama
from utils import download_model_from_hf
import numpy as np
import faiss

response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

# chunking based on characters
chunk_size = 512 
# bge model supports max 512 tokens in a chunk. Since I am chunking based on characters, not tokens, I still put the same value
# so that chunk_size_in_tokens <= n_ctx os ensured
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
# rich_print(f'''[bold magenta]Number of chunks:[/bold magenta] {len(chunks)}''')

embedding_llm_path = download_model_from_hf(
    model_id="CompendiumLabs/bge-base-en-v1.5-gguf",
    filename="bge-base-en-v1.5-q4_k_m.gguf",
    save_dir="./models/bge"
)
embedding_llm = Llama(
    model_path=embedding_llm_path,
    embedding=True,
    n_ctx=512,       # max value for bge model
    n_threads=8,     # best for my cpu
    verbose=False
)

llm_path = download_model_from_hf(
    model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    save_dir="./models/mistral"
)
llm = Llama(
    model_path=llm_path,
    n_ctx=32768,
    n_gpu_layers=35,
    verbose=False
)

def get_text_embedding(text: str):
    response = embedding_llm.create_embedding(text)
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

# print(chunks[0])
# print(get_text_embedding(chunks[0]))

text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

rich_print("[bold green]Please read the essay at: https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt[/bold green]")
rich_print("[bold green]Please ask questions based on the essay[/bold green]")

question = Prompt.ask("[bold blue]Question[/bold blue]") 
# Example Questions: 
# - What were the two main things the author worked on before college?
# - Which computer did the author use?
# - Which grad schools did the author apply to? Which one accepted him?

question_embeddings = np.array([get_text_embedding(question)])

# print(question_embeddings)

D, I = index.search(question_embeddings, k=2) # distance, index
retrieved_chunk = [chunks[i] for i in I.tolist()[0]]

prompt = f"""
<<SYS>>
Context information is below.
---------------------
{retrieved_chunk}
---------------------
Given the context information and not prior knowledge, answer the query.
<</SYS>>

[INST]
{question}
[/INST]
"""

# print(prompt)

rich_print(f"[bold magenta]Answer:[/bold magenta] ", end="")
output_stream = llm(prompt, max_tokens=1000, stop=["</s>"], stream=True)
for chunk in output_stream:
    rich_print(chunk["choices"][0]["text"], end="")

rich_print("")