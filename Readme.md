### Setup:

```bash
python -m venv venv

source venv/bin/activate

which python
which pip

pip install -r requirements.txt

# Donot forget to `pip freeze > requirements.txt` on installing new libraries
```

### Roadmaps:

- https://roadmap.sh/r/llm-engineer-ay1q6
- https://youtu.be/v1pj9XrJ_Lw?feature=shared

### HuggingFace Repos used:

- Mistral (main LLM): https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main
- BGE (Text Embedding LLM): https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/tree/main

**Basic RAG using Mistral:** https://docs.mistral.ai/guides/rag/

### Notes:

- Using `gguf` format models locally, and loading them with `llama.cpp` for efficiency (to fit my system specs)

- `Qunatization` => Reducing the precision of values / weights stored in models => To reduce model size. Example: `Q4` => All values truncated to 4 decimal places. It definitely does also impact model performance, and `Q6>Q5>Q4` obviously.

- Instantiating LLama:
  - `verbose: False` avoids unnecessary logs
  - `n_ctx` refers to text context. To get best usage out of a model, a context size >= the one which it was trained on should be used. For example, for the `mistral-7b-q4` model I was using, it was `32768`, and when I used values lower than that, it warned me that the model won't perform to it's best potential. Note: Higher values required more RAM. For example, `32768` takes up about 4GB of RAM by itself, on loading the model.
  - `n_gpu_layers` is the number of model layers to be run on gpu (rest run on cpu). For 4GB VRAM GPU, `35-40` is fine
  - `n_threads` is the number of threads. For my cpu, 8 is an optimal value (neither low nor high)
  - `embedding: True` creates embeddings

- `Prompting` notes:
  - Prompt structure and role specification during prompting is done in various formats for various models. It depends on how the model was trained.
  - Mistral, LLaMA, Alpaca and ChatML all use same format for structured dialogues, whose rules are:
    - `user` messages get wrapped in `[INST]...[/INST]` (INST => Instruction)
    - `system` messages get wrapped in `<<SYS>>...<</SYS>>` (SYS => System)
    - `assisstant` responses get wrapped in nothing
  - To better understand prompting capabilities, it is best to consult relevant docs. For mistral: `https://docs.mistral.ai/guides/prompting_capabilities/`

- `Retrieval Augmented Generation (RAG)` notes:
  - The embedding model used during RAG should be `semantically compatible` (extarct same meaning / similar vectors for a given text) with the embedding process used when training the main LLM, for best results. `Mistral` and `BGE` are semantically compatible.
  - Chunk size (in tokens) used during the chunking of context, should not exceed the `n_ctx` (context length, in tokens) of the embedding model that would later be used to embed the chunks into vectors. Otherwise, information will be lost, as the embedding model won't consider the entire chunk.
  - `faiss` stores vectors in-memory. However, the data can be exported to a `.faiss` file, or even a `sqlite` local database.